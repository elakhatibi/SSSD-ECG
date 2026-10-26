import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from care_ecg.ensemble import sha256_file
from scripts.export_care_probabilities import (
    _artifact_tree_sha256,
    export_care_probabilities,
    parse_args,
)


class _TrackingArchive:
    def __init__(self, values):
        self.values = values
        self.accessed = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __contains__(self, name):
        return name in self.values

    def __getitem__(self, name):
        self.accessed.append(name)
        if name == "labels":
            raise AssertionError("labels must not be opened")
        return self.values[name]


class _PipelineStub:
    def __init__(self, probabilities):
        self.causal_model = SimpleNamespace(label_names=["NORM", "MI"])
        self.probabilities = probabilities
        self.call = None

    def encode_batch_with_neural(self, signals, source_rate=None, batch_size=64):
        self.call = (signals.copy(), source_rate, batch_size)
        return np.zeros((len(signals), 3), dtype=np.float32), self.probabilities


class CAREProbabilityExportTests(unittest.TestCase):
    def _artifact(self, root: Path) -> Path:
        artifact = root / "artifact"
        artifact.mkdir()
        (artifact / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
        (artifact / "causal_vae.pt").write_bytes(b"checkpoint")
        (artifact / "manifest.json").write_text('{"format_version": 1}\n', encoding="utf-8")
        return artifact

    def test_exports_strict_provenance_without_opening_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = self._artifact(root)
            split = root / "validation.npz"
            output = root / "care-probabilities.npz"
            signals = np.arange(48, dtype=np.float32).reshape(2, 2, 12)
            record_ids = np.asarray(["ecg-2", "ecg-1"])
            folds = np.asarray([9, 9], dtype=np.int16)
            archive = _TrackingArchive(
                {
                    "signals": signals,
                    "record_ids": record_ids,
                    "folds": folds,
                    "labels": np.asarray([1, 0]),
                }
            )
            pipeline = _PipelineStub(np.asarray([[0.75, 0.25], [0.1, 0.9]]))

            with mock.patch(
                "scripts.export_care_probabilities.np.load", return_value=archive
            ), mock.patch(
                "scripts.export_care_probabilities.CAREECGPipeline.load",
                return_value=pipeline,
            ):
                resolved = export_care_probabilities(
                    artifact=artifact,
                    prepared_split=split,
                    expected_fold=9,
                    output=output,
                    batch_size=7,
                    preprocessed=True,
                )

            self.assertEqual(archive.accessed, ["signals", "record_ids", "folds"])
            self.assertEqual(resolved, output.resolve())
            np.testing.assert_array_equal(pipeline.call[0], signals)
            self.assertIsNone(pipeline.call[1])
            self.assertEqual(pipeline.call[2], 7)
            with np.load(output, allow_pickle=False) as exported:
                np.testing.assert_array_equal(exported["record_ids"], record_ids)
                np.testing.assert_array_equal(exported["folds"], folds)
                np.testing.assert_array_equal(exported["label_order"], ["NORM", "MI"])
                np.testing.assert_allclose(exported["probabilities"].sum(axis=1), 1.0)
                self.assertEqual(str(exported["artifact_sha256"]), _artifact_tree_sha256(artifact))
                self.assertEqual(str(exported["config_sha256"]), sha256_file(artifact / "config.yaml"))
                self.assertEqual(
                    str(exported["model_sha256"]), sha256_file(artifact / "causal_vae.pt")
                )
                self.assertTrue(bool(exported["deterministic_latent_sampling"]))
                self.assertEqual(int(exported["encoder_batch_size"]), 7)
                self.assertIn("labels not opened", str(exported["inference_contract"]))

    def test_raw_signals_pass_source_rate_to_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = self._artifact(root)
            split = root / "validation.npz"
            np.savez(
                split,
                signals=np.zeros((1, 12, 100), dtype=np.float32),
                record_ids=np.asarray(["1"]),
                folds=np.asarray([9]),
                labels=np.asarray([0]),
            )
            pipeline = _PipelineStub(np.asarray([[0.4, 0.6]]))
            with mock.patch(
                "scripts.export_care_probabilities.CAREECGPipeline.load",
                return_value=pipeline,
            ):
                export_care_probabilities(
                    artifact=artifact,
                    prepared_split=split,
                    expected_fold=9,
                    output=root / "probabilities.npz",
                    batch_size=3,
                    source_rate=100,
                )
            self.assertEqual(pipeline.call[1:], (100, 3))

    def test_fold_ten_and_test_named_paths_are_guarded_before_opening(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = self._artifact(root)
            with mock.patch("scripts.export_care_probabilities.np.load") as loader, mock.patch(
                "scripts.export_care_probabilities.CAREECGPipeline.load"
            ) as pipeline_loader:
                with self.assertRaisesRegex(PermissionError, "allow-test-after-lock"):
                    export_care_probabilities(
                        artifact=artifact,
                        prepared_split=root / "heldout.npz",
                        expected_fold=10,
                        output=root / "probabilities.npz",
                        preprocessed=True,
                    )
                with self.assertRaisesRegex(PermissionError, "allow-test-after-lock"):
                    export_care_probabilities(
                        artifact=artifact,
                        prepared_split=root / "test-records.npz",
                        expected_fold=9,
                        output=root / "probabilities.npz",
                        preprocessed=True,
                    )
                loader.assert_not_called()
                pipeline_loader.assert_not_called()

    def test_rejects_duplicate_ids_missing_classifier_and_unnormalized_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = self._artifact(root)
            duplicate_split = root / "validation.npz"
            np.savez(
                duplicate_split,
                signals=np.zeros((2, 12, 10), dtype=np.float32),
                record_ids=np.asarray(["same", "same"]),
                folds=np.asarray([9, 9]),
            )
            with self.assertRaisesRegex(ValueError, "must be unique"):
                export_care_probabilities(
                    artifact=artifact,
                    prepared_split=duplicate_split,
                    expected_fold=9,
                    output=root / "duplicate-output.npz",
                    preprocessed=True,
                )

            valid_split = root / "validation-unique.npz"
            np.savez(
                valid_split,
                signals=np.zeros((1, 12, 10), dtype=np.float32),
                record_ids=np.asarray(["one"]),
                folds=np.asarray([9]),
            )
            for probabilities, message in (
                (None, "no classifier head"),
                (np.asarray([[0.2, 0.2]]), "mass"),
            ):
                with self.subTest(message=message), mock.patch(
                    "scripts.export_care_probabilities.CAREECGPipeline.load",
                    return_value=_PipelineStub(probabilities),
                ), self.assertRaisesRegex(ValueError, message):
                    export_care_probabilities(
                        artifact=artifact,
                        prepared_split=valid_split,
                        expected_fold=9,
                        output=root / f"{message.replace(' ', '-')}.npz",
                        preprocessed=True,
                    )

    def test_parser_requires_an_explicit_preprocessing_mode(self):
        common = [
            "--artifact",
            "artifact",
            "--prepared-split",
            "validation.npz",
            "--expected-fold",
            "9",
            "--output",
            "probabilities.npz",
        ]
        with self.assertRaises(SystemExit):
            parse_args(common)
        with self.assertRaises(SystemExit):
            parse_args([*common, "--preprocessed", "--source-rate", "100"])
        self.assertTrue(parse_args([*common, "--preprocessed"]).preprocessed)


if __name__ == "__main__":
    unittest.main()
