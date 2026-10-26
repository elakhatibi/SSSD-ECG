import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from care_ecg.causal import BayesianCausalModel
from care_ecg.ensemble import (
    CARE_LABELS,
    LinearProbabilityStacker,
    ProbabilityTable,
    build_stacking_features,
    sha256_file,
)
from care_ecg.finalization import (
    build_locked_expert_artifact,
    parse_named_probability_paths,
)


class _PipelineStub:
    def __init__(self, causal_model: BayesianCausalModel):
        self.causal_model = causal_model

    def save(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True)
        self.causal_model.save(destination / "causal_graph.json")
        (destination / "manifest.json").write_text(
            '{"format_version": 1, "framework": "CARE-ECG"}\n',
            encoding="utf-8",
        )


class LockedExpertFinalizationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, object]:
        base = root / "base"
        base.mkdir()
        causal = BayesianCausalModel(CARE_LABELS, ["z0"], 2).fit(
            np.asarray([[0], [0], [1], [1], [0]]),
            np.arange(5),
        )
        causal.neural_weight = 0.4
        causal.class_biases = np.asarray([0.0, 0.2, -0.1, 0.3, 0.1])
        causal.save(base / "causal_graph.json")
        (base / "knowledge_graph.json").write_text('{"graph": true}\n', encoding="utf-8")
        (base / "training_history.json").write_text(
            '{"history": true}\n', encoding="utf-8"
        )

        record_ids = np.asarray(["r1", "r2"])
        source_a = ProbabilityTable(
            np.asarray([[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1]]),
            record_ids,
            CARE_LABELS,
        )
        source_b = ProbabilityTable(
            np.asarray([[0.2, 0.2, 0.2, 0.2, 0.2], [0.1, 0.1, 0.5, 0.2, 0.1]]),
            record_ids,
            CARE_LABELS,
        )
        sources = [("primary", source_a), ("secondary", source_b)]
        _, feature_names, source_names = build_stacking_features(sources, record_ids)
        weights = np.zeros((10, 5))
        weights[np.arange(5), np.arange(5)] = 1.0
        stacker = LinearProbabilityStacker(
            mean=np.zeros(10),
            scale=np.ones(10),
            weights=weights,
            biases=np.zeros(5),
            class_biases=np.zeros(5),
            feature_names=feature_names,
            source_names=source_names,
        )
        stacker_path = root / "locked_stacker.npz"
        stacker.save(stacker_path)

        source_paths: list[tuple[str, Path]] = []
        for name, table in sources:
            path = root / f"{name}.npz"
            np.savez_compressed(
                path,
                probabilities=table.probabilities,
                record_ids=table.record_ids,
                label_order=np.asarray(table.label_order),
            )
            source_paths.append((name, path))
        expert_path = root / "expert.npz"
        np.savez_compressed(
            expert_path,
            probabilities=stacker.predict_sources(sources, record_ids),
            record_ids=record_ids,
            label_order=np.asarray(CARE_LABELS),
        )
        report_path = root / "locked_selection.json"
        report_path.write_text(
            json.dumps(
                {
                    "scope": "validation only",
                    "source_order": list(source_names),
                    "stacker_sha256": sha256_file(stacker_path),
                }
            ),
            encoding="utf-8",
        )
        return {
            "base": base,
            "pipeline": _PipelineStub(causal),
            "stacker": stacker_path,
            "report": report_path,
            "expert": expert_path,
            "sources": source_paths,
        }

    def _build(self, fixture: dict[str, object], output: Path, **changes):
        arguments = {
            "base_artifact": fixture["base"],
            "output_artifact": output,
            "locked_stacker": fixture["stacker"],
            "selection_report": fixture["report"],
            "expert_probabilities": fixture["expert"],
            "source_probabilities": fixture["sources"],
            "paper_artifact_deviation": "Uses a documented replacement ensemble.",
        }
        arguments.update(changes)
        with mock.patch(
            "care_ecg.finalization.CAREECGPipeline.load",
            return_value=fixture["pipeline"],
        ):
            return build_locked_expert_artifact(**arguments)

    def test_builds_weight_one_zero_offset_artifact_with_strict_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            output = root / "final"

            manifest = self._build(fixture, output)

            graph = BayesianCausalModel.load(output / "causal_graph.json")
            self.assertEqual(graph.neural_weight, 1.0)
            np.testing.assert_array_equal(graph.class_biases, np.zeros(5))
            self.assertEqual(manifest["expert_weight"], 1.0)
            self.assertEqual(manifest["internal_care_weight"], 0.0)
            self.assertEqual(manifest["source_order"], ["primary", "secondary"])
            self.assertTrue(manifest["contract_validation"]["stack_recomputed_from_sources"])
            self.assertEqual(
                manifest["selection_report"]["sha256"],
                sha256_file(fixture["report"]),
            )
            self.assertEqual(
                (output / "knowledge_graph.json").read_bytes(),
                (fixture["base"] / "knowledge_graph.json").read_bytes(),
            )
            self.assertEqual(
                (output / "training_history.json").read_bytes(),
                (fixture["base"] / "training_history.json").read_bytes(),
            )
            persisted = json.loads((output / "external_expert.json").read_text())
            self.assertEqual(persisted, manifest)
            for entry in persisted["source_probabilities"]:
                self.assertTrue(Path(entry["path"]).is_absolute())
                self.assertGreater(entry["size_bytes"], 0)
                self.assertEqual(len(entry["sha256"]), 64)

    def test_rejects_test_named_expert_before_lock_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            test_named = root / "test_probabilities.npz"
            Path(fixture["expert"]).replace(test_named)

            with self.assertRaisesRegex(PermissionError, "allow-test-after-lock"):
                self._build(
                    fixture,
                    root / "final",
                    expert_probabilities=test_named,
                )

            manifest = self._build(
                fixture,
                root / "final",
                expert_probabilities=test_named,
                allow_test_after_lock=True,
            )
            self.assertEqual(manifest["record_count"], 2)

    def test_rejects_mismatched_expert_and_nonempty_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            output = root / "final"
            output.mkdir()
            (output / "keep.txt").write_text("do not replace", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                self._build(fixture, output)
            self.assertEqual((output / "keep.txt").read_text(), "do not replace")

            (output / "keep.txt").unlink()
            output.rmdir()
            expert_path = Path(fixture["expert"])
            with np.load(expert_path, allow_pickle=False) as archive:
                probabilities = np.asarray(archive["probabilities"])
                record_ids = np.asarray(archive["record_ids"])
                labels = np.asarray(archive["label_order"])
            probabilities[0] = np.asarray([0.2] * 5)
            np.savez_compressed(
                expert_path,
                probabilities=probabilities,
                record_ids=record_ids,
                label_order=labels,
            )
            with self.assertRaisesRegex(ValueError, "do not match the locked stacker"):
                self._build(fixture, output)
            self.assertFalse(output.exists())

    def test_named_probability_parser_preserves_order_and_rejects_duplicates(self):
        parsed = parse_named_probability_paths(["a=one.npz", "b=two.npz"])
        self.assertEqual(parsed, [("a", Path("one.npz")), ("b", Path("two.npz"))])
        with self.assertRaisesRegex(ValueError, "uniquely named"):
            parse_named_probability_paths(["a=one.npz", "a=two.npz"])


if __name__ == "__main__":
    unittest.main()
