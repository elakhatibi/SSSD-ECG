import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from care_ecg.agents import DiagnosisAgent, HistoryAgent
from care_ecg.causal import BayesianCausalModel
from care_ecg.cli import _sha256_file, build_parser
from care_ecg.expert import blend_external_expert, load_aligned_expert_probabilities
from care_ecg.pipeline import CAREECGPipeline
from care_ecg.types import ResponseResult


class ExternalExpertTests(unittest.TestCase):
    def test_cli_hashes_expert_artifacts_for_prediction_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert.bin"
            path.write_bytes(b"care-ecg")

            self.assertEqual(
                _sha256_file(path),
                "7e84678feb1b980ad1767b027f5e8b9d72f2c75180e84389899b940b1d12be03",
            )

    def test_loader_reorders_columns_after_exact_record_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert.npz"
            np.savez(
                path,
                probabilities=np.asarray([[0.8, 0.2], [0.1, 0.9]]),
                record_ids=np.asarray(["r1", "r2"]),
                label_order=np.asarray(["MI", "NORM"]),
            )

            result = load_aligned_expert_probabilities(
                path,
                np.asarray(["r1", "r2"]),
                ["NORM", "MI"],
            )

        np.testing.assert_allclose(result, [[0.2, 0.8], [0.9, 0.1]])

    def test_loader_rejects_missing_or_misaligned_record_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert.npz"
            np.savez(
                path,
                probabilities=np.asarray([[0.8, 0.2], [0.1, 0.9]]),
                record_ids=np.asarray(["r2", "r1"]),
                label_order=np.asarray(["NORM", "MI"]),
            )

            with self.assertRaisesRegex(ValueError, "requires record_ids"):
                load_aligned_expert_probabilities(path, None, ["NORM", "MI"])
            with self.assertRaisesRegex(ValueError, "first mismatch at row 0"):
                load_aligned_expert_probabilities(
                    path,
                    np.asarray(["r1", "r2"]),
                    ["NORM", "MI"],
                )

    def test_loader_rejects_missing_contract_array_and_label_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.npz"
            np.savez(
                missing_path,
                probabilities=np.asarray([[0.8, 0.2]]),
                label_order=np.asarray(["NORM", "MI"]),
            )
            with self.assertRaisesRegex(ValueError, "record_ids"):
                load_aligned_expert_probabilities(
                    missing_path,
                    np.asarray(["r1"]),
                    ["NORM", "MI"],
                )

            wrong_labels_path = Path(directory) / "wrong-labels.npz"
            np.savez(
                wrong_labels_path,
                probabilities=np.asarray([[0.8, 0.2]]),
                record_ids=np.asarray(["r1"]),
                label_order=np.asarray(["NORM", "AFIB"]),
            )
            with self.assertRaisesRegex(ValueError, "missing=.*MI.*extra=.*AFIB"):
                load_aligned_expert_probabilities(
                    wrong_labels_path,
                    np.asarray(["r1"]),
                    ["NORM", "MI"],
                )

    def test_loader_rejects_invalid_probability_values_and_mass(self):
        cases = (
            (np.asarray([[np.nan, 0.0]]), "finite"),
            (np.asarray([[1.1, -0.1]]), "nonnegative"),
            (np.asarray([[0.3, 0.3]]), "mass equal to 1"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (probabilities, message) in enumerate(cases):
                path = Path(directory) / f"invalid-{index}.npz"
                np.savez(
                    path,
                    probabilities=probabilities,
                    record_ids=np.asarray(["r1"]),
                    label_order=np.asarray(["NORM", "MI"]),
                )
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    load_aligned_expert_probabilities(
                        path,
                        np.asarray(["r1"]),
                        ["NORM", "MI"],
                    )

    def test_blend_requires_complete_audit_fields_and_valid_weight(self):
        internal = np.asarray([0.8, 0.2])
        expert = np.asarray([0.2, 0.8])

        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            blend_external_expert(internal, expert, None, "checkpoint", ["NORM", "MI"])
        for weight in (-0.01, 1.01, np.nan):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                blend_external_expert(
                    internal,
                    expert,
                    weight,
                    "checkpoint",
                    ["NORM", "MI"],
                )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            blend_external_expert(internal, expert, 0.5, "  ", ["NORM", "MI"])

    def test_pipeline_blends_before_fusion_and_records_audit_metadata(self):
        model = BayesianCausalModel(["NORM", "MI"], ["z0"], 2).fit(
            np.asarray([[0], [0], [1], [1]]),
            np.asarray([0, 0, 1, 1]),
        )
        model.neural_weight = 1.0
        pipeline = object.__new__(CAREECGPipeline)
        pipeline.config = SimpleNamespace(generation=SimpleNamespace(backend="template"))
        pipeline.causal_model = model
        pipeline.discretizer = SimpleNamespace(transform=lambda _: np.asarray([0]))
        pipeline.history_agent = HistoryAgent(model.feature_names)
        pipeline.diagnosis_agent = DiagnosisAgent(model)
        pipeline.rag_agent = SimpleNamespace(retrieve=lambda query, diagnosis: [])
        pipeline.response_agent = SimpleNamespace(
            respond=lambda query, diagnosis, facts, history: ResponseResult(
                explanation="Evidence-only test response.",
                hallucination_score=0.0,
                used_fallback=False,
                raw_with_causal="Evidence-only test response.",
            )
        )
        pipeline.encode_with_neural = lambda signal, source_rate=None: (
            np.asarray([0.0]),
            np.asarray([0.8, 0.2]),
        )

        output = pipeline.run(
            np.zeros((1, 1)),
            "diagnosis?",
            external_expert_probabilities=np.asarray([0.2, 0.8]),
            external_expert_weight=0.75,
            external_expert_source="checkpoint-sha256:abc",
        )

        self.assertEqual(output.diagnosis, "MI")
        np.testing.assert_allclose(list(output.probabilities.values()), [0.35, 0.65])
        audit = output.metadata["diagnosis_fusion"]["external_expert"]
        self.assertEqual(audit["source"], "checkpoint-sha256:abc")
        self.assertEqual(audit["weight"], 0.75)
        self.assertEqual(audit["blended_neural_probabilities"], output.probabilities)

    def test_default_pipeline_metadata_has_no_external_expert_block(self):
        internal, audit = blend_external_expert(
            np.asarray([0.8, 0.2]),
            None,
            None,
            None,
            ["NORM", "MI"],
        )

        np.testing.assert_array_equal(internal, [0.8, 0.2])
        self.assertIsNone(audit)

    def test_predict_parser_exposes_expert_contract_options(self):
        args = build_parser().parse_args(
            [
                "predict",
                "artifacts",
                "dataset.npz",
                "predictions.jsonl",
                "--expert-probabilities",
                "expert.npz",
                "--expert-weight",
                "0.25",
                "--expert-source",
                "checkpoint:abc",
            ]
        )

        self.assertEqual(args.expert_probabilities, "expert.npz")
        self.assertEqual(args.expert_weight, 0.25)
        self.assertEqual(args.expert_source, "checkpoint:abc")


if __name__ == "__main__":
    unittest.main()
