import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from care_ecg.ensemble import (
    CARE_LABELS,
    LinearProbabilityStacker,
    ProbabilityTable,
    build_scp_meta_features,
    build_stacking_features,
    load_hash_verified_legacy_array,
    normalize_probability_rows,
)


class EnsembleContractTests(unittest.TestCase):
    def test_scp_feature_order_is_deterministic(self):
        scp_labels = ("normal", "infarct", "st", "block", "hyper")
        classes = dict(zip(scp_labels, CARE_LABELS, strict=True))
        probabilities = np.asarray([[0.9, 0.2, 0.3, 0.4, 0.1]])

        features, names = build_scp_meta_features(probabilities, scp_labels, classes)

        self.assertEqual(features.shape, (1, 30))
        self.assertEqual(
            names[:5],
            tuple(f"scp_probability:{label}" for label in scp_labels),
        )
        self.assertEqual(
            names[5:10],
            ("NORM:sum", "NORM:max", "NORM:mean", "NORM:noisy_or", "NORM:max_logit"),
        )
        self.assertEqual(names[-1], "HYP:max_logit")

    def test_stacking_features_preserve_declared_source_order_and_exact_ids(self):
        record_ids = np.asarray(["r1", "r2"])
        first = ProbabilityTable(
            normalize_probability_rows(np.asarray([[4, 2, 1, 1, 2], [1, 2, 3, 2, 2]])),
            record_ids,
            CARE_LABELS,
        )
        second = ProbabilityTable(
            np.full((2, 5), 0.2),
            record_ids.copy(),
            CARE_LABELS,
        )

        features, names, sources = build_stacking_features(
            [("primary", first), ("secondary", second)], record_ids
        )

        self.assertEqual(features.shape, (2, 10))
        self.assertEqual(sources, ("primary", "secondary"))
        self.assertEqual(names[0], "log_probability:primary:NORM")
        self.assertEqual(names[5], "log_probability:secondary:NORM")
        np.testing.assert_allclose(np.exp(features[:, :5]), first.probabilities)

        misaligned = ProbabilityTable(
            second.probabilities,
            record_ids[::-1],
            CARE_LABELS,
        )
        with self.assertRaisesRegex(ValueError, "not exactly aligned"):
            build_stacking_features(
                [("primary", first), ("secondary", misaligned)], record_ids
            )

    def test_legacy_pickle_is_deserialized_only_after_hash_verification(self):
        values = np.asarray([[0.1, 0.9]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npy"
            path.write_bytes(pickle.dumps(values, protocol=2))
            expected = hashlib.sha256(path.read_bytes()).hexdigest()

            loaded = load_hash_verified_legacy_array(path, expected, expected_columns=2)
            np.testing.assert_allclose(loaded, values)

            with mock.patch("care_ecg.ensemble.pickle.load") as unsafe_loader:
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    load_hash_verified_legacy_array(path, "0" * 64, expected_columns=2)
                unsafe_loader.assert_not_called()

    def test_probability_rows_are_normalized_and_zero_mass_is_rejected(self):
        normalized = normalize_probability_rows(np.asarray([[2.0, 2.0], [1.0, 3.0]]))
        np.testing.assert_allclose(normalized, [[0.5, 0.5], [0.25, 0.75]])
        np.testing.assert_allclose(normalized.sum(axis=1), 1.0)
        with self.assertRaisesRegex(ValueError, "no positive probability mass"):
            normalize_probability_rows(np.asarray([[0.0, 0.0]]))

    def test_saved_stacker_inference_checks_contract_and_normalizes_output(self):
        record_ids = np.asarray(["r1", "r2"])
        primary_probabilities = normalize_probability_rows(
            np.asarray([[4, 2, 1, 1, 2], [1, 2, 3, 2, 2]])
        )
        primary = ProbabilityTable(primary_probabilities, record_ids, CARE_LABELS)
        secondary = ProbabilityTable(np.full((2, 5), 0.2), record_ids, CARE_LABELS)
        sources = [("primary", primary), ("secondary", secondary)]
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

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stacker.npz"
            stacker.save(path)
            restored = LinearProbabilityStacker.load(path)
            predicted = restored.predict_sources(sources, record_ids)

        np.testing.assert_allclose(predicted, primary_probabilities)
        np.testing.assert_allclose(predicted.sum(axis=1), 1.0)
        bad = LinearProbabilityStacker(
            mean=np.full(10, np.nan),
            scale=np.ones(10),
            weights=weights,
            biases=np.zeros(5),
            class_biases=np.zeros(5),
            feature_names=feature_names,
            source_names=source_names,
        )
        with self.assertRaisesRegex(ValueError, "parameters must be finite"):
            bad.predict_sources(sources, record_ids)


if __name__ == "__main__":
    unittest.main()
