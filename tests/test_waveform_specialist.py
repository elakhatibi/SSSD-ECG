import tempfile
import unittest
from pathlib import Path

import numpy as np

from care_ecg.ensemble import CARE_LABELS
from care_ecg.waveform_specialist import (
    HypSpecialistPolicy,
    MulticlassSpecialistPolicy,
    apply_hyp_logit_adjustment,
    apply_multiclass_logit_adjustment,
    encode_demographics,
    extract_waveform_features,
    fit_demographic_medians,
    load_inference_inputs,
    specialist_feature_names,
)


class WaveformSpecialistTests(unittest.TestCase):
    def test_feature_width_and_order_are_fixed(self) -> None:
        time = np.arange(1000, dtype=np.float64) / 100.0
        signal = np.sin(2.0 * np.pi * 5.0 * time)
        signals = np.tile(signal, (2, 12, 1)).astype(np.float32)

        features = extract_waveform_features(signals, batch_size=1)
        names = specialist_feature_names()

        self.assertEqual(features.shape, (2, 252))
        self.assertEqual(len(names), 260)
        self.assertEqual(names[0], "mean:lead_1")
        self.assertEqual(names[11], "mean:lead_12")
        self.assertEqual(names[12], "std:lead_1")
        self.assertEqual(names[251], "fft_fraction_40_50_hz:lead_12")
        self.assertEqual(names[252], "demographic:age")
        self.assertEqual(names[-1], "demographic_missing:weight")

    def test_demographic_imputation_uses_fit_rows_only(self) -> None:
        train = np.asarray(((20.0, 0.0, 160.0, 50.0), (80.0, 1.0, 180.0, 90.0)))
        validation = np.full((1, 4), np.nan)

        medians = fit_demographic_medians(train)
        encoded = encode_demographics(validation, medians)

        np.testing.assert_allclose(medians, (50.0, 0.5, 170.0, 70.0))
        np.testing.assert_allclose(encoded[0, :4], medians)
        np.testing.assert_array_equal(encoded[0, 4:], np.ones(4))

    def test_adjusted_probabilities_are_normalized(self) -> None:
        baseline = np.asarray(((0.2, 0.1, 0.3, 0.25, 0.15), (1.0, 2.0, 3.0, 4.0, 5.0)))
        adjusted = apply_hyp_logit_adjustment(
            baseline,
            np.asarray((0.8, 0.01)),
            beta=0.2,
            bias=0.175,
        )

        self.assertTrue(np.isfinite(adjusted).all())
        self.assertTrue((adjusted >= 0.0).all())
        np.testing.assert_allclose(adjusted.sum(axis=1), 1.0)

    def test_policy_round_trip(self) -> None:
        policy = HypSpecialistPolicy(
            beta=0.2,
            bias=0.175,
            demographic_medians=np.asarray((50.0, 0.5, 170.0, 70.0)),
            model_sha256="a" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.npz"
            policy.save(path)
            restored = HypSpecialistPolicy.load(path)

        self.assertEqual(restored.beta, policy.beta)
        self.assertEqual(restored.bias, policy.bias)
        self.assertEqual(restored.model_sha256, policy.model_sha256)
        self.assertEqual(restored.random_state, policy.random_state)
        np.testing.assert_array_equal(
            restored.demographic_medians,
            policy.demographic_medians,
        )

    def test_multiclass_policy_round_trip_and_probability_normalization(self) -> None:
        policy = MulticlassSpecialistPolicy(
            beta=0.075,
            biases=np.asarray((0.0, 0.15, 0.15, 0.15, -0.15)),
            specialist_logit_center=np.arange(5, dtype=np.float64),
            specialist_logit_scale=np.arange(1, 6, dtype=np.float64),
            demographic_medians=np.asarray((50.0, 0.5, 170.0, 70.0)),
            model_sha256="b" * 64,
            transform="zscore",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multiclass_policy.npz"
            policy.save(path)
            restored = MulticlassSpecialistPolicy.load(path)

        self.assertEqual(restored.transform, "zscore")
        self.assertEqual(restored.model_sha256, policy.model_sha256)
        np.testing.assert_array_equal(restored.biases, policy.biases)
        adjusted = apply_multiclass_logit_adjustment(
            np.full((2, len(CARE_LABELS)), 1.0 / len(CARE_LABELS)),
            np.asarray(((0.2, 0.3, 0.4, 0.5, 0.6), (0.8, 0.7, 0.6, 0.5, 0.4))),
            beta=restored.beta,
            biases=restored.biases,
            specialist_logit_center=restored.specialist_logit_center,
            specialist_logit_scale=restored.specialist_logit_scale,
        )
        self.assertTrue(np.isfinite(adjusted).all())
        self.assertTrue((adjusted >= 0.0).all())
        np.testing.assert_allclose(adjusted.sum(axis=1), 1.0)

    def test_inference_loader_does_not_access_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference.npz"
            np.savez(
                path,
                signals=np.zeros((2, 12, 10), dtype=np.float32),
                record_ids=np.asarray(("1", "2")),
                labels=np.asarray(({"target": 1}, {"target": 2}), dtype=object),
            )

            signals, record_ids = load_inference_inputs(path)

        self.assertEqual(signals.shape, (2, 12, 10))
        np.testing.assert_array_equal(record_ids, ("1", "2"))


if __name__ == "__main__":
    unittest.main()
