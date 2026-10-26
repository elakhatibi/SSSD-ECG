import unittest

import numpy as np

from care_ecg.security.contracts import ClinicalTarget, Measurements, MeasurementStatus
from care_ecg.security.measurements import (
    ClinicalAdmissibilityChecker,
    NumpyMeasurementBackend,
    ScipyMeasurementBackend,
    assert_independent_backends,
    measure_with_backends,
)


def synthetic_ecg(scale: float = 1.0) -> np.ndarray:
    sample_rate = 500
    time = np.arange(sample_rate * 10) / sample_rate
    waveform = np.zeros((12, len(time)), dtype=float)
    for beat in np.arange(0.4, 9.8, 0.8):
        waveform += 0.08 * np.exp(-0.5 * ((time - beat + 0.16) / 0.025) ** 2)
        waveform += 1.00 * np.exp(-0.5 * ((time - beat) / 0.015) ** 2)
        waveform += 0.25 * np.exp(-0.5 * ((time - beat - 0.24) / 0.060) ** 2)
    return (waveform * scale).astype(np.float32)


class SecurityMeasurementTests(unittest.TestCase):
    def test_available_status_requires_values_and_unavailable_rejects_values(self):
        with self.assertRaises(ValueError):
            Measurements("backend", MeasurementStatus.AVAILABLE)
        with self.assertRaises(ValueError):
            Measurements(
                "backend",
                MeasurementStatus.UNAVAILABLE,
                values={"qrs_ms": 90.0},
                reason="not measured",
            )

    def test_independent_backends_measure_synthetic_ecg(self):
        waveform = synthetic_ecg()
        first = NumpyMeasurementBackend()
        second = ScipyMeasurementBackend()
        assert_independent_backends(first, second)

        results = measure_with_backends(waveform, 500.0, [first, second])

        self.assertEqual(len(results.available), 2)
        for measured in results.available:
            self.assertAlmostEqual(measured.values["heart_rate_bpm"], 75.0, places=1)
            self.assertIn("qtc_ms", measured.values)

    def test_flat_signal_is_explicitly_unavailable(self):
        waveform = np.zeros((12, 5000), dtype=np.float32)

        for backend in (NumpyMeasurementBackend(), ScipyMeasurementBackend()):
            measured = backend.measure(waveform, 500.0)
            self.assertEqual(measured.status, MeasurementStatus.UNAVAILABLE)
            self.assertFalse(measured.available)
            self.assertTrue(measured.reason)

    def test_admissibility_uses_reference_measurements_and_rejects_unavailable(self):
        checker = ClinicalAdmissibilityChecker(
            backend=ScipyMeasurementBackend(),
            tolerances={
                "heart_rate_bpm": 1.0,
                "rhythm_irregularity": 0.01,
                "pr_ms": 2.0,
                "qrs_ms": 2.0,
                "qtc_ms": 2.0,
                "st_deviation_mv": 0.01,
            },
        )
        clean = synthetic_ecg()
        target = ClinicalTarget("T1", "MI", "NORM", "normal ECG", "omission")

        preserved = checker.evaluate(
            clean,
            clean.copy(),
            500.0,
            patient_id="p1",
            record_id="r1",
            target=target,
        )
        unavailable = checker.evaluate(clean, np.zeros_like(clean), 500.0)

        self.assertTrue(preserved.admissible)
        self.assertEqual(preserved.status, MeasurementStatus.AVAILABLE)
        self.assertFalse(preserved.primary_endpoint_eligible)
        self.assertEqual(preserved.metadata["endpoint_classification"], "exploratory")
        self.assertEqual(preserved.metadata["case_context"]["target"]["target_id"], "T1")
        self.assertFalse(unavailable.admissible)
        self.assertEqual(unavailable.status, MeasurementStatus.UNAVAILABLE)

    def test_independence_guard_rejects_same_implementation(self):
        first = NumpyMeasurementBackend(name="first")
        second = NumpyMeasurementBackend(name="second")
        with self.assertRaises(ValueError):
            assert_independent_backends(first, second)


if __name__ == "__main__":
    unittest.main()
