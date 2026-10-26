import unittest

import numpy as np
import torch

from care_ecg.security.constraints import (
    S2Constraint,
    SInfConstraint,
    SPhysConstraint,
    SSpecConstraint,
    make_constraint,
    scale_to_constraint_distance,
)
from care_ecg.security.contracts import ConstraintKind


class SecurityConstraintTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.zeros((12, 500), dtype=np.float32)

    @staticmethod
    def _synthetic_multilead_ecg(
        sample_rate: float = 250.0,
        duration_seconds: float = 6.0,
        rr_seconds: float = 0.60,
    ) -> tuple[np.ndarray, np.ndarray]:
        time = np.arange(int(sample_rate * duration_seconds)) / sample_rate
        r_times = np.arange(0.45, duration_seconds, rr_seconds)
        template = np.zeros_like(time)
        for r_time in r_times:
            template += 0.10 * np.exp(
                -0.5 * ((time - (r_time - 0.15)) / 0.035) ** 2
            )
            template += np.exp(-0.5 * ((time - r_time) / 0.018) ** 2)
            template += 0.28 * np.exp(
                -0.5 * ((time - (r_time + 0.22)) / 0.070) ** 2
            )
        lead_weights = np.asarray([1.0, -0.65, 0.30])[:, None]
        waveform = (lead_weights * template).astype(np.float32)
        return waveform, np.rint(r_times * sample_rate).astype(int)

    def test_linf_projection_sampling_and_distance_use_raw_mv(self):
        constraint = SInfConstraint(epsilon_mv=0.02)
        candidate = self.clean + 0.2

        projected = constraint.project(self.clean, candidate)
        sampled = constraint.sample(self.clean, np.random.default_rng(7))

        self.assertAlmostEqual(constraint.distance(self.clean, projected), 0.02, places=6)
        self.assertTrue(constraint.contains(self.clean, projected))
        self.assertTrue(constraint.contains(self.clean, sampled))
        self.assertEqual(constraint.violations(self.clean, candidate), ("linf_bound",))

    def test_l2_projection_is_per_lead_and_preserves_tensor_backend(self):
        constraint = S2Constraint(epsilon_mv=0.5)
        clean = torch.zeros((2, 10), dtype=torch.float32)
        candidate = torch.ones_like(clean)

        projected = constraint.project(clean, candidate)

        self.assertIsInstance(projected, torch.Tensor)
        np.testing.assert_allclose(
            torch.linalg.vector_norm(projected - clean, dim=1).numpy(),
            [0.5, 0.5],
            rtol=1e-6,
        )
        self.assertTrue(constraint.contains(clean.numpy(), projected.numpy()))
        self.assertAlmostEqual(constraint.distance(clean, projected), 0.5, places=6)

    def test_spectral_projection_enforces_frequency_amplitude_and_slew(self):
        sample_rate = 500.0
        time = np.arange(self.clean.shape[1]) / sample_rate
        low = 0.2 * np.sin(2 * np.pi * 5 * time)
        high = 0.3 * np.sin(2 * np.pi * 100 * time)
        candidate = self.clean + np.tile(low + high, (12, 1)).astype(np.float32)
        constraint = SSpecConstraint(
            epsilon_mv=0.05,
            sample_rate_hz=sample_rate,
            cutoff_hz=40.0,
            max_high_frequency_ratio=0.01,
            max_slew_mv=0.02,
        )

        projected = constraint.project(self.clean, candidate)

        self.assertTrue(constraint.contains(self.clean, projected))
        self.assertLessEqual(constraint.distance(self.clean, projected), 0.05 + 1e-7)
        self.assertIn("high_frequency_energy", constraint.violations(self.clean, candidate))

    def test_physiological_projection_and_sampling_stay_in_parameter_space(self):
        sample_rate = 100.0
        time = np.arange(500) / sample_rate
        clean = np.tile(np.sin(2 * np.pi * 1.25 * time), (3, 1)).astype(np.float32)
        constraint = SPhysConstraint(sample_rate_hz=sample_rate)
        parameters = {
            "p_amplitude_mv": 0.02,
            "qrs_amplitude_mv": -0.04,
            "t_amplitude_mv": 0.01,
            "p_timing_ms": 4.0,
            "qrs_timing_ms": -2.0,
            "t_timing_ms": 8.0,
            "baseline_drift_mv": 0.01,
            "rr_jitter_ms": 3.0,
        }

        synthesized = constraint.synthesize(clean, parameters)
        sampled = constraint.sample(clean, np.random.default_rng(11))
        arbitrary = clean + np.random.default_rng(3).normal(0, 0.4, clean.shape)
        projected = constraint.project(clean, arbitrary)

        self.assertTrue(constraint.contains(clean, synthesized))
        self.assertTrue(constraint.contains(clean, sampled))
        self.assertTrue(constraint.contains(clean, projected))
        self.assertLessEqual(constraint.parameter_distance(clean, projected), 1.0 + 1e-5)
        self.assertIn(
            "off_physiological_parameter_manifold",
            constraint.violations(clean, arbitrary),
        )

    def test_torch_physiological_render_survives_float32_numpy_membership(self):
        rng = np.random.default_rng(19)
        clean = rng.normal(0.0, 1.0, (3, 500)).astype(np.float32)
        constraint = SPhysConstraint(sample_rate_hz=100.0)
        parameters = torch.tensor(
            [0.02, -0.04, 0.01, 4.0, -2.0, 8.0, 0.01, 3.0],
            dtype=torch.float32,
        )

        rendered = constraint.synthesize(torch.from_numpy(clean), parameters)
        serialized = rendered.detach().numpy().astype(np.float32)

        self.assertTrue(constraint.contains(clean, serialized))

    def test_physiological_basis_uses_patient_rr_phase_and_lead_morphology(self):
        sample_rate = 250.0
        clean, expected_r_peaks = self._synthetic_multilead_ecg(sample_rate)
        constraint = SPhysConstraint(
            sample_rate_hz=sample_rate,
            nominal_rr_seconds=0.8,
        )

        profile = constraint.patient_profile(clean)
        detected_r_peaks = np.asarray(profile["r_peak_indices"])
        basis = constraint._numpy_basis(clean)

        self.assertEqual(profile["source"], "consensus_r_peaks")
        self.assertAlmostEqual(profile["rr_seconds"], 0.60, places=2)
        self.assertNotAlmostEqual(profile["rr_seconds"], 0.8, places=2)
        for expected_peak in expected_r_peaks:
            self.assertLessEqual(np.min(np.abs(detected_r_peaks - expected_peak)), 2)

        interior_peak = int(expected_r_peaks[2])
        self.assertGreater(abs(basis[1, 0, interior_peak]), 0.95)
        self.assertLess(basis[1, 1, interior_peak], -0.55)
        self.assertGreater(basis[1, 2, interior_peak], 0.25)
        self.assertFalse(np.allclose(basis[1, 0], basis[1, 1]))
        np.testing.assert_allclose(
            profile["lead_amplitude_weights"]["qrs"],
            (1.0, 0.65, 0.30),
            atol=1e-5,
        )

    def test_patient_specific_physiological_projection_invariants(self):
        sample_rate = 250.0
        clean, _ = self._synthetic_multilead_ecg(sample_rate)
        constraint = SPhysConstraint(sample_rate_hz=sample_rate)
        limits = np.asarray(
            [constraint.bounds[name] for name in constraint.parameter_names],
            dtype=np.float32,
        )
        parameters = limits * np.asarray(
            [0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1],
            dtype=np.float32,
        )

        numpy_rendered = constraint.synthesize(clean, parameters)
        torch_rendered = constraint.synthesize(
            torch.from_numpy(clean), torch.from_numpy(parameters)
        )
        arbitrary = clean + np.random.default_rng(31).normal(
            0.0, 0.4, clean.shape
        ).astype(np.float32)
        projected = constraint.project(clean, arbitrary)
        reprojected = constraint.project(clean, projected)

        self.assertEqual(numpy_rendered.dtype, np.float32)
        self.assertEqual(projected.dtype, np.float32)
        self.assertTrue(constraint.contains(clean, numpy_rendered))
        self.assertTrue(
            constraint.contains(clean, torch_rendered.detach().numpy().astype(np.float32))
        )
        self.assertTrue(constraint.contains(clean, projected))
        self.assertTrue(constraint.contains(clean, reprojected))
        self.assertLessEqual(constraint.parameter_distance(clean, projected), 1.0 + 1e-5)
        np.testing.assert_allclose(reprojected, projected, atol=2e-5, rtol=0.0)

    def test_native_distance_scaling_uses_each_constraint_unit(self):
        sample_rate = 100.0
        time = np.arange(500) / sample_rate
        clean = np.tile(2.0 * np.sin(2 * np.pi * 1.25 * time), (2, 1)).astype(
            np.float32
        )
        constraints_and_targets = (
            (SInfConstraint(0.02), 0.01),
            (S2Constraint(0.5, per_lead=True), 0.2),
            (SSpecConstraint(0.02, sample_rate, cutoff_hz=40.0), 0.01),
            (SPhysConstraint(sample_rate), 0.4),
        )
        for index, (constraint, target) in enumerate(constraints_and_targets):
            with self.subTest(constraint=constraint.kind.value):
                sampled = constraint.sample(clean, np.random.default_rng(100 + index))
                scaled = scale_to_constraint_distance(
                    constraint, clean, sampled, target
                )
                self.assertTrue(constraint.contains(clean, scaled))
                self.assertAlmostEqual(
                    constraint.distance(clean, scaled), target, delta=2e-5
                )

    def test_constraint_factory_uses_declared_kind(self):
        constraint = make_constraint(ConstraintKind.S_INF, epsilon_mv=0.01)
        self.assertIsInstance(constraint, SInfConstraint)


if __name__ == "__main__":
    unittest.main()
