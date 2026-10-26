import unittest
from unittest.mock import patch

import numpy as np
import torch

from care_ecg.causal import BayesianCausalModel
from care_ecg.config import CAREConfig
from care_ecg.discretization import QuantileDiscretizer
from care_ecg.models import CausalVAE, InceptionTimeBackbone
from care_ecg.pipeline import CAREECGPipeline
from care_ecg.retrieval import CausalRetriever, KnowledgeDocument
from care_ecg.training import (
    CausalVAETrainer,
    ECGAugmenter,
    apply_class_biases,
    balanced_class_weights,
    select_class_biases,
    select_probability_fusion,
    set_seed,
)


class AdvancedTrainingTests(unittest.TestCase):
    def _config(self) -> CAREConfig:
        config = CAREConfig()
        config.device = "cpu"
        config.data.leads = 2
        config.model.hidden_dim = 8
        config.model.latent_dim = 4
        config.model.lstm_layers = 1
        config.model.classifier_hidden_dim = 6
        config.training.batch_size = 2
        return config

    def _model(self, config: CAREConfig, num_classes: int = 3) -> CausalVAE:
        return CausalVAE(
            leads=config.data.leads,
            latent_dim=config.model.latent_dim,
            hidden_dim=config.model.hidden_dim,
            backbone=config.model.backbone,
            lstm_layers=config.model.lstm_layers,
            dropout=config.model.dropout,
            num_classes=num_classes,
            classifier_hidden_dim=config.model.classifier_hidden_dim,
            inception_kernel_size=config.model.inception_kernel_size,
            classifier_input=config.model.classifier_input,
        )

    def test_resnet_joint_loss_updates_encoder_and_classifier(self):
        config = self._config()
        model = self._model(config)
        signals = torch.randn(3, 2, 128)
        labels = torch.tensor([0, 1, 2])

        output = model(signals)
        losses = model.loss(
            output,
            signals,
            beta=0.001,
            labels=labels,
            classification_weight=1.0,
            label_smoothing=0.05,
        )
        losses["loss"].backward()

        self.assertEqual(output.logits.shape, (3, 3))
        self.assertIn("classification", losses)
        self.assertIsNotNone(model.mu_head.weight.grad)
        self.assertIsNotNone(model.classifier[-1].weight.grad)

    def test_model_constructor_remains_compatible_with_old_checkpoint_config(self):
        model = CausalVAE(
            leads=2,
            latent_dim=4,
            hidden_dim=8,
            backbone="bilstm",
            lstm_layers=1,
            dropout=0.1,
        )

        self.assertEqual(model.num_classes, 0)
        self.assertIsNone(model.classifier)
        self.assertNotIn("logits", model.loss(model(torch.zeros(1, 2, 64)), torch.zeros(1, 2, 64)))

    def test_inceptiontime_encoder_has_multiscale_residual_stack(self):
        config = self._config()
        config.model.backbone = "inceptiontime"
        config.model.lstm_layers = 3
        config.model.classifier_input = "encoder"
        config.validate()
        model = self._model(config)
        signals = torch.randn(2, 2, 128)

        output = model(signals)
        output.logits.sum().backward()

        self.assertEqual(output.reconstruction.shape, signals.shape)
        self.assertEqual(output.logits.shape, (2, 3))
        self.assertIsInstance(model.encoder.temporal, InceptionTimeBackbone)
        self.assertEqual(len(model.encoder.temporal.shortcuts), 1)
        self.assertIsNotNone(model.encoder.stem[0].weight.grad)
        self.assertIsNone(model.mu_head.weight.grad)

    def test_inceptiontime_checkpoint_configuration_round_trips(self):
        model = CausalVAE(
            leads=2,
            latent_dim=4,
            hidden_dim=8,
            backbone="inceptiontime",
            lstm_layers=3,
            num_classes=3,
            classifier_hidden_dim=6,
            inception_kernel_size=15,
            classifier_input="hybrid",
        )

        restored = CausalVAE(**model.init_config)
        restored.load_state_dict(model.state_dict())

        self.assertEqual(restored.init_config, model.init_config)
        self.assertEqual(restored(torch.randn(2, 2, 64)).logits.shape, (2, 3))

    def test_inceptiontime_configuration_rejects_invalid_shape_settings(self):
        config = self._config()
        config.model.backbone = "inceptiontime"
        config.model.lstm_layers = 2
        with self.assertRaisesRegex(ValueError, "at least three"):
            config.validate()

        config.model.lstm_layers = 3
        config.model.hidden_dim = 10
        with self.assertRaisesRegex(ValueError, "divisible by four"):
            config.validate()

        config.model.hidden_dim = 8
        config.model.inception_kernel_size = 10
        with self.assertRaisesRegex(ValueError, "odd integer"):
            config.validate()

    def test_encoder_classifier_is_not_available_through_latent_only_api(self):
        config = self._config()
        config.model.classifier_input = "encoder"
        model = self._model(config)

        with self.assertRaisesRegex(RuntimeError, "classification_logits"):
            model.classify_latent(torch.zeros(1, config.model.latent_dim))

    def test_batched_pipeline_encoding_matches_single_record_encoding(self):
        config = self._config()
        config.data.sample_rate = 32
        config.data.seconds = 2
        config.data.normalization = "robust"
        model = self._model(config, num_classes=2)
        rng = np.random.default_rng(9)
        reference_latent = rng.normal(size=(20, config.model.latent_dim))
        discretizer = QuantileDiscretizer(3).fit(reference_latent)
        discrete = discretizer.transform(reference_latent)
        labels = np.arange(len(discrete)) % 2
        causal_model = BayesianCausalModel(
            ["NORM", "MI"], discretizer.feature_names, 3
        ).fit(discrete, labels, max_parents=1)
        retriever = CausalRetriever().fit(
            [KnowledgeDocument("fact", "ECG evidence requires clinical review.", "test")]
        )
        pipeline = CAREECGPipeline(
            config,
            model,
            discretizer,
            causal_model,
            retriever,
        )
        signals = rng.normal(size=(3, config.data.leads, config.data.samples)).astype(np.float32)

        batch_latent, batch_probabilities = pipeline.encode_batch_with_neural(
            signals, batch_size=2
        )
        single = [pipeline.encode_with_neural(signal) for signal in signals]

        np.testing.assert_allclose(batch_latent, np.stack([item[0] for item in single]), atol=1e-6)
        np.testing.assert_allclose(
            batch_probabilities,
            np.stack([item[1] for item in single]),
            atol=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            pipeline.encode_batch_with_neural(signals, batch_size=0)

    def test_balanced_class_weights_upweight_minority(self):
        weights = balanced_class_weights(np.asarray([0, 0, 0, 1]), num_classes=2)

        self.assertGreater(weights[1], weights[0])
        np.testing.assert_allclose(weights, [2 / 3, 2.0])

    def test_augmentation_is_deterministic_and_configurable(self):
        config = self._config()
        config.training.augmentation_probability = 1.0
        config.training.noise_std = 0.02
        config.training.amplitude_scale = 0.1
        config.training.time_shift_fraction = 0.05
        config.training.lead_dropout_probability = 0.1
        config.training.baseline_wander_std = 0.02
        batch = torch.linspace(-1.0, 1.0, 256).reshape(2, 2, 64)
        augmenter = ECGAugmenter(config)

        set_seed(17)
        first = augmenter(batch)
        set_seed(17)
        second = augmenter(batch)

        torch.testing.assert_close(first, second)
        self.assertEqual(first.shape, batch.shape)
        self.assertTrue(torch.isfinite(first).all())
        self.assertFalse(torch.equal(first, batch))

        config.training.augmentation_probability = 0.0
        self.assertIs(ECGAugmenter(config)(batch), batch)

    def test_early_stopping_uses_validation_macro_f1_and_records_diagnostics(self):
        config = self._config()
        config.training.epochs = 5
        config.training.early_stopping_patience = 1
        model = self._model(config, num_classes=2)
        trainer = CausalVAETrainer(model, config)
        signals = np.zeros((8, 2, 64), dtype=np.float32)
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        train_metrics = {"loss": 1.0, "reconstruction": 0.5, "kl": 0.1, "macro_f1": 0.4}
        best_validation = {
            "loss": 0.9,
            "reconstruction": 0.5,
            "kl": 0.1,
            "macro_f1": 0.6,
        }
        worse_validation = {
            "loss": 0.8,
            "reconstruction": 0.5,
            "kl": 0.1,
            "macro_f1": 0.5,
        }

        with patch.object(
            trainer,
            "_epoch",
            side_effect=[train_metrics, best_validation, train_metrics, worse_validation],
        ):
            history = trainer.fit(signals, labels=labels, validation_fraction=0.25)

        self.assertTrue(history.diagnostics["early_stopped"])
        self.assertEqual(history.diagnostics["best_epoch"], 1)
        self.assertEqual(history.diagnostics["epochs_completed"], 2)
        self.assertEqual(history.diagnostics["best_validation_metrics"]["macro_f1"], 0.6)
        self.assertEqual(len(history.diagnostics["class_weights"]), 2)

    def test_cosine_scheduler_records_decreasing_learning_rates(self):
        config = self._config()
        config.training.epochs = 2
        config.training.lr_scheduler = "cosine"
        config.training.minimum_learning_rate = 1e-5
        config.training.augmentation_probability = 0.0
        config.training.early_stopping_patience = 0
        model = self._model(config, num_classes=2)
        trainer = CausalVAETrainer(model, config)
        signals = np.zeros((8, 2, 64), dtype=np.float32)
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

        history = trainer.fit(signals, labels=labels, validation_fraction=0.25)

        self.assertEqual(history.diagnostics["lr_scheduler"], "cosine")
        self.assertEqual(len(history.diagnostics["learning_rates"]), 2)
        self.assertGreater(
            history.diagnostics["learning_rates"][0],
            history.diagnostics["learning_rates"][1],
        )

    def test_batched_classifier_probabilities(self):
        config = self._config()
        model = self._model(config, num_classes=3)
        trainer = CausalVAETrainer(model, config)
        signals = np.zeros((5, 2, 64), dtype=np.float32)

        probabilities = trainer.predict_proba(signals, batch_size=2)
        predicted = trainer.predict(signals, batch_size=2)

        self.assertEqual(probabilities.shape, (5, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_array_equal(predicted, probabilities.argmax(axis=1))

    def test_supervised_external_validation_requires_labels(self):
        config = self._config()
        config.training.epochs = 1
        trainer = CausalVAETrainer(self._model(config, num_classes=2), config)
        signals = np.zeros((4, 2, 64), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "validation_labels"):
            trainer.fit(
                signals,
                labels=np.asarray([0, 0, 1, 1]),
                validation_signals=signals,
            )

    def test_probability_fusion_selects_best_macro_f1_and_lower_weight_on_tie(self):
        labels = np.asarray([0, 1, 2])
        bn = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        neural = np.eye(3) * 0.9 + (1.0 - np.eye(3)) * 0.05

        weight, probabilities, diagnostics = select_probability_fusion(bn, neural, labels)

        self.assertEqual(weight, 0.75)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), labels)
        self.assertEqual(diagnostics["selection_split"], "validation")
        self.assertEqual(diagnostics["selected_metrics"]["macro_f1"], 1.0)
        self.assertEqual(len(diagnostics["candidates"]), 5)

    def test_probability_fusion_breaks_macro_f1_tie_with_accuracy(self):
        labels = np.asarray([0, 0, 0, 1, 1, 2])
        bn_predictions = np.asarray([0, 0, 1, 0, 2, 1])
        neural_predictions = np.zeros(len(labels), dtype=int)
        bn = np.eye(3)[bn_predictions]
        neural = np.eye(3)[neural_predictions]

        weight, _, diagnostics = select_probability_fusion(
            bn, neural, labels, candidate_weights=(0.0, 1.0)
        )

        self.assertEqual(
            diagnostics["candidates"][0]["macro_f1"],
            diagnostics["candidates"][1]["macro_f1"],
        )
        self.assertGreater(
            diagnostics["candidates"][1]["accuracy"],
            diagnostics["candidates"][0]["accuracy"],
        )
        self.assertEqual(weight, 1.0)

    def test_probability_fusion_prefers_bn_when_predictions_are_identical(self):
        probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9]])

        weight, fused, diagnostics = select_probability_fusion(
            probabilities,
            probabilities,
            np.asarray([0, 1]),
        )

        self.assertEqual(weight, 0.0)
        np.testing.assert_allclose(fused, probabilities)
        self.assertEqual(diagnostics["tie_breakers"][-1], "lower_neural_weight")

    def test_probability_fusion_rejects_shape_and_weight_errors(self):
        probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9]])

        with self.assertRaisesRegex(ValueError, "identical shapes"):
            select_probability_fusion(probabilities, np.ones((2, 3)), np.asarray([0, 1]))
        with self.assertRaisesRegex(ValueError, "candidate_weights"):
            select_probability_fusion(
                probabilities,
                probabilities,
                np.asarray([0, 1]),
                candidate_weights=(-0.1, 0.5),
            )

    def test_class_bias_selection_uses_validation_objective_and_anchor(self):
        probabilities = np.asarray(
            [
                [0.9, 0.1],
                [0.85, 0.15],
                [0.8, 0.2],
                [0.75, 0.25],
                [0.6, 0.4],
                [0.6, 0.4],
            ]
        )
        labels = np.asarray([0, 0, 0, 0, 1, 1])

        biases, calibrated, diagnostics = select_class_biases(probabilities, labels)

        self.assertEqual(biases[0], 0.0)
        self.assertGreater(biases[1], 0.0)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
        self.assertEqual(diagnostics["selection_split"], "validation")
        self.assertEqual(diagnostics["objective"], "macro_f1")
        self.assertGreater(
            diagnostics["selected_metrics"]["macro_f1"],
            diagnostics["initial_metrics"]["macro_f1"],
        )
        self.assertEqual(diagnostics["selected_class_biases"], biases.tolist())

    def test_class_bias_application_is_identity_for_zero_biases(self):
        probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9]])

        calibrated = apply_class_biases(probabilities, [0.0, 0.0])

        np.testing.assert_array_equal(calibrated, probabilities)

    def test_class_bias_selection_rejects_invalid_search_settings(self):
        probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9]])
        labels = np.asarray([0, 1])

        with self.assertRaisesRegex(ValueError, "anchor_class"):
            select_class_biases(probabilities, labels, anchor_class=2)
        with self.assertRaisesRegex(ValueError, "step_sizes"):
            select_class_biases(probabilities, labels, step_sizes=())
        with self.assertRaisesRegex(ValueError, "max_abs_bias"):
            select_class_biases(probabilities, labels, max_abs_bias=0.0)


if __name__ == "__main__":
    unittest.main()
