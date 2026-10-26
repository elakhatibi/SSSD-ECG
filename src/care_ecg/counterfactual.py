from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .causal import BayesianCausalModel
from .types import CounterfactualResult


class CounterfactualEngine:
    """Paper-specified exhaustive single-variable, discrete minimal-edit search."""

    def __init__(self, model: BayesianCausalModel):
        self.model = model

    def search(
        self,
        states: Sequence[int] | Mapping[str, int],
        target_label: str | int,
        candidate_features: Sequence[str] | None = None,
        neural_probabilities: Sequence[float] | None = None,
        posterior_scope: str = "reported_fused_posterior",
    ) -> CounterfactualResult:
        if posterior_scope not in {"reported_fused_posterior", "bayesian_component"}:
            raise ValueError("Unknown counterfactual posterior scope")
        evidence = self.model.evidence(states)
        if isinstance(target_label, str):
            if target_label not in self.model.label_names:
                raise ValueError(f"Unknown target diagnosis: {target_label}")
            target_index = self.model.label_names.index(target_label)
        else:
            target_index = int(target_label)
        if target_index < 0 or target_index >= len(self.model.label_names):
            raise ValueError("Target diagnosis index is out of range")
        target_name = self.model.label_names[target_index]
        posterior = (
            self.model.posterior_array
            if posterior_scope == "bayesian_component"
            else lambda value: self.model.fused_posterior_array(value, neural_probabilities)
        )
        before_array = posterior(evidence)
        before = {name: float(before_array[i]) for i, name in enumerate(self.model.label_names)}
        if int(np.argmax(before_array)) == target_index:
            return CounterfactualResult(
                found=True,
                target_label=target_name,
                feature=None,
                old_state=None,
                new_state=None,
                posterior_before=before,
                posterior_after=before.copy(),
                edit_distance=0,
                message=f"{target_name} is already the most probable diagnosis; no edit is required.",
                scope=posterior_scope,
            )
        features = list(
            [
                name
                for name in self.model._check().parents[self.model.target]
                if name in evidence
            ]
            if candidate_features is None
            else candidate_features
        )
        invalid = set(features) - set(self.model.feature_names)
        if invalid:
            raise ValueError(f"Unknown candidate factors: {sorted(invalid)}")
        if not features:
            return CounterfactualResult(
                found=False,
                target_label=target_name,
                feature=None,
                old_state=None,
                new_state=None,
                posterior_before=before,
                posterior_after=before.copy(),
                edit_distance=0,
                message="No candidate factors were supplied for counterfactual search.",
                scope=posterior_scope,
            )

        achieving: list[tuple[int, float, str, int, np.ndarray]] = []
        improvements: list[tuple[float, int, str, int, np.ndarray]] = []
        for feature in features:
            old = evidence[feature]
            for new_state in self.model.supported_states(feature):
                if new_state == old:
                    continue
                edited = {**evidence, feature: new_state}
                edited_posterior = posterior(edited)
                distance = abs(new_state - old)
                improvements.append(
                    (
                        float(edited_posterior[target_index]),
                        -distance,
                        feature,
                        new_state,
                        edited_posterior,
                    )
                )
                if int(np.argmax(edited_posterior)) == target_index:
                    achieving.append(
                        (
                            distance,
                            -float(edited_posterior[target_index]),
                            feature,
                            new_state,
                            edited_posterior,
                        )
                    )

        if achieving:
            distance, _, feature, new_state, after_array = min(achieving)
            message = (
                f"Changing {feature} from state {evidence[feature]} to {new_state} makes "
                f"{target_name} the most probable diagnosis."
            )
            found = True
        elif improvements and max(improvements)[0] > before_array[target_index] + 1e-12:
            _, neg_distance, feature, new_state, after_array = max(improvements)
            distance = -neg_distance
            message = (
                f"No single-factor edit makes {target_name} most probable; the returned edit "
                "produces the largest target posterior found."
            )
            found = False
        else:
            return CounterfactualResult(
                found=False,
                target_label=target_name,
                feature=None,
                old_state=None,
                new_state=None,
                posterior_before=before,
                posterior_after=before.copy(),
                edit_distance=0,
                message=(
                    "No empirically supported single-factor edit improves the target "
                    "probability under the "
                    + (
                        "Bayesian-network component."
                        if posterior_scope == "bayesian_component"
                        else "reported fused model."
                    )
                ),
                scope=posterior_scope,
            )
        after = {name: float(after_array[i]) for i, name in enumerate(self.model.label_names)}
        return CounterfactualResult(
            found=found,
            target_label=target_name,
            feature=feature,
            old_state=evidence[feature],
            new_state=new_state,
            posterior_before=before,
            posterior_after=after,
            edit_distance=1,
            message=message,
            scope=posterior_scope,
        )
