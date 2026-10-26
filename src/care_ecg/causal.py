from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from .types import CausalDriver


@dataclass
class Factor:
    variables: list[str]
    values: np.ndarray

    def condition(self, evidence: Mapping[str, int]) -> Factor:
        variables = list(self.variables)
        values = self.values
        for variable, state in evidence.items():
            if variable not in variables:
                continue
            axis = variables.index(variable)
            if state < 0 or state >= values.shape[axis]:
                raise ValueError(f"State {state} is invalid for {variable}")
            values = np.take(values, state, axis=axis)
            variables.pop(axis)
        return Factor(variables, np.asarray(values, dtype=float))

    def multiply(self, other: Factor, cardinalities: Mapping[str, int]) -> Factor:
        union = list(self.variables) + [v for v in other.variables if v not in self.variables]

        def align(factor: Factor) -> np.ndarray:
            present = [v for v in union if v in factor.variables]
            if present:
                axes = [factor.variables.index(v) for v in present]
                value = np.transpose(factor.values, axes)
            else:
                value = factor.values
            shape = [cardinalities[v] if v in factor.variables else 1 for v in union]
            return np.asarray(value).reshape(shape)

        return Factor(union, align(self) * align(other))

    def sum_out(self, variable: str) -> Factor:
        if variable not in self.variables:
            return self
        axis = self.variables.index(variable)
        variables = list(self.variables)
        variables.pop(axis)
        return Factor(variables, self.values.sum(axis=axis))


class DiscreteBayesianNetwork:
    """A compact categorical Bayesian network with exact variable elimination."""

    def __init__(
        self,
        variables: Sequence[str],
        cardinalities: Mapping[str, int],
        parents: Mapping[str, Sequence[str]],
    ):
        self.variables = list(variables)
        self.cardinalities = {v: int(cardinalities[v]) for v in self.variables}
        self.parents = {v: list(parents.get(v, [])) for v in self.variables}
        self.cpts: dict[str, np.ndarray] = {}
        self._validate_dag()

    def _validate_dag(self) -> None:
        index = {v: i for i, v in enumerate(self.variables)}
        for child, parents in self.parents.items():
            if child not in index:
                raise ValueError(f"Unknown child node {child}")
            if len(set(parents)) != len(parents):
                raise ValueError(f"Duplicate parents for {child}")
            for parent in parents:
                if parent not in index:
                    raise ValueError(f"Unknown parent node {parent}")
                if index[parent] >= index[child]:
                    raise ValueError("variables must be in topological order")

    def fit(self, data: np.ndarray, alpha: float = 1.0) -> DiscreteBayesianNetwork:
        x = np.asarray(data, dtype=np.int64)
        if x.ndim != 2 or x.shape[1] != len(self.variables):
            raise ValueError("Data shape must be [rows, number of variables]")
        if alpha <= 0:
            raise ValueError("Bayesian smoothing alpha must be positive")
        column = {name: i for i, name in enumerate(self.variables)}
        for name in self.variables:
            card = self.cardinalities[name]
            parents = self.parents[name]
            shape = (card,) + tuple(self.cardinalities[p] for p in parents)
            counts = np.full(shape, alpha, dtype=float)
            for row in x:
                state = int(row[column[name]])
                parent_states = tuple(int(row[column[p]]) for p in parents)
                if state < 0 or state >= card:
                    raise ValueError(f"Observed state outside cardinality for {name}")
                counts[(state,) + parent_states] += 1.0
            self.cpts[name] = counts / counts.sum(axis=0, keepdims=True)
        return self

    def posterior(self, query: str, evidence: Mapping[str, int] | None = None) -> np.ndarray:
        if query not in self.cardinalities:
            raise KeyError(query)
        evidence = dict(evidence or {})
        unknown = set(evidence) - set(self.variables)
        if unknown:
            raise KeyError(f"Unknown evidence variables: {sorted(unknown)}")
        if query in evidence:
            answer = np.zeros(self.cardinalities[query], dtype=float)
            answer[evidence[query]] = 1.0
            return answer
        if set(self.cpts) != set(self.variables):
            raise RuntimeError("Network CPTs have not been fitted")

        factors = [
            Factor([name] + self.parents[name], self.cpts[name]).condition(evidence)
            for name in self.variables
        ]
        eliminate = [v for v in reversed(self.variables) if v != query and v not in evidence]
        for variable in eliminate:
            selected = [factor for factor in factors if variable in factor.variables]
            factors = [factor for factor in factors if variable not in factor.variables]
            if not selected:
                continue
            product = selected[0]
            for factor in selected[1:]:
                product = product.multiply(factor, self.cardinalities)
            factors.append(product.sum_out(variable))
        product = factors[0]
        for factor in factors[1:]:
            product = product.multiply(factor, self.cardinalities)
        for variable in list(product.variables):
            if variable != query:
                product = product.sum_out(variable)
        values = np.asarray(product.values, dtype=float).reshape(-1)
        total = values.sum()
        return values / total if total > 0 else np.full_like(values, 1.0 / len(values))

    def shortest_path(self, source: str, target: str) -> list[str]:
        children = {v: [] for v in self.variables}
        for child, parents in self.parents.items():
            for parent in parents:
                children[parent].append(child)
        queue: list[tuple[str, list[str]]] = [(source, [source])]
        visited = {source}
        while queue:
            node, path = queue.pop(0)
            if node == target:
                return path
            for child in children[node]:
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
        return []

    def to_dict(self) -> dict:
        return {
            "variables": self.variables,
            "cardinalities": self.cardinalities,
            "parents": self.parents,
            "cpts": {name: value.tolist() for name, value in self.cpts.items()},
        }

    @classmethod
    def from_dict(cls, values: dict) -> DiscreteBayesianNetwork:
        result = cls(values["variables"], values["cardinalities"], values["parents"])
        result.cpts = {
            name: np.asarray(value, dtype=float) for name, value in values["cpts"].items()
        }
        return result


class K2StructureLearner:
    """Greedy K2 DAG search with optional required clinical-prior edges."""

    def __init__(self, max_parents: int = 3, equivalent_sample_size: float = 5.0):
        if max_parents < 0 or equivalent_sample_size <= 0:
            raise ValueError("Invalid K2 hyperparameters")
        self.max_parents = max_parents
        self.ess = equivalent_sample_size

    def _local_score(
        self,
        data: np.ndarray,
        node: int,
        parents: Sequence[int],
        cardinalities: Sequence[int],
    ) -> float:
        r = cardinalities[node]
        parent_cards = [cardinalities[p] for p in parents]
        q = int(np.prod(parent_cards)) if parents else 1
        counts = np.zeros((q, r), dtype=np.int64)
        if parents:
            multipliers = np.cumprod([1] + parent_cards[::-1])[:-1][::-1]
            configurations = (data[:, parents] * multipliers).sum(axis=1)
        else:
            configurations = np.zeros(len(data), dtype=np.int64)
        np.add.at(counts, (configurations, data[:, node]), 1)
        alpha_ij = self.ess / q
        alpha_ijk = self.ess / (q * r)
        score = np.sum(gammaln(alpha_ij) - gammaln(alpha_ij + counts.sum(axis=1)))
        score += np.sum(gammaln(alpha_ijk + counts) - gammaln(alpha_ijk))
        return float(score)

    def learn(
        self,
        data: np.ndarray,
        variables: Sequence[str],
        cardinalities: Mapping[str, int],
        required_edges: Iterable[tuple[str, str]] = (),
        forbidden_edges: Iterable[tuple[str, str]] = (),
    ) -> dict[str, list[str]]:
        x = np.asarray(data, dtype=np.int64)
        index = {name: i for i, name in enumerate(variables)}
        cards = [int(cardinalities[name]) for name in variables]
        required = list(required_edges)
        forbidden = set(forbidden_edges)
        parents: dict[str, list[str]] = {name: [] for name in variables}
        for parent, child in required:
            if parent not in index or child not in index:
                raise ValueError(f"Clinical prior edge references unknown node: {(parent, child)}")
            if index[parent] >= index[child]:
                raise ValueError(f"Clinical prior edge violates K2 order: {(parent, child)}")
            parents[child].append(parent)
        for position, child in enumerate(variables):
            chosen = list(dict.fromkeys(parents[child]))
            if len(chosen) > self.max_parents:
                raise ValueError(f"Required parents exceed max_parents for {child}")
            best = self._local_score(x, position, [index[p] for p in chosen], cards)
            candidates = [
                variables[i]
                for i in range(position)
                if variables[i] not in chosen and (variables[i], child) not in forbidden
            ]
            while candidates and len(chosen) < self.max_parents:
                scored = [
                    (
                        self._local_score(
                            x, position, [index[p] for p in chosen + [candidate]], cards
                        ),
                        candidate,
                    )
                    for candidate in candidates
                ]
                candidate_score, candidate = max(scored)
                if candidate_score <= best + 1e-9:
                    break
                chosen.append(candidate)
                candidates.remove(candidate)
                best = candidate_score
            parents[child] = chosen
        return parents


class BayesianCausalModel:
    def __init__(self, label_names: Sequence[str], feature_names: Sequence[str], n_bins: int):
        if len(label_names) < 2:
            raise ValueError("At least two diagnosis labels are required")
        self.label_names = list(label_names)
        self.feature_names = list(feature_names)
        self.n_bins = int(n_bins)
        self.target = "diagnosis"
        self.neural_weight = 0.0
        self.class_biases = np.zeros(len(self.label_names), dtype=float)
        self.network: DiscreteBayesianNetwork | None = None
        self.state_support: dict[str, list[int]] = {
            name: list(range(self.n_bins)) for name in self.feature_names
        }

    def fit(
        self,
        discrete_latents: np.ndarray,
        labels: np.ndarray,
        max_parents: int = 3,
        equivalent_sample_size: float = 5.0,
        alpha: float = 1.0,
        clinical_prior_edges: Iterable[tuple[str, str]] = (),
    ) -> BayesianCausalModel:
        z = np.asarray(discrete_latents, dtype=np.int64)
        y = np.asarray(labels, dtype=np.int64).reshape(-1, 1)
        if z.ndim != 2 or z.shape[1] != len(self.feature_names) or len(z) != len(y):
            raise ValueError("Latent/label dimensions are inconsistent")
        if y.min() < 0 or y.max() >= len(self.label_names):
            raise ValueError("Labels must be contiguous indices into label_names")
        variables = self.feature_names + [self.target]
        cardinalities = {name: self.n_bins for name in self.feature_names}
        cardinalities[self.target] = len(self.label_names)
        data = np.concatenate([z, y], axis=1)
        self.state_support = {
            name: [int(value) for value in np.unique(z[:, index])]
            for index, name in enumerate(self.feature_names)
        }
        parents = K2StructureLearner(max_parents, equivalent_sample_size).learn(
            data, variables, cardinalities, clinical_prior_edges
        )
        self.network = DiscreteBayesianNetwork(variables, cardinalities, parents).fit(data, alpha)
        return self

    def _check(self) -> DiscreteBayesianNetwork:
        if self.network is None:
            raise RuntimeError("Causal model has not been fitted")
        return self.network

    def evidence(self, states: Sequence[int] | Mapping[str, int]) -> dict[str, int]:
        if isinstance(states, Mapping):
            return {name: int(states[name]) for name in self.feature_names if name in states}
        if len(states) != len(self.feature_names):
            raise ValueError("Evidence length does not match feature names")
        return {name: int(value) for name, value in zip(self.feature_names, states)}

    def posterior_array(self, states: Sequence[int] | Mapping[str, int]) -> np.ndarray:
        return self._check().posterior(self.target, self.evidence(states))

    def posterior(self, states: Sequence[int] | Mapping[str, int]) -> dict[str, float]:
        probabilities = self.posterior_array(states)
        return {label: float(probabilities[i]) for i, label in enumerate(self.label_names)}

    def fused_posterior_array(
        self,
        states: Sequence[int] | Mapping[str, int],
        neural_probabilities: Sequence[float] | None = None,
    ) -> np.ndarray:
        causal = self.posterior_array(states)
        if neural_probabilities is None or self.neural_weight <= 0:
            fused = causal
        else:
            neural = np.asarray(neural_probabilities, dtype=float)
            if (
                neural.shape != causal.shape
                or not np.all(np.isfinite(neural))
                or np.any(neural < 0)
            ):
                raise ValueError("Neural probabilities must be finite and match diagnosis classes")
            total = neural.sum()
            if total <= 0:
                raise ValueError("Neural probabilities must have positive mass")
            neural = neural / total
            fused = (1.0 - self.neural_weight) * causal + self.neural_weight * neural
        biases = np.asarray(self.class_biases, dtype=float)
        if biases.shape != fused.shape or not np.all(np.isfinite(biases)):
            raise ValueError("Class biases must be finite and match diagnosis classes")
        if not np.any(biases):
            return fused
        logits = np.log(np.clip(fused, np.finfo(float).tiny, None)) + biases
        logits -= logits.max()
        calibrated = np.exp(logits)
        return calibrated / calibrated.sum()

    def rank_drivers(
        self,
        states: Sequence[int] | Mapping[str, int],
        top_k: int = 5,
        target_label: str | int | None = None,
        neural_probabilities: Sequence[float] | None = None,
        posterior_scope: str = "reported_fused_posterior",
    ) -> list[CausalDriver]:
        """Rank single-state sensitivity factors for a declared probability scope.

        ``bayesian_component`` is useful when validation assigns all predictive weight to
        a neural expert.  It exposes what the learned Bayesian network says without
        implying that those edits change the reported neural prediction.
        """

        if posterior_scope not in {"reported_fused_posterior", "bayesian_component"}:
            raise ValueError("Unknown driver posterior scope")
        network = self._check()
        evidence = self.evidence(states)
        posterior = (
            self.posterior_array
            if posterior_scope == "bayesian_component"
            else lambda value: self.fused_posterior_array(value, neural_probabilities)
        )
        baseline = posterior(evidence)
        if target_label is None:
            predicted = int(np.argmax(baseline))
        elif isinstance(target_label, str):
            if target_label not in self.label_names:
                raise ValueError(f"Unknown driver target diagnosis: {target_label}")
            predicted = self.label_names.index(target_label)
        else:
            predicted = int(target_label)
            if predicted < 0 or predicted >= len(self.label_names):
                raise ValueError("Driver target diagnosis is out of range")
        drivers: list[CausalDriver] = []
        # Every latent feature is observed here, so the Markov blanket reduces
        # P(diagnosis | all latents) to the diagnosis node's direct parents.
        # Testing other ancestors cannot change this posterior and is needlessly costly.
        candidate_features = [name for name in network.parents[self.target] if name in evidence]
        for name in candidate_features:
            old = evidence[name]
            alternatives: list[tuple[float, float, int]] = []
            for state in self.supported_states(name):
                if state == old:
                    continue
                edited = {**evidence, name: state}
                value = posterior(edited)[predicted]
                alternatives.append(
                    (
                        abs(float(baseline[predicted] - value)),
                        float(baseline[predicted] - value),
                        state,
                    )
                )
            importance, effect, alternative = max(alternatives, default=(0.0, 0.0, old))
            path = network.shortest_path(name, self.target)
            if importance > 1e-12:
                drivers.append(
                    CausalDriver(
                        name=name,
                        state=old,
                        importance=importance,
                        effect=effect,
                        alternative_state=alternative,
                        path=path,
                        scope=posterior_scope,
                    )
                )
        return sorted(drivers, key=lambda item: item.importance, reverse=True)[:top_k]

    def supported_states(self, feature: str) -> list[int]:
        if feature not in self.feature_names:
            raise KeyError(feature)
        return list(self.state_support.get(feature, range(self.n_bins)))

    def evidence_paths(self, drivers: Sequence[CausalDriver]) -> list[list[str]]:
        return [driver.path for driver in drivers if driver.path]

    def to_dict(self) -> dict:
        return {
            "label_names": self.label_names,
            "feature_names": self.feature_names,
            "n_bins": self.n_bins,
            "neural_weight": self.neural_weight,
            "class_biases": [float(value) for value in self.class_biases],
            "state_support": self.state_support,
            "network": self._check().to_dict(),
        }

    @classmethod
    def from_dict(cls, values: dict) -> BayesianCausalModel:
        result = cls(values["label_names"], values["feature_names"], values["n_bins"])
        result.neural_weight = float(values.get("neural_weight", 0.0))
        if not 0 <= result.neural_weight <= 1:
            raise ValueError("Serialized neural fusion weight must be in [0, 1]")
        result.class_biases = np.asarray(
            values.get("class_biases", np.zeros(len(result.label_names))), dtype=float
        )
        if result.class_biases.shape != (len(result.label_names),) or not np.all(
            np.isfinite(result.class_biases)
        ):
            raise ValueError("Serialized class biases must be finite and match diagnosis classes")
        result.class_biases -= result.class_biases[0]
        result.state_support = {
            name: [int(state) for state in states]
            for name, states in values.get("state_support", result.state_support).items()
        }
        result.network = DiscreteBayesianNetwork.from_dict(values["network"])
        return result

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BayesianCausalModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
