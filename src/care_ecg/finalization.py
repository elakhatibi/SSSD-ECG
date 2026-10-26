from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .ensemble import LinearProbabilityStacker, load_probability_table, sha256_file
from .expert import load_aligned_expert_probabilities
from .pipeline import CAREECGPipeline

_AUXILIARY_ARTIFACTS = ("knowledge_graph.json", "training_history.json")
_EXPERT_MATCH_ATOL = 1e-6
_EXPERT_MATCH_RTOL = 1e-6


def parse_named_probability_paths(values: Sequence[str]) -> list[tuple[str, Path]]:
    """Parse repeated ``NAME=PATH`` arguments without changing their order."""
    parsed: list[tuple[str, Path]] = []
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError("Probability sources must use NAME=PATH syntax")
        parsed.append((name.strip(), Path(path.strip())))
    names = [name for name, _ in parsed]
    if len(parsed) < 2 or len(names) != len(set(names)):
        raise ValueError("At least two uniquely named probability sources are required")
    return parsed


def _is_test_named(path: Path) -> bool:
    return "test" in path.name.lower()


def _require_file(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} is not a file: {resolved}")
    return resolved


def _file_manifest(
    path: Path,
    *,
    reported_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    resolved = path.resolve()
    result: dict[str, Any] = {
        "path": str((reported_path or resolved).resolve()),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    result.update(extra)
    return result


def _declared_source_orders(report: Mapping[str, Any]) -> list[tuple[str, ...]]:
    orders: list[tuple[str, ...]] = []
    for key in ("source_order", "selected_sources", "sources"):
        values = report.get(key)
        if isinstance(values, list) and all(isinstance(value, str) for value in values):
            orders.append(tuple(values))

    recommendation = report.get("recommendation")
    if isinstance(recommendation, Mapping):
        values = recommendation.get("sources")
        if isinstance(values, list) and all(isinstance(value, str) for value in values):
            orders.append(tuple(values))

    feature_sources = report.get("feature_sources")
    if isinstance(feature_sources, list) and feature_sources:
        names = []
        for source in feature_sources:
            if not isinstance(source, Mapping) or not isinstance(source.get("name"), str):
                break
            names.append(source["name"])
        else:
            orders.append(tuple(names))
    return orders


def _load_selection_report(path: Path, source_order: tuple[str, ...]) -> dict[str, Any]:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Selection report is not valid UTF-8 JSON: {path}") from error
    if not isinstance(values, dict):
        raise TypeError("Selection report root must be a JSON object")

    declared_orders = _declared_source_orders(values)
    if not declared_orders:
        raise ValueError("Selection report must declare its ordered probability sources")
    inconsistent = [order for order in declared_orders if order != source_order]
    if inconsistent:
        raise ValueError(
            f"Selection report source order must be {source_order}, received {inconsistent[0]}"
        )
    return values


def _validate_output_location(base: Path, output: Path, overwrite: bool) -> None:
    base_resolved = base.resolve()
    output_resolved = output.resolve()
    if (
        output_resolved == base_resolved
        or base_resolved in output_resolved.parents
        or output_resolved in base_resolved.parents
    ):
        raise ValueError("Base and output artifact directories must not overlap")
    if not output.exists():
        return
    if not output.is_dir():
        raise FileExistsError(f"Output artifact path is not a directory: {output_resolved}")
    if any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output artifact directory is not empty: {output_resolved}; use --overwrite"
        )


def _replace_output(staging: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if any(output.iterdir()) and not overwrite:
            raise FileExistsError(f"Output artifact directory is not empty: {output.resolve()}")
        shutil.rmtree(output)
    staging.replace(output)


def build_locked_expert_artifact(
    *,
    base_artifact: str | Path,
    output_artifact: str | Path,
    locked_stacker: str | Path,
    selection_report: str | Path,
    expert_probabilities: str | Path,
    source_probabilities: Sequence[tuple[str, str | Path]],
    paper_artifact_deviation: str,
    allow_test_after_lock: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a CARE artifact whose reported diagnosis is the locked expert exactly.

    This function intentionally accepts no prepared dataset. It inspects probability
    archives only through their label-free record-ID and probability contracts.
    """
    base = Path(base_artifact).expanduser().resolve()
    output = Path(output_artifact).expanduser().resolve()
    stacker_path = Path(locked_stacker)
    report_path = Path(selection_report)
    expert_path = Path(expert_probabilities)
    named_paths = [(str(name), Path(path)) for name, path in source_probabilities]

    guarded_paths = [expert_path, *(path for _, path in named_paths)]
    if any(_is_test_named(path) for path in guarded_paths) and not allow_test_after_lock:
        raise PermissionError(
            "Refusing test-named expert probabilities without explicit "
            "--allow-test-after-lock"
        )
    if not isinstance(paper_artifact_deviation, str) or not paper_artifact_deviation.strip():
        raise ValueError("paper_artifact_deviation must be a non-empty statement")
    if not base.is_dir():
        raise FileNotFoundError(f"Base CARE artifact is not a directory: {base}")
    _validate_output_location(base, output, overwrite)

    stacker_path = _require_file(stacker_path, "Locked stacker")
    report_path = _require_file(report_path, "Selection report")
    expert_path = _require_file(expert_path, "Expert probability archive")
    resolved_sources = [
        (name, _require_file(path, f"Probability source {name!r}"))
        for name, path in named_paths
    ]
    source_names = tuple(name for name, _ in resolved_sources)
    if len(resolved_sources) < 2 or any(not name.strip() for name in source_names):
        raise ValueError("At least two named probability sources are required")
    if len(set(source_names)) != len(source_names):
        raise ValueError("Probability source names must be unique")

    output_inputs = [stacker_path, report_path, expert_path]
    output_inputs.extend(path for _, path in resolved_sources)
    if any(output == path or output in path.parents for path in output_inputs):
        raise ValueError("Input artifacts must not be stored inside the output directory")

    stacker = LinearProbabilityStacker.load(stacker_path)
    if source_names != stacker.source_names:
        raise ValueError(
            f"Named probability source order must be {stacker.source_names}, "
            f"received {source_names}"
        )
    if not np.array_equal(stacker.class_biases, np.zeros(len(stacker.label_order))):
        raise ValueError("Locked stacker class_biases must be exactly zero")

    report = _load_selection_report(report_path, stacker.source_names)
    expert_table = load_probability_table(expert_path, expected_labels=stacker.label_order)
    strict_expert = load_aligned_expert_probabilities(
        expert_path,
        expert_table.record_ids,
        stacker.label_order,
    )
    sources = [
        (name, load_probability_table(path, expected_labels=stacker.label_order))
        for name, path in resolved_sources
    ]
    recomputed = stacker.predict_sources(sources, expert_table.record_ids)
    if not np.allclose(
        strict_expert,
        recomputed,
        rtol=_EXPERT_MATCH_RTOL,
        atol=_EXPERT_MATCH_ATOL,
    ):
        difference = float(np.max(np.abs(strict_expert - recomputed)))
        raise ValueError(
            "Expert probabilities do not match the locked stacker and named sources "
            f"(maximum absolute difference {difference:.8g})"
        )
    max_difference = float(np.max(np.abs(strict_expert - recomputed)))

    pipeline = CAREECGPipeline.load(base)
    label_order = tuple(str(label) for label in pipeline.causal_model.label_names)
    if label_order != stacker.label_order:
        raise ValueError(
            f"Base CARE label order must be {stacker.label_order}, received {label_order}"
        )
    pipeline.causal_model.neural_weight = 1.0
    pipeline.causal_model.class_biases = np.zeros(len(label_order), dtype=float)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staging = Path(temporary) / "artifact"
        pipeline.save(staging)
        preserved_auxiliary: list[dict[str, Any]] = []
        for filename in _AUXILIARY_ARTIFACTS:
            source = base / filename
            if source.is_file():
                destination = staging / filename
                shutil.copy2(source, destination)
                preserved_auxiliary.append(
                    _file_manifest(destination, reported_path=output / filename)
                )

        reloaded_graph = pipeline.causal_model.load(staging / "causal_graph.json")
        if reloaded_graph.neural_weight != 1.0:
            raise RuntimeError("Reloaded causal graph did not preserve neural_weight=1.0")
        if not np.array_equal(
            reloaded_graph.class_biases,
            np.zeros(len(label_order), dtype=float),
        ):
            raise RuntimeError("Reloaded causal graph did not preserve exact zero class offsets")

        stacker_manifest = _file_manifest(stacker_path)
        declared_stacker_hashes = [
            report.get(key)
            for key in ("model_sha256", "stacker_sha256", "locked_stacker_sha256")
            if report.get(key) is not None
        ]
        if any(value != stacker_manifest["sha256"] for value in declared_stacker_hashes):
            raise ValueError("Selection report stacker SHA-256 does not match locked stacker")

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "role": "validation-locked external diagnosis expert",
            "expert_weight": 1.0,
            "internal_care_weight": 0.0,
            "label_order": list(label_order),
            "source_order": list(stacker.source_names),
            "record_count": len(expert_table.record_ids),
            "zero_offset_policy": {
                "causal_neural_weight": 1.0,
                "causal_class_biases": [0.0] * len(label_order),
                "stacker_class_biases": [0.0] * len(label_order),
                "post_selection_decision_offsets": "none",
            },
            "base_care_artifact": {
                "path": str(base),
                "causal_graph": _file_manifest(base / "causal_graph.json"),
            },
            "locked_stacker": stacker_manifest,
            "selection_report": _file_manifest(report_path),
            "expert_probabilities": _file_manifest(
                expert_path,
                record_count=len(expert_table.record_ids),
            ),
            "source_probabilities": [
                _file_manifest(path, name=name) for name, path in resolved_sources
            ],
            "contract_validation": {
                "record_ids_exactly_aligned": True,
                "label_order_exactly_aligned": True,
                "stack_recomputed_from_sources": True,
                "maximum_absolute_probability_difference": max_difference,
            },
            "preserved_auxiliary_artifacts": preserved_auxiliary,
            "paper_artifact_deviation": {
                "uses_exact_paper_artifact": False,
                "description": paper_artifact_deviation.strip(),
            },
        }
        (staging / "external_expert.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _replace_output(staging, output, overwrite)

    verified_graph = pipeline.causal_model.load(output / "causal_graph.json")
    if verified_graph.neural_weight != 1.0 or not np.array_equal(
        verified_graph.class_biases,
        np.zeros(len(label_order), dtype=float),
    ):
        raise RuntimeError("Final artifact causal graph violates the locked expert policy")
    return manifest
