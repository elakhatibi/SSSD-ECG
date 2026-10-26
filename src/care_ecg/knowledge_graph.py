from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .data import PTBXL_DIAGNOSTIC_SUPERCLASSES
from .retrieval import KnowledgeDocument

KNOWLEDGE_GRAPH_SCHEMA_VERSION = "care-ecg.knowledge-graph/v1"

NODE_KINDS = frozenset(
    {
        "dataset",
        "diagnostic_superclass",
        "diagnostic_subclass",
        "scp_statement",
        "model_variable",
    }
)
EDGE_RELATIONS = frozenset(
    {
        "contains_label",
        "classified_as",
        "subclass_of",
        "causal_prior",
    }
)
PROVENANCE_KINDS = frozenset(
    {"dataset_metadata", "publication", "expert_review", "model_specification"}
)
CAUSAL_PROVENANCE_KINDS = frozenset({"publication", "expert_review", "model_specification"})


class KnowledgeGraphValidationError(ValueError):
    """Raised when a graph violates the CARE-ECG knowledge graph contract."""


@dataclass(frozen=True, slots=True)
class KnowledgeProvenance:
    provenance_id: str
    title: str
    kind: str
    locator: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KnowledgeNode:
    node_id: str
    kind: str
    label: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    model_variable: str | None = None
    provenance_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KnowledgeEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    statement: str
    provenance_id: str
    include_in_retrieval: bool = True
    allowed_as_prior: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClinicalKnowledgeGraph:
    graph_id: str
    title: str
    provenance: list[KnowledgeProvenance]
    nodes: list[KnowledgeNode]
    edges: list[KnowledgeEdge]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = KNOWLEDGE_GRAPH_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != KNOWLEDGE_GRAPH_SCHEMA_VERSION:
            raise KnowledgeGraphValidationError(
                f"Unsupported knowledge graph schema version: {self.schema_version!r}"
            )
        _require_nonempty(self.graph_id, "graph_id")
        _require_nonempty(self.title, "title")
        _ensure_json_object(self.metadata, "metadata")

        provenance_by_id = _unique_by(
            self.provenance, lambda item: item.provenance_id, "provenance"
        )
        for item in self.provenance:
            _require_nonempty(item.provenance_id, "provenance_id")
            _require_nonempty(item.title, f"provenance {item.provenance_id!r} title")
            if item.kind not in PROVENANCE_KINDS:
                raise KnowledgeGraphValidationError(
                    f"Unsupported provenance kind {item.kind!r} for {item.provenance_id!r}"
                )
            _ensure_json_object(item.metadata, f"provenance {item.provenance_id!r} metadata")

        nodes_by_id = _unique_by(self.nodes, lambda item: item.node_id, "node")
        model_variables: dict[str, str] = {}
        for node in self.nodes:
            _require_nonempty(node.node_id, "node_id")
            _require_nonempty(node.label, f"node {node.node_id!r} label")
            if node.kind not in NODE_KINDS:
                raise KnowledgeGraphValidationError(
                    f"Unsupported node kind {node.kind!r} for {node.node_id!r}"
                )
            if node.provenance_id is not None and node.provenance_id not in provenance_by_id:
                raise KnowledgeGraphValidationError(
                    f"Node {node.node_id!r} references unknown provenance {node.provenance_id!r}"
                )
            if len(node.aliases) != len(set(node.aliases)):
                raise KnowledgeGraphValidationError(f"Node {node.node_id!r} has duplicate aliases")
            if node.kind == "model_variable":
                _require_nonempty(
                    node.model_variable,
                    f"model_variable for node {node.node_id!r}",
                )
            elif node.model_variable is not None:
                raise KnowledgeGraphValidationError(
                    f"Only model_variable nodes may set model_variable ({node.node_id!r})"
                )
            if node.model_variable is not None:
                if node.model_variable in model_variables:
                    raise KnowledgeGraphValidationError(
                        f"Duplicate model variable {node.model_variable!r} on nodes "
                        f"{model_variables[node.model_variable]!r} and {node.node_id!r}"
                    )
                model_variables[node.model_variable] = node.node_id
            _ensure_json_object(node.metadata, f"node {node.node_id!r} metadata")

        _unique_by(self.edges, lambda item: item.edge_id, "edge")
        prior_adjacency: dict[str, list[str]] = {}
        seen_edges: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            _require_nonempty(edge.edge_id, "edge_id")
            if edge.source not in nodes_by_id or edge.target not in nodes_by_id:
                raise KnowledgeGraphValidationError(
                    f"Edge {edge.edge_id!r} references an unknown endpoint"
                )
            if edge.source == edge.target:
                raise KnowledgeGraphValidationError(f"Edge {edge.edge_id!r} is a self-loop")
            if edge.relation not in EDGE_RELATIONS:
                raise KnowledgeGraphValidationError(
                    f"Unsupported edge relation {edge.relation!r} for {edge.edge_id!r}"
                )
            signature = (edge.source, edge.target, edge.relation)
            if signature in seen_edges:
                raise KnowledgeGraphValidationError(
                    f"Duplicate {edge.relation!r} edge from {edge.source!r} to {edge.target!r}"
                )
            seen_edges.add(signature)
            _require_nonempty(edge.statement, f"edge {edge.edge_id!r} statement")
            if edge.provenance_id not in provenance_by_id:
                raise KnowledgeGraphValidationError(
                    f"Edge {edge.edge_id!r} references unknown provenance {edge.provenance_id!r}"
                )
            _ensure_json_object(edge.metadata, f"edge {edge.edge_id!r} metadata")
            self._validate_edge_types(edge, nodes_by_id)

            if edge.relation == "causal_prior":
                if not edge.allowed_as_prior:
                    raise KnowledgeGraphValidationError(
                        f"Causal edge {edge.edge_id!r} must explicitly set allowed_as_prior"
                    )
                provenance_kind = provenance_by_id[edge.provenance_id].kind
                if provenance_kind not in CAUSAL_PROVENANCE_KINDS:
                    raise KnowledgeGraphValidationError(
                        f"Causal edge {edge.edge_id!r} requires reviewed/model provenance, "
                        f"not {provenance_kind!r}"
                    )
                prior_adjacency.setdefault(edge.source, []).append(edge.target)
            elif edge.allowed_as_prior:
                raise KnowledgeGraphValidationError(
                    f"Non-causal edge {edge.edge_id!r} cannot be allowed as a prior"
                )

        _reject_cycles(prior_adjacency)

    @staticmethod
    def _validate_edge_types(edge: KnowledgeEdge, nodes_by_id: Mapping[str, KnowledgeNode]) -> None:
        source_kind = nodes_by_id[edge.source].kind
        target_kind = nodes_by_id[edge.target].kind
        allowed: dict[str, set[tuple[str, str]]] = {
            "contains_label": {
                ("dataset", "diagnostic_superclass"),
                ("dataset", "scp_statement"),
            },
            "classified_as": {("scp_statement", "diagnostic_superclass")},
            "subclass_of": {("diagnostic_subclass", "diagnostic_superclass")},
            "causal_prior": {("model_variable", "model_variable")},
        }
        if (source_kind, target_kind) not in allowed[edge.relation]:
            raise KnowledgeGraphValidationError(
                f"Edge {edge.edge_id!r} has invalid endpoint kinds for {edge.relation!r}: "
                f"{source_kind!r} -> {target_kind!r}"
            )

    def causal_prior_edges(self, variable_order: Sequence[str]) -> list[tuple[str, str]]:
        """Return reviewed graph priors that are compatible with the K2 variable order."""
        self.validate()
        order = list(variable_order)
        if len(order) != len(set(order)):
            raise KnowledgeGraphValidationError("variable_order must not contain duplicates")
        positions = {name: index for index, name in enumerate(order)}
        nodes = {node.node_id: node for node in self.nodes}
        result: list[tuple[str, str]] = []
        for edge in self.edges:
            if edge.relation != "causal_prior" or not edge.allowed_as_prior:
                continue
            parent = nodes[edge.source].model_variable
            child = nodes[edge.target].model_variable
            if parent not in positions or child not in positions:
                raise KnowledgeGraphValidationError(
                    f"Causal edge {edge.edge_id!r} references variables outside the model: "
                    f"{parent!r} -> {child!r}"
                )
            if positions[parent] >= positions[child]:
                raise KnowledgeGraphValidationError(
                    f"Causal edge {edge.edge_id!r} violates K2 order: {parent!r} -> {child!r}"
                )
            result.append((parent, child))
        return result

    def retrieval_documents(self) -> list[KnowledgeDocument]:
        """Convert explicitly retrievable graph edges into provenance-rich documents."""
        self.validate()
        provenance = {item.provenance_id: item for item in self.provenance}
        nodes = {node.node_id: node for node in self.nodes}
        documents: list[KnowledgeDocument] = []
        for edge in self.edges:
            if not edge.include_in_retrieval:
                continue
            source = provenance[edge.provenance_id]
            source_node = nodes[edge.source]
            target_node = nodes[edge.target]
            metadata = dict(edge.metadata)
            superclass_labels = {
                node.label
                for node in (source_node, target_node)
                if node.kind == "diagnostic_superclass"
            }
            if "diagnostic_class" not in metadata and len(superclass_labels) == 1:
                metadata["diagnostic_class"] = next(iter(superclass_labels))
            documents.append(
                KnowledgeDocument(
                    doc_id=f"kg:{self.graph_id}:{edge.edge_id}",
                    text=edge.statement,
                    source=source.title,
                    metadata={
                        **metadata,
                        "knowledge_graph": self.graph_id,
                        "schema_version": self.schema_version,
                        "edge_id": edge.edge_id,
                        "relation": edge.relation,
                        "source_node": edge.source,
                        "source_label": source_node.label,
                        "target_node": edge.target,
                        "target_label": target_node.label,
                        "provenance_id": edge.provenance_id,
                        "provenance_kind": source.kind,
                    },
                )
            )
        return documents

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_id": self.graph_id,
            "title": self.title,
            "description": self.description,
            "provenance": [asdict(item) for item in self.provenance],
            "nodes": [
                {
                    **asdict(node),
                    "aliases": list(node.aliases),
                }
                for node in self.nodes
            ],
            "edges": [asdict(edge) for edge in self.edges],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ClinicalKnowledgeGraph:
        try:
            _reject_unknown_keys(
                values,
                {
                    "schema_version",
                    "graph_id",
                    "title",
                    "description",
                    "provenance",
                    "nodes",
                    "edges",
                    "metadata",
                },
                "knowledge graph",
            )
            graph = cls(
                schema_version=_required_string(values, "schema_version"),
                graph_id=_required_string(values, "graph_id"),
                title=_required_string(values, "title"),
                description=_optional_string(values, "description"),
                provenance=[
                    _parse_provenance(item, index)
                    for index, item in enumerate(_required_list(values, "provenance"))
                ],
                nodes=[
                    _parse_node(item, index)
                    for index, item in enumerate(_required_list(values, "nodes"))
                ],
                edges=[
                    _parse_edge(item, index)
                    for index, item in enumerate(_required_list(values, "edges"))
                ],
                metadata=_optional_mapping(values, "metadata"),
            )
        except KnowledgeGraphValidationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise KnowledgeGraphValidationError("Malformed knowledge graph") from exc
        graph.validate()
        return graph

    def save(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ClinicalKnowledgeGraph:
        try:
            values = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise KnowledgeGraphValidationError(f"Invalid knowledge graph JSON: {path}") from exc
        if not isinstance(values, Mapping):
            raise KnowledgeGraphValidationError("Knowledge graph root must be a JSON object")
        return cls.from_dict(values)


def build_ptbxl_knowledge_graph(root: str | Path) -> ClinicalKnowledgeGraph:
    """Build a non-causal PTB-XL ontology graph from ``scp_statements.csv``.

    The generated edges encode dataset taxonomy only. They are retrievable evidence,
    but are deliberately ineligible as Bayesian-network causal priors.
    """
    path = Path(root) / "scp_statements.csv"
    raw = path.read_bytes()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"diagnostic", "diagnostic_class"}
        if not fields or not required.issubset(fields):
            raise KnowledgeGraphValidationError(
                "scp_statements.csv must contain SCP code, diagnostic, and diagnostic_class columns"
            )
        code_field = fields[0]
        description_field = next(
            (name for name in ("description", "SCP-ECG Statement Description") if name in fields),
            None,
        )
        rows = list(reader)

    provenance_id = "ptbxl-scp-statements"
    provenance = [
        KnowledgeProvenance(
            provenance_id=provenance_id,
            title="PTB-XL SCP statements metadata",
            kind="dataset_metadata",
            locator="scp_statements.csv",
            metadata={"sha256": hashlib.sha256(raw).hexdigest()},
        )
    ]
    nodes = [
        KnowledgeNode(
            node_id="ptbxl:dataset",
            kind="dataset",
            label="PTB-XL",
            description="PTB-XL dataset diagnostic ontology.",
            provenance_id=provenance_id,
        )
    ]
    edges: list[KnowledgeEdge] = []
    for superclass in PTBXL_DIAGNOSTIC_SUPERCLASSES:
        node_id = f"ptbxl:superclass:{superclass}"
        nodes.append(
            KnowledgeNode(
                node_id=node_id,
                kind="diagnostic_superclass",
                label=superclass,
                description=f"PTB-XL diagnostic superclass label {superclass}.",
                provenance_id=provenance_id,
            )
        )
        edges.append(
            KnowledgeEdge(
                edge_id=f"ptbxl:contains:{superclass}",
                source="ptbxl:dataset",
                target=node_id,
                relation="contains_label",
                statement=f"PTB-XL defines {superclass} as a diagnostic superclass label.",
                provenance_id=provenance_id,
            )
        )

    seen_codes: set[str] = set()
    for row in rows:
        diagnostic = str(row.get("diagnostic", "")).strip().lower()
        superclass = str(row.get("diagnostic_class", "")).strip().upper()
        code = str(row.get(code_field, "")).strip()
        if diagnostic not in {"1", "1.0", "true", "yes"}:
            continue
        if not code or superclass not in PTBXL_DIAGNOSTIC_SUPERCLASSES:
            continue
        if code in seen_codes:
            raise KnowledgeGraphValidationError(f"Duplicate diagnostic SCP code {code!r}")
        seen_codes.add(code)
        description = str(row.get(description_field, "")).strip() if description_field else ""
        code_token = _stable_id_token(code)
        node_id = f"ptbxl:scp:{code_token}"
        nodes.append(
            KnowledgeNode(
                node_id=node_id,
                kind="scp_statement",
                label=code,
                description=description,
                aliases=(description,) if description and description != code else (),
                provenance_id=provenance_id,
                metadata={"diagnostic_class": superclass},
            )
        )
        edges.append(
            KnowledgeEdge(
                edge_id=f"ptbxl:classified-as:{code_token}",
                source=node_id,
                target=f"ptbxl:superclass:{superclass}",
                relation="classified_as",
                statement=(
                    f"In PTB-XL metadata, diagnostic SCP statement {code} is classified "
                    f"under the {superclass} diagnostic superclass."
                ),
                provenance_id=provenance_id,
                metadata={"scp_code": code, "diagnostic_class": superclass},
            )
        )

    graph = ClinicalKnowledgeGraph(
        graph_id="ptbxl-diagnostic-ontology",
        title="PTB-XL diagnostic ontology",
        description=(
            "Dataset-derived SCP statement to diagnostic-superclass mappings. "
            "Classification edges are not physiological causal claims."
        ),
        provenance=provenance,
        nodes=nodes,
        edges=edges,
        metadata={
            "diagnostic_superclasses": list(PTBXL_DIAGNOSTIC_SUPERCLASSES),
            "causal_claims": False,
        },
    )
    graph.validate()
    return graph


def _parse_provenance(value: Any, index: int) -> KnowledgeProvenance:
    item = _as_mapping(value, f"provenance[{index}]")
    _reject_unknown_keys(
        item,
        {"provenance_id", "title", "kind", "locator", "metadata"},
        f"provenance[{index}]",
    )
    return KnowledgeProvenance(
        provenance_id=_required_string(item, "provenance_id"),
        title=_required_string(item, "title"),
        kind=_required_string(item, "kind"),
        locator=_nullable_string(item, "locator"),
        metadata=_optional_mapping(item, "metadata"),
    )


def _parse_node(value: Any, index: int) -> KnowledgeNode:
    item = _as_mapping(value, f"nodes[{index}]")
    _reject_unknown_keys(
        item,
        {
            "node_id",
            "kind",
            "label",
            "description",
            "aliases",
            "model_variable",
            "provenance_id",
            "metadata",
        },
        f"nodes[{index}]",
    )
    aliases = _optional_list(item, "aliases")
    if any(not isinstance(alias, str) for alias in aliases):
        raise KnowledgeGraphValidationError(f"nodes[{index}].aliases must contain strings")
    return KnowledgeNode(
        node_id=_required_string(item, "node_id"),
        kind=_required_string(item, "kind"),
        label=_required_string(item, "label"),
        description=_optional_string(item, "description"),
        aliases=tuple(aliases),
        model_variable=_nullable_string(item, "model_variable"),
        provenance_id=_nullable_string(item, "provenance_id"),
        metadata=_optional_mapping(item, "metadata"),
    )


def _parse_edge(value: Any, index: int) -> KnowledgeEdge:
    item = _as_mapping(value, f"edges[{index}]")
    _reject_unknown_keys(
        item,
        {
            "edge_id",
            "source",
            "target",
            "relation",
            "statement",
            "provenance_id",
            "include_in_retrieval",
            "allowed_as_prior",
            "metadata",
        },
        f"edges[{index}]",
    )
    return KnowledgeEdge(
        edge_id=_required_string(item, "edge_id"),
        source=_required_string(item, "source"),
        target=_required_string(item, "target"),
        relation=_required_string(item, "relation"),
        statement=_required_string(item, "statement"),
        provenance_id=_required_string(item, "provenance_id"),
        include_in_retrieval=_optional_bool(item, "include_in_retrieval", True),
        allowed_as_prior=_optional_bool(item, "allowed_as_prior", False),
        metadata=_optional_mapping(item, "metadata"),
    )


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KnowledgeGraphValidationError(f"{context} must be an object")
    return value


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeGraphValidationError(f"{key} must be a non-empty string")
    return value


def _optional_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key, "")
    if not isinstance(value, str):
        raise KnowledgeGraphValidationError(f"{key} must be a string")
    return value


def _nullable_string(values: Mapping[str, Any], key: str) -> str | None:
    value = values.get(key)
    if value is not None and not isinstance(value, str):
        raise KnowledgeGraphValidationError(f"{key} must be a string or null")
    return value


def _required_list(values: Mapping[str, Any], key: str) -> list[Any]:
    value = values.get(key)
    if not isinstance(value, list):
        raise KnowledgeGraphValidationError(f"{key} must be an array")
    return value


def _optional_list(values: Mapping[str, Any], key: str) -> list[Any]:
    value = values.get(key, [])
    if not isinstance(value, list):
        raise KnowledgeGraphValidationError(f"{key} must be an array")
    return value


def _optional_mapping(values: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = values.get(key, {})
    if not isinstance(value, Mapping):
        raise KnowledgeGraphValidationError(f"{key} must be an object")
    return dict(value)


def _optional_bool(values: Mapping[str, Any], key: str, default: bool) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise KnowledgeGraphValidationError(f"{key} must be a boolean")
    return value


def _reject_unknown_keys(values: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise KnowledgeGraphValidationError(f"{context} has unsupported fields: {sorted(unknown)}")


def _require_nonempty(value: str | None, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeGraphValidationError(f"{context} must be a non-empty string")


def _unique_by(items: Sequence[Any], key, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        identifier = key(item)
        if identifier in result:
            raise KnowledgeGraphValidationError(f"Duplicate {label} identifier {identifier!r}")
        result[identifier] = item
    return result


def _ensure_json_object(value: Mapping[str, Any], context: str) -> None:
    if not isinstance(value, Mapping):
        raise KnowledgeGraphValidationError(f"{context} must be an object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise KnowledgeGraphValidationError(f"{context} must contain valid JSON values") from exc


def _reject_cycles(adjacency: Mapping[str, Sequence[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise KnowledgeGraphValidationError(
                "Causal-prior edges must form a directed acyclic graph"
            )
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency.get(node, ()):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in adjacency:
        visit(node)


def _stable_id_token(value: str) -> str:
    safe = "".join(character if character.isalnum() else "-" for character in value).strip("-")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{safe or 'code'}-{digest}"
