import json
import tempfile
import unittest
from pathlib import Path

from care_ecg.knowledge_graph import (
    KNOWLEDGE_GRAPH_SCHEMA_VERSION,
    ClinicalKnowledgeGraph,
    KnowledgeEdge,
    KnowledgeGraphValidationError,
    KnowledgeNode,
    KnowledgeProvenance,
    build_ptbxl_knowledge_graph,
)


class KnowledgeGraphTests(unittest.TestCase):
    def _causal_graph(self, provenance_kind="expert_review"):
        return ClinicalKnowledgeGraph(
            graph_id="reviewed-priors",
            title="Reviewed model priors",
            provenance=[
                KnowledgeProvenance(
                    "review-1",
                    "Reviewed model topology",
                    provenance_kind,
                    "review-record-1",
                )
            ],
            nodes=[
                KnowledgeNode(
                    "model:z0",
                    "model_variable",
                    "Latent factor 0",
                    model_variable="z0",
                    provenance_id="review-1",
                ),
                KnowledgeNode(
                    "model:diagnosis",
                    "model_variable",
                    "Diagnosis",
                    model_variable="diagnosis",
                    provenance_id="review-1",
                ),
            ],
            edges=[
                KnowledgeEdge(
                    "prior:z0-diagnosis",
                    "model:z0",
                    "model:diagnosis",
                    "causal_prior",
                    "The reviewed model specification permits z0 to precede diagnosis.",
                    "review-1",
                    allowed_as_prior=True,
                )
            ],
        )

    def test_bundled_ptbxl_graph_is_valid_noncausal_retrieval_knowledge(self):
        path = (
            Path(__file__).parents[1]
            / "src"
            / "care_ecg"
            / "assets"
            / "ptbxl_superclasses.kg.v1.json"
        )
        graph = ClinicalKnowledgeGraph.load(path)

        self.assertEqual(graph.schema_version, KNOWLEDGE_GRAPH_SCHEMA_VERSION)
        self.assertEqual(graph.metadata["causal_claims"], False)
        self.assertEqual(len(graph.retrieval_documents()), 5)
        self.assertEqual(graph.causal_prior_edges(["z0", "diagnosis"]), [])
        self.assertTrue(
            all(
                document.metadata["relation"] == "contains_label"
                for document in graph.retrieval_documents()
            )
        )
        self.assertEqual(
            {document.metadata["diagnostic_class"] for document in graph.retrieval_documents()},
            {"NORM", "MI", "STTC", "CD", "HYP"},
        )

    def test_ptbxl_builder_uses_only_diagnostic_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "scp_statements.csv"
            csv_path.write_text(
                ",description,diagnostic,diagnostic_class\n"
                "NORM,normal ECG,1,NORM\n"
                "IMI,inferior myocardial infarction,1.0,MI\n"
                "SR,sinus rhythm,0,\n"
                "OTHER,not a target class,true,OTHER\n",
                encoding="utf-8",
            )

            graph = build_ptbxl_knowledge_graph(directory)

        scp_nodes = [node for node in graph.nodes if node.kind == "scp_statement"]
        self.assertEqual({node.label for node in scp_nodes}, {"NORM", "IMI"})
        self.assertEqual(len(graph.edges), 7)
        self.assertEqual(len(graph.retrieval_documents()), 7)
        self.assertFalse(any(edge.allowed_as_prior for edge in graph.edges))
        self.assertEqual(len(graph.provenance[0].metadata["sha256"]), 64)
        imi = next(node for node in scp_nodes if node.label == "IMI")
        self.assertEqual(imi.aliases, ("inferior myocardial infarction",))

    def test_reviewed_causal_edges_convert_to_ordered_k2_priors(self):
        graph = self._causal_graph()

        self.assertEqual(
            graph.causal_prior_edges(["z0", "diagnosis"]),
            [("z0", "diagnosis")],
        )
        document = graph.retrieval_documents()[0]
        self.assertEqual(document.metadata["provenance_kind"], "expert_review")
        self.assertEqual(document.metadata["source_label"], "Latent factor 0")

    def test_graph_round_trip_preserves_prior_policy(self):
        graph = self._causal_graph()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            graph.save(path)
            restored = ClinicalKnowledgeGraph.load(path)

        self.assertEqual(restored.to_dict(), graph.to_dict())
        self.assertEqual(restored.causal_prior_edges(["z0", "diagnosis"]), [("z0", "diagnosis")])

    def test_causal_prior_requires_explicit_permission_and_reviewed_provenance(self):
        graph = self._causal_graph()
        graph.edges[0] = KnowledgeEdge(
            "prior:z0-diagnosis",
            "model:z0",
            "model:diagnosis",
            "causal_prior",
            "A candidate model edge.",
            "review-1",
            allowed_as_prior=False,
        )
        with self.assertRaisesRegex(KnowledgeGraphValidationError, "explicitly"):
            graph.validate()

        dataset_graph = self._causal_graph(provenance_kind="dataset_metadata")
        with self.assertRaisesRegex(KnowledgeGraphValidationError, "reviewed/model provenance"):
            dataset_graph.validate()

    def test_taxonomy_edge_cannot_be_promoted_to_causal_prior(self):
        graph = ClinicalKnowledgeGraph(
            graph_id="unsafe-taxonomy",
            title="Unsafe taxonomy",
            provenance=[KnowledgeProvenance("data", "Dataset metadata", "dataset_metadata")],
            nodes=[
                KnowledgeNode("dataset", "dataset", "Dataset", provenance_id="data"),
                KnowledgeNode("class", "diagnostic_superclass", "NORM", provenance_id="data"),
            ],
            edges=[
                KnowledgeEdge(
                    "taxonomy",
                    "dataset",
                    "class",
                    "contains_label",
                    "The dataset contains the NORM label.",
                    "data",
                    allowed_as_prior=True,
                )
            ],
        )

        with self.assertRaisesRegex(KnowledgeGraphValidationError, "Non-causal edge"):
            graph.validate()

    def test_prior_must_match_model_variables_and_k2_order(self):
        graph = self._causal_graph()
        with self.assertRaisesRegex(KnowledgeGraphValidationError, "outside the model"):
            graph.causal_prior_edges(["z1", "diagnosis"])
        with self.assertRaisesRegex(KnowledgeGraphValidationError, "violates K2 order"):
            graph.causal_prior_edges(["diagnosis", "z0"])

    def test_prior_cycles_are_rejected(self):
        graph = self._causal_graph()
        graph.edges.append(
            KnowledgeEdge(
                "prior:diagnosis-z0",
                "model:diagnosis",
                "model:z0",
                "causal_prior",
                "A deliberately invalid reverse edge.",
                "review-1",
                allowed_as_prior=True,
            )
        )

        with self.assertRaisesRegex(KnowledgeGraphValidationError, "acyclic"):
            graph.validate()

    def test_loader_rejects_unknown_schema_versions(self):
        values = self._causal_graph().to_dict()
        values["schema_version"] = "care-ecg.knowledge-graph/v999"
        with self.assertRaisesRegex(KnowledgeGraphValidationError, "Unsupported"):
            ClinicalKnowledgeGraph.from_dict(values)

    def test_loader_rejects_unknown_fields_instead_of_ignoring_typos(self):
        values = self._causal_graph().to_dict()
        values["edges"][0]["allow_as_prior"] = True
        with self.assertRaisesRegex(KnowledgeGraphValidationError, "allow_as_prior"):
            ClinicalKnowledgeGraph.from_dict(values)

    def test_loader_rejects_non_object_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            path.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaisesRegex(KnowledgeGraphValidationError, "root"):
                ClinicalKnowledgeGraph.load(path)


if __name__ == "__main__":
    unittest.main()
