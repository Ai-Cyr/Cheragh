import unittest

from cheragh import Document, StaticLLMClient
from cheragh.community_graph import (
    CommunityGraphRAGEngine,
    DeterministicCommunitySummarizer,
    detect_communities,
)
from cheragh.graph import KnowledgeGraph, KnowledgeTriple


def _triples():
    return [
        KnowledgeTriple("Alpha", "collabore avec", "Beta", "climate-a", {"confidence": 0.9}),
        KnowledgeTriple("Beta", "collabore avec", "Gamma", "climate-b"),
        KnowledgeTriple("Alpha", "finance", "Gamma", "climate-a"),
        KnowledgeTriple("Gamma", "pont faible", "Delta", "bridge"),
        KnowledgeTriple("Delta", "collabore avec", "Echo", "finance-a"),
        KnowledgeTriple("Echo", "collabore avec", "Foxtrot", "finance-b"),
        KnowledgeTriple("Delta", "finance", "Foxtrot", "finance-a"),
    ]


def _graph(reverse=False):
    graph = KnowledgeGraph()
    triples = list(reversed(_triples())) if reverse else _triples()
    for triple in triples:
        graph.add_triple(triple)
    return graph


def _documents():
    return [
        Document(
            "Alpha et Beta financent un programme climat et solaire.",
            metadata={"source": "climate.txt", "nested": {"owner": "original"}},
            doc_id="climate-a",
        ),
        Document("Beta et Gamma mesurent les émissions du programme climat.", doc_id="climate-b"),
        Document("Gamma rencontre Delta lors d'un forum.", doc_id="bridge"),
        Document("Delta et Echo publient les résultats financiers annuels.", doc_id="finance-a"),
        Document("Echo et Foxtrot analysent les revenus et les marges.", doc_id="finance-b"),
    ]


class CommunityDetectionTests(unittest.TestCase):
    def test_partition_is_deterministic_and_splits_weakly_connected_clusters(self):
        forward = detect_communities(_graph())
        reverse = detect_communities(_graph(reverse=True))

        self.assertEqual([community.entities for community in forward], [community.entities for community in reverse])
        self.assertEqual(
            [community.to_dict() for community in forward],
            [community.to_dict() for community in reverse],
        )
        self.assertEqual(
            [community.entities for community in forward],
            [("Alpha", "Beta", "Gamma"), ("Delta", "Echo", "Foxtrot")],
        )
        all_entities = [entity for community in forward for entity in community.entities]
        self.assertEqual(len(all_entities), len(set(all_entities)))

    def test_partition_configuration_is_strict(self):
        with self.assertRaises(TypeError):
            detect_communities(_graph(), resolution=True)
        with self.assertRaises(ValueError):
            detect_communities(_graph(), resolution=0)
        with self.assertRaises(TypeError):
            detect_communities(_graph(), max_iterations=1.5)


class CommunityReportTests(unittest.TestCase):
    def test_blank_document_id_is_canonicalized_before_graph_construction(self):
        engine = CommunityGraphRAGEngine([Document("Alpha travaille avec Beta.", doc_id="   ")])

        self.assertEqual(engine.documents[0].doc_id, "doc-0")
        self.assertEqual(engine.local_search("Alpha", top_k=1)[0].doc_id, "doc-0")

    def test_injected_summarizer_receives_snapshots_and_provenance_is_preserved(self):
        calls = []

        def summarize(community, documents):
            calls.append((community.community_id, tuple(document.doc_id for document in documents)))
            if documents:
                documents[0].metadata["mutated_by_summarizer"] = True
            if community.triples:
                community.triples[0].metadata["mutated_by_summarizer"] = True
            return f"Rapport injecté sur {', '.join(community.entities)}"

        engine = CommunityGraphRAGEngine(_documents(), graph=_graph(), summarizer=summarize)

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(report.summary.startswith("Rapport injecté") for report in engine.reports))
        self.assertNotIn("mutated_by_summarizer", engine.documents[0].metadata)
        self.assertNotIn("mutated_by_summarizer", engine.graph.triples[0].metadata)
        report_document = engine.global_search("Alpha", top_k=1)[0]
        self.assertEqual(report_document.metadata["source_doc_ids"], ["bridge", "climate-a", "climate-b"])
        self.assertEqual(report_document.metadata["provenance"][0]["doc_id"], "bridge")

    def test_fallback_report_is_deterministic_and_searches_source_content(self):
        first = CommunityGraphRAGEngine(
            _documents(),
            graph=_graph(),
            summarizer=DeterministicCommunitySummarizer(),
        )
        second = CommunityGraphRAGEngine(
            list(reversed(_documents())),
            graph=_graph(reverse=True),
            summarizer=DeterministicCommunitySummarizer(),
        )

        self.assertEqual([report.to_dict() for report in first.reports], [report.to_dict() for report in second.reports])
        climate = first.global_search("programme solaire climat", top_k=1)[0]
        finance = first.global_search("revenus marges financiers", top_k=1)[0]
        self.assertEqual(climate.metadata["community_id"], 0)
        self.assertEqual(finance.metadata["community_id"], 1)
        self.assertGreater(climate.score, 0)
        self.assertGreater(finance.score, 0)

    def test_duplicate_ids_fail_and_missing_ids_do_not_collide(self):
        with self.assertRaisesRegex(ValueError, "duplicate document id"):
            CommunityGraphRAGEngine([Document("a", doc_id="same"), Document("b", doc_id="same")])

        engine = CommunityGraphRAGEngine(
            [Document("Alice collabore avec Bob."), Document("Bob connaît Alice.", doc_id="doc-0")]
        )
        self.assertEqual({document.doc_id for document in engine.documents}, {"doc-0", "doc-0-1"})


class CommunitySearchTests(unittest.TestCase):
    def setUp(self):
        self.source_documents = _documents()
        self.source_graph = _graph()
        self.llm = StaticLLMClient("Synthèse globale [source: community:0]")
        self.engine = CommunityGraphRAGEngine(
            self.source_documents,
            graph=self.source_graph,
            llm_client=self.llm,
            top_k=2,
            require_citations=True,
        )

    def test_global_and_local_search_enforce_top_k(self):
        self.assertEqual(len(self.engine.global_search("thèmes", top_k=1)), 1)
        self.assertEqual(len(self.engine.local_search("Alpha", top_k=1)), 1)
        for method in (self.engine.global_search, self.engine.local_search, self.engine.retrieve):
            with self.assertRaises(TypeError):
                method("query", top_k=True)
            with self.assertRaises(ValueError):
                method("query", top_k=0)
        with self.assertRaises(TypeError):
            self.engine.ask("query", top_k=1.5)

    def test_local_search_is_entity_and_community_aware(self):
        result = self.engine.local_search("Que fait Alpha ?", top_k=2)

        self.assertEqual(result[0].doc_id, "climate-a")
        self.assertLessEqual(len(result), 2)
        self.assertEqual(result[0].metadata["retrieval_method"], "community_graph_local")
        self.assertEqual(result[0].metadata["matched_entities"], ["Alpha"])
        self.assertEqual(result[0].metadata["community_ids"], [0])
        self.assertEqual(result[0].metadata["source"], "climate.txt")

    def test_answers_return_citations_and_end_to_end_provenance(self):
        global_response = self.engine.ask_global("Quels thèmes climat ressortent ?", top_k=1)

        self.assertEqual(global_response.metadata["architecture"], "community_graph_rag_baseline")
        self.assertEqual(global_response.metadata["mode"], "global")
        self.assertEqual(global_response.citations, ["community:0"])
        self.assertTrue(global_response.citation_validation.ok)
        self.assertEqual(len(global_response.retrieved_documents), 1)
        self.assertIn("climate-a", global_response.sources[0].metadata["source_doc_ids"])
        self.assertTrue(global_response.sources[0].metadata["provenance"])

        self.llm.response = "Détail local [source: climate-a]"
        local_response = self.engine.ask_local("Que fait Alpha ?", top_k=1)
        self.assertEqual(local_response.metadata["mode"], "local")
        self.assertEqual(local_response.citations, ["climate-a"])
        self.assertTrue(local_response.citation_validation.ok)
        self.assertEqual([source.doc_id for source in local_response.sources], ["climate-a"])

    def test_index_and_search_results_are_defensive_snapshots(self):
        self.source_documents[0].content = "contenu externe modifié"
        self.source_documents[0].metadata["nested"]["owner"] = "external mutation"
        self.source_graph.add_triple(KnowledgeTriple("Injected", "rel", "Entity", "climate-a"))

        documents = self.engine.documents
        documents[0].metadata["nested"]["owner"] = "returned mutation"
        graph = self.engine.graph
        graph.triples[0].metadata["confidence"] = -1
        reports = self.engine.reports
        reports[0].metadata["provenance"][0]["preview"] = "returned mutation"
        result = self.engine.local_search("Alpha", top_k=1)[0]
        result.content = "returned mutation"
        result.metadata["nested"]["owner"] = "returned mutation"

        fresh = self.engine.local_search("Alpha", top_k=1)[0]
        fresh_report = self.engine.reports[0]
        self.assertIn("programme climat", fresh.content)
        self.assertEqual(fresh.metadata["nested"]["owner"], "original")
        self.assertNotIn("Injected", [entity for community in self.engine.communities for entity in community.entities])
        self.assertEqual(self.engine.graph.triples[0].metadata["confidence"], 0.9)
        self.assertNotEqual(fresh_report.metadata["provenance"][0]["preview"], "returned mutation")

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.search("query", mode="hybrid")
        with self.assertRaises(TypeError):
            self.engine.ask("query", mode=None)


if __name__ == "__main__":
    unittest.main()
