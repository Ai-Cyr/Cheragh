import unittest

from cheragh import Document, HashingEmbedding, StaticLLMClient, RAGEngine
from cheragh.multihop import MultiHopRAGEngine
from cheragh.graph import GraphRAGEngine
from cheragh.raptor_engine import RAPTOREngine
from cheragh.federated import FederatedRAGEngine, FederatedRetriever


class V06ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document("Client Alpha a généré 100 euros au Q1. Client Alpha utilise Produit A.", doc_id="sales-alpha"),
            Document("Client Beta a généré 180 euros au Q1. Client Beta utilise Produit B.", doc_id="sales-beta"),
            Document("Produit B a un risque contractuel élevé avec Fournisseur Delta.", doc_id="risk-beta"),
        ]
        self.embedder = HashingEmbedding(dimension=64)

    def test_multihop_engine_returns_hops(self):
        base = RAGEngine.from_documents(self.docs, embedding_model=self.embedder, llm_client=StaticLLMClient("Réponse multi-hop [source: sales-beta]"), retriever_type="memory")
        engine = MultiHopRAGEngine(base.retriever, llm_client=StaticLLMClient("Réponse multi-hop [source: sales-beta]"), max_steps=3)
        result = engine.ask("Compare Client Alpha et Client Beta au Q1")
        self.assertTrue(result.hops)
        self.assertEqual(result.response.metadata["architecture"], "multi_hop")
        self.assertIn("Réponse multi-hop", result.answer)

    def test_graph_rag_engine_exposes_graph_metadata(self):
        engine = GraphRAGEngine.from_documents(
            self.docs,
            embedding_model=self.embedder,
            llm_client=StaticLLMClient("Beta est lié au Produit B. [source: sales-beta]"),
        )
        result = engine.ask("Que sait-on sur Client Beta ?")
        self.assertEqual(result.metadata["architecture"], "graph_rag")
        self.assertIn("graph_triples", result.metadata)
        self.assertGreaterEqual(len(engine.graph.triples), 1)

    def test_raptor_engine_builds_summary_nodes(self):
        engine = RAPTOREngine.from_documents(
            self.docs,
            embedding_model=self.embedder,
            llm_client=StaticLLMClient("Résumé cluster"),
            levels=1,
            branching_factor=2,
        )
        self.assertGreater(len(engine.index.nodes), len(self.docs))
        result = engine.ask("Quel client a le plus généré ?")
        self.assertEqual(result.metadata["architecture"], "raptor")
        self.assertIn("raptor_index", result.metadata)

    def test_federated_engine_merges_sources(self):
        finance = RAGEngine.from_documents([self.docs[0], self.docs[1]], embedding_model=self.embedder, llm_client=StaticLLMClient("finance"), retriever_type="memory")
        risk = RAGEngine.from_documents([self.docs[2]], embedding_model=self.embedder, llm_client=StaticLLMClient("risk"), retriever_type="memory")
        engine = FederatedRAGEngine(
            {"finance": finance, "risk": risk},
            llm_client=StaticLLMClient("Synthèse fédérée [source: sales-beta]"),
            top_k_per_source=2,
        )
        result = engine.ask("Compare revenus et risques de Beta")
        self.assertEqual(result.metadata["architecture"], "federated_rag")
        self.assertIn("finance", result.metadata["sources_queried"])
        self.assertTrue(any(src.metadata.get("source_name") for src in result.sources))

    def test_federated_retriever_accepts_callable(self):
        retriever = FederatedRetriever({"callable": lambda query: [Document("réponse callable", doc_id="c1")]})
        docs = retriever.retrieve("test", top_k=1)
        self.assertEqual(docs[0].metadata["source_name"], "callable")


if __name__ == "__main__":
    unittest.main()
