import tempfile
import unittest
from pathlib import Path

from cheragh import (
    AdvancedRAGPipeline,
    Document,
    HashingEmbedding,
    HybridSearchRetriever,
    StaticLLMClient,
)


class CorePackageTest(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document("Le RAG combine retrieval et génération avec des sources.", doc_id="rag"),
            Document("Un package Python contient un pyproject.toml et du code réutilisable.", doc_id="python"),
            Document("Le café est une boisson chaude.", doc_id="cafe"),
        ]
        self.embedder = HashingEmbedding(dimension=64)

    def test_hybrid_retrieval_returns_relevant_doc(self):
        retriever = HybridSearchRetriever(self.docs, self.embedder, alpha=0.25)
        result = retriever.retrieve("package python", top_k=1)
        self.assertEqual(result[0].doc_id, "python")
        self.assertIsNotNone(result[0].score)

    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = str(Path(tmp) / "hybrid.pkl")
            retriever_1 = HybridSearchRetriever(self.docs, self.embedder, cache_path=cache_path)
            self.assertTrue(Path(cache_path).exists())
            retriever_2 = HybridSearchRetriever(self.docs, self.embedder, cache_path=cache_path)
            self.assertEqual(
                retriever_1.retrieve("RAG sources", top_k=1)[0].doc_id,
                retriever_2.retrieve("RAG sources", top_k=1)[0].doc_id,
            )

    def test_pipeline_uses_llm_client(self):
        retriever = HybridSearchRetriever(self.docs, self.embedder)
        llm = StaticLLMClient("réponse contrôlée")
        pipeline = AdvancedRAGPipeline(retriever, llm, top_k=2)
        output = pipeline.run("Qu'est-ce que le RAG ?")
        self.assertEqual(output["answer"], "réponse contrôlée")
        self.assertEqual(output["query"], "Qu'est-ce que le RAG ?")
        self.assertEqual(len(output["sources"]), 2)
        self.assertGreater(len(llm.prompts), 0)


if __name__ == "__main__":
    unittest.main()
