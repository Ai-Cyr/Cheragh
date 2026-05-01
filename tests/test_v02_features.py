import json
import tempfile
import unittest
from pathlib import Path

from cheragh import (
    Document,
    HashingEmbedding,
    MemoryVectorStore,
    RAGEngine,
    StaticLLMClient,
    chunk_documents,
    evaluate_retrieval,
    load_documents,
)
from cheragh.evaluation import RetrievalExample
from cheragh.ingestion import ingest_path


class V02FeaturesTest(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document("Le RAG combine recherche documentaire et génération.", doc_id="rag", metadata={"kind": "ai"}),
            Document("Python utilise pyproject.toml pour le packaging moderne.", doc_id="python", metadata={"kind": "code"}),
        ]
        self.embedder = HashingEmbedding(dimension=64)

    def test_chunk_documents_adds_parent_metadata(self):
        chunks = chunk_documents([Document("A" * 200 + "\n\n" + "B" * 200, doc_id="parent")], chunk_size=120, chunk_overlap=20)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["parent_doc_id"], "parent")
        self.assertTrue(chunks[0].doc_id.startswith("parent#chunk-"))

    def test_load_documents_and_ingest_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "note.md"
            p.write_text("# Titre\n\nUn contenu RAG en markdown.", encoding="utf-8")
            docs = load_documents(tmp)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].metadata["extension"], ".md")
            chunks = ingest_path(tmp, chunk_size=40, chunk_overlap=5)
            self.assertGreaterEqual(len(chunks), 1)

    def test_memory_vectorstore_persist_and_filter(self):
        store = MemoryVectorStore(self.embedder)
        store.add_documents(self.docs)
        result = store.similarity_search("packaging python", top_k=1)
        self.assertEqual(result[0].doc_id, "python")
        filtered = store.similarity_search("python", top_k=2, filters={"kind": "ai"})
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].doc_id, "rag")
        with tempfile.TemporaryDirectory() as tmp:
            store.save(tmp)
            loaded = MemoryVectorStore.load(tmp, self.embedder)
            self.assertEqual(loaded.similarity_search("RAG", top_k=1)[0].doc_id, "rag")

    def test_rag_engine_response(self):
        llm = StaticLLMClient("Le RAG est documenté. [source: rag]")
        engine = RAGEngine.from_documents(self.docs, embedding_model=self.embedder, llm_client=llm, top_k=1)
        response = engine.ask("Qu'est-ce que le RAG ?")
        self.assertEqual(response.answer, "Le RAG est documenté. [source: rag]")
        self.assertIn("rag", response.citations)
        self.assertEqual(len(response.sources), 1)
        self.assertIsInstance(response.to_dict(), dict)

    def test_evaluate_retrieval(self):
        engine = RAGEngine.from_documents(self.docs, embedding_model=self.embedder, top_k=1)
        result = evaluate_retrieval([RetrievalExample("packaging python", {"python"})], engine.retriever, top_k=1)
        self.assertEqual(result.metrics["hit_rate@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
