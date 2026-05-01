from pathlib import Path
import tempfile
import unittest

from cheragh import (
    Document,
    HashingEmbedding,
    RAGEngine,
    StaticLLMClient,
    MarkdownHeaderChunker,
    ExtractiveContextCompressor,
    MultiQueryTransformer,
    index_path,
    inspect_index,
    evaluate_generation,
)


class TestV04Features(unittest.TestCase):
    def test_markdown_header_chunker_preserves_sections(self):
        doc = Document("# Intro\nTexte intro.\n\n## Détails\nBeaucoup de détails sur le RAG.", doc_id="md")
        chunks = MarkdownHeaderChunker(chunk_size=200).split_documents([doc])
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["section"], "Intro")
        self.assertEqual(chunks[1].metadata["section"], "Détails")

    def test_compressor_reduces_context(self):
        docs = [Document("Le chat dort. Le moteur RAG récupère des documents pertinents. Rien à voir.", doc_id="d1")]
        compressed = ExtractiveContextCompressor(max_sentences_per_doc=1).compress("moteur RAG", docs)
        self.assertIn("RAG", compressed[0].content)
        self.assertLess(len(compressed[0].content), len(docs[0].content))

    def test_engine_trace_and_query_transform(self):
        docs = [Document("Le RAG combine retrieval et génération.", doc_id="d1")]
        engine = RAGEngine.from_documents(
            docs,
            embedding_model=HashingEmbedding(64),
            retriever_type="memory",
            llm_client=StaticLLMClient("Réponse [source: d1]"),
            query_transformer=MultiQueryTransformer(num_queries=2),
            compressor="default",
            require_citations=True,
        )
        response = engine.ask("Comment fonctionne le RAG ?")
        self.assertEqual(response.citations, ["d1"])
        self.assertIsNotNone(response.trace)
        self.assertGreaterEqual(len(response.trace.query_variants), 1)
        self.assertIn("trace", response.to_dict())

    def test_incremental_index_and_inspect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "docs"
            root.mkdir()
            (root / "a.md").write_text("# A\nLe RAG indexe les documents.", encoding="utf-8")
            out = Path(tmp) / "index"
            first = index_path(root, out, embedding_model=HashingEmbedding(64), incremental=True)
            second = index_path(root, out, embedding_model=HashingEmbedding(64), incremental=True)
            info = inspect_index(out)
            self.assertGreater(first["indexed_documents"], 0)
            self.assertEqual(second["changed_files"], 0)
            self.assertGreater(info["documents"], 0)

    def test_generation_metrics(self):
        result = evaluate_generation([
            {
                "query": "Qu'est-ce que le RAG ?",
                "answer": "Le RAG utilise des documents. [source: d1]",
                "contexts": ["Le RAG utilise des documents."],
                "source_ids": ["d1"],
            }
        ])
        self.assertEqual(result.metrics["citation_accuracy"], 1.0)
        self.assertGreater(result.metrics["groundedness"], 0.0)


if __name__ == "__main__":
    unittest.main()
