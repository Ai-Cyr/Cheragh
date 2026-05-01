import unittest

from cheragh import Document, EmbeddingModel, QueryRouter
from cheragh.ingestion import (
    CodeChunker,
    HierarchicalChunker,
    PDFLayoutChunker,
    SemanticChunker,
    TableChunker,
)


class TinySemanticEmbedding(EmbeddingModel):
    def embed_documents(self, texts):
        import numpy as np

        return np.vstack([self.embed_query(text) for text in texts])

    def embed_query(self, text):
        import numpy as np

        lower = text.lower()
        if any(word in lower for word in ["chat", "chien", "animal"]):
            vec = np.array([1.0, 0.0, 0.0], dtype=float)
        elif any(word in lower for word in ["rag", "retrieval", "document"]):
            vec = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            vec = np.array([0.0, 0.0, 1.0], dtype=float)
        return vec


class TestV041ChunkingAndRouting(unittest.TestCase):
    def test_semantic_chunker_breaks_on_topic_shift(self):
        doc = Document(
            "Le chat dort sur le canapé. Le chien joue dans le jardin. "
            "Le RAG récupère des documents. Le retrieval améliore la réponse.",
            doc_id="d",
        )
        chunks = SemanticChunker(
            embedding_model=TinySemanticEmbedding(),
            breakpoint_threshold=0.75,
            min_chunk_size=10,
        ).split_documents([doc])
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("chat", chunks[0].content.lower())
        self.assertIn("rag", chunks[-1].content.lower())
        self.assertEqual(chunks[0].metadata["chunk_method"], "semantic")

    def test_code_chunker_splits_python_symbols(self):
        doc = Document(
            "import os\n\nclass Loader:\n    pass\n\ndef run():\n    return 1\n",
            metadata={"source": "app.py"},
            doc_id="code",
        )
        chunks = CodeChunker().split_documents([doc])
        symbols = {chunk.metadata.get("symbol_name") for chunk in chunks}
        self.assertIn("Loader", symbols)
        self.assertIn("run", symbols)
        self.assertTrue(all(chunk.metadata["code_language"] == "python" for chunk in chunks))

    def test_table_chunker_extracts_markdown_table(self):
        doc = Document("Intro\n\n| Produit | CA |\n| --- | ---: |\n| A | 10 |\n| B | 20 |", doc_id="table")
        chunks = TableChunker(rows_per_chunk=1).split_documents([doc])
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["chunk_method"], "table")
        self.assertEqual(chunks[0].metadata["column_count"], 2)
        self.assertIn("Produit", chunks[0].content)

    def test_pdf_layout_chunker_keeps_page_metadata(self):
        doc = Document(
            "1 Introduction\n\nCe paragraphe décrit le document.\n\nTABLE 1 Résultats",
            metadata={"source": "paper.pdf", "page": 3},
            doc_id="pdf-p3",
        )
        chunks = PDFLayoutChunker(min_chunk_size=5).split_documents([doc])
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["page"], 3)
        self.assertEqual(chunks[0].metadata["chunk_method"], "pdf-layout")

    def test_hierarchical_chunker_adds_parent_and_child_metadata(self):
        doc = Document("# A\nTexte A.\n\n## B\nTexte B très important.", doc_id="md")
        chunks = HierarchicalChunker(chunk_size=30, chunk_overlap=5, min_chunk_size=5, include_parent_sections=True).split_documents([doc])
        roles = {chunk.metadata.get("chunk_role") for chunk in chunks}
        self.assertIn("parent_section", roles)
        self.assertIn("child_chunk", roles)
        self.assertTrue(any("A" in chunk.metadata.get("section_path", "") for chunk in chunks))

    def test_query_router_selects_sql_summary_and_qa_routes(self):
        router = QueryRouter(
            routes={
                "qa": lambda query, **_: {"answer": f"qa:{query}"},
                "summary": lambda query, **_: {"answer": f"summary:{query}"},
                "sql": lambda query, **_: {"answer": f"sql:{query}"},
            }
        )
        sql_response = router.ask("Compare les ventes Q1 et Q2")
        self.assertEqual(sql_response["routing"]["route"], "sql")
        self.assertEqual(sql_response["answer"].split(":", 1)[0], "sql")

        summary_response = router.ask("Résume ce document")
        self.assertEqual(summary_response["routing"]["route"], "summary")

        qa_response = router.ask("Quelle est la politique de remboursement ?")
        self.assertEqual(qa_response["routing"]["route"], "qa")


if __name__ == "__main__":
    unittest.main()
