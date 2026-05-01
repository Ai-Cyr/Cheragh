import unittest

from cheragh import (
    Document,
    HashingEmbedding,
    HybridSearchRetriever,
    RetrievalExample,
    RetrievalTokenizer,
    chunk_documents,
    evaluate_retrieval,
    metadata_matches,
)
from cheragh.citations import citation_location
from cheragh.ingestion import TokenTextChunker


class V090QualityTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(
                "La sécurité SQLite read-only empêche les écritures dangereuses.",
                doc_id="sqlite",
                metadata={"tenant": "acme", "lang": "fr", "quality": 0.95, "tags": ["security", "sql"]},
            ),
            Document(
                "Le cache signé protège les entrées persistantes contre la falsification.",
                doc_id="cache",
                metadata={"tenant": "acme", "lang": "fr", "quality": 0.9, "tags": ["security", "cache"]},
            ),
            Document(
                "A coffee machine heats water.",
                doc_id="coffee",
                metadata={"tenant": "other", "lang": "en", "quality": 0.4, "tags": ["demo"]},
            ),
        ]
        self.embedder = HashingEmbedding(dimension=64)

    def test_retrieval_tokenizer_handles_accents_hyphen_and_ngrams(self):
        tokenizer = RetrievalTokenizer(ngram_range=(1, 2))
        tokens = tokenizer.tokenize("Sécurité read-only du cache")
        self.assertIn("securite", tokens)
        self.assertIn("read-only", tokens)
        self.assertIn("read", tokens)
        self.assertIn("only", tokens)
        self.assertIn("securite read-only", tokens)

    def test_hybrid_filters_support_metadata_operators(self):
        retriever = HybridSearchRetriever(self.docs, self.embedder, alpha=0.2)
        results = retriever.retrieve(
            "sécurité",
            top_k=5,
            filters={"tenant": "acme", "quality": {"$gte": 0.9}, "tags": {"$contains": "sql"}},
        )
        self.assertEqual([doc.doc_id for doc in results], ["sqlite"])
        self.assertTrue(metadata_matches(self.docs[0].metadata, {"lang": {"$in": ["fr"]}}))
        self.assertFalse(metadata_matches(self.docs[2].metadata, {"tenant": {"$ne": "other"}}))

    def test_chunkers_add_citation_offsets(self):
        text = "Intro courte.\n\nLa clause importante commence ici et continue longtemps. Fin."
        chunks = chunk_documents([Document(text, doc_id="contract")], chunk_size=45, chunk_overlap=8, min_chunk_size=1)
        self.assertGreaterEqual(len(chunks), 2)
        first = chunks[0]
        self.assertEqual(first.metadata["parent_doc_id"], "contract")
        self.assertIn("source_char_start", first.metadata)
        self.assertIn("source_char_end", first.metadata)
        start = first.metadata["source_char_start"]
        end = first.metadata["source_char_end"]
        self.assertEqual(text[start:end].replace("\n\n", "\n\n").strip(), first.content.strip())
        self.assertIn("chars=", citation_location(first))

    def test_token_chunker_offsets(self):
        text = "un deux trois quatre cinq six"
        chunks = TokenTextChunker(chunk_size=3, chunk_overlap=1).split_documents([Document(text, doc_id="tok")])
        self.assertEqual(chunks[0].content, "un deux trois")
        self.assertEqual(text[chunks[1].metadata["source_char_start"] : chunks[1].metadata["source_char_end"]], chunks[1].content)

    def test_retrieval_metrics_include_recall_ndcg_context_precision_and_parent_match(self):
        chunks = chunk_documents([Document("SQLite read-only mode. Cache signé.", doc_id="guide")], chunk_size=20, chunk_overlap=5, min_chunk_size=1)
        retriever = HybridSearchRetriever(chunks + self.docs, self.embedder, alpha=0.1)
        result = evaluate_retrieval([RetrievalExample("read only sqlite", {"guide"})], retriever, top_k=3)
        self.assertIn("recall@3", result.metrics)
        self.assertIn("ndcg@3", result.metrics)
        self.assertIn("context_precision@3", result.metrics)
        self.assertGreater(result.metrics["recall@3"], 0.0)
        self.assertGreater(result.metrics["ndcg@3"], 0.0)


if __name__ == "__main__":
    unittest.main()
