"""Regression tests for the fixes applied during the code review pass."""
import sqlite3
import unittest

from cheragh import Document, HashingEmbedding
from cheragh.adaptive import AdaptiveRetriever, GateDecision
from cheragh.base import BaseRetriever, LLMClient
from cheragh.cache import CachedRetriever, MemoryCache
from cheragh.corrective_rag import CorrectiveRAGRetriever
from cheragh.evaluation.retrieval import RetrievalExample, evaluate_retrieval
from cheragh.filters import metadata_matches
from cheragh.graph.engine import GraphRAGRetriever, build_knowledge_graph, extract_triples
from cheragh.ingestion.chunkers.recursive import RecursiveTextChunker
from cheragh.ingestion.chunkers.structured import SentenceWindowChunker
from cheragh.ingestion.chunkers.token import TokenTextChunker
from cheragh.retrieval.parent_child import ParentChildRetriever
from cheragh.self_rag.engine import LexicalEvidenceCritic
from cheragh.structured.engine import SQLRAGEngine
from cheragh.vectorstores.memory import MemoryVectorStore


class _StaticRetriever(BaseRetriever):
    def __init__(self, documents):
        self.documents = list(documents)

    def retrieve(self, query: str, top_k: int = 5):
        return self.documents[:top_k]


class _StaticLLM(LLMClient):
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str, **kwargs) -> str:
        return self.response

    def stream(self, prompt: str, **kwargs):
        yield self.generate(prompt, **kwargs)


class MetadataFilterUnhashableTests(unittest.TestCase):
    def test_in_operator_with_unhashable_metadata_value(self):
        self.assertFalse(metadata_matches({"tags": ["legal", "hr"]}, {"tags": {"$in": ["legal"]}}))
        self.assertTrue(metadata_matches({"tags": ["legal"]}, {"tags": {"$in": [["legal"]]}}))
        self.assertTrue(metadata_matches({"tags": ["legal"]}, {"tags": {"$nin": ["other"]}}))

    def test_membership_shortcut_with_set_filter_and_unhashable_value(self):
        self.assertFalse(metadata_matches({"tags": ["a"]}, {"tags": {"a", "b"}}))

    def test_retrieval_does_not_crash_on_list_metadata(self):
        store = MemoryVectorStore(HashingEmbedding())
        store.add_documents([Document("contract law", metadata={"tags": ["legal", "hr"]}, doc_id="d1")])
        retriever = store.as_retriever(filters={"tags": {"$in": ["legal"]}})
        self.assertEqual(retriever.retrieve("contract"), [])


class ParentChildGeneratorTests(unittest.TestCase):
    def test_from_hierarchical_chunks_accepts_generator(self):
        docs = [
            Document("the capital of france is paris", doc_id="a"),
            Document("bananas are yellow", doc_id="b"),
        ]
        retriever = ParentChildRetriever.from_hierarchical_chunks(iter(docs))
        results = retriever.retrieve("capital of france", top_k=2)
        self.assertTrue(results)


class CachedRetrieverFreshnessTests(unittest.TestCase):
    def test_documents_added_after_wrapping_invalidate_cache(self):
        store = MemoryVectorStore(HashingEmbedding())
        store.add_documents([Document("paris is the capital of france", doc_id="d1")])
        cached = CachedRetriever(store.as_retriever(), MemoryCache())
        cached.retrieve("capital", top_k=5)
        store.add_documents([Document("the capital gains tax rose", doc_id="d2")])
        ids = {doc.doc_id for doc in cached.retrieve("capital", top_k=5)}
        self.assertIn("d2", ids)

    def test_explicit_fingerprint_stays_pinned(self):
        store = MemoryVectorStore(HashingEmbedding())
        cached = CachedRetriever(store.as_retriever(), MemoryCache(), fingerprint="index-v7")
        self.assertEqual(cached.fingerprint, "index-v7")


class ChunkerOverlapTests(unittest.TestCase):
    def test_recursive_overlap_keeps_separator_and_offsets(self):
        chunker = RecursiveTextChunker(chunk_size=40, chunk_overlap=15, min_chunk_size=5)
        text = "The cat sat on the mat. The bird flew away. The fish swam. The dog barked loudly."
        chunks = chunker.split_text_with_offsets(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            # No glued sentences such as "away.The", and exact source spans.
            self.assertNotRegex(chunk.text, r"[a-z]\.[A-Z]")
            self.assertEqual(text[chunk.source_char_start : chunk.source_char_end], chunk.text)

    def test_token_chunker_no_trailing_contained_window(self):
        chunker = TokenTextChunker(chunk_size=250, chunk_overlap=40)
        exactly_one_window = " ".join(f"w{i}" for i in range(250))
        self.assertEqual(len(chunker.split_text(exactly_one_window)), 1)
        longer = " ".join(f"w{i}" for i in range(430))
        lengths = [len(chunk.split()) for chunk in chunker.split_text(longer)]
        self.assertEqual(lengths, [250, 220])

    def test_sentence_window_no_duplicate_final_window(self):
        chunker = SentenceWindowChunker(window_size=5, window_overlap=1, min_chunk_size=1)
        text = "One one. Two two. Three three. Four four. Five five."
        docs = chunker.split_documents([Document(text, doc_id="d")])
        self.assertEqual(len(docs), 1)

    def test_character_fallback_no_contained_final_window(self):
        chunker = RecursiveTextChunker(chunk_size=250, chunk_overlap=30, min_chunk_size=1)
        pieces = chunker._split_recursive("x" * 450, ())
        self.assertEqual([len(piece) for piece in pieces], [250, 230])


class RetrievalMetricBoundsTests(unittest.TestCase):
    def test_chunked_corpus_metrics_bounded_by_one(self):
        docs = [
            Document("text", doc_id=f"doc#chunk-{i}", metadata={"parent_doc_id": "doc"})
            for i in range(3)
        ]
        example = RetrievalExample(query="q", expected_doc_ids={"doc"})
        result = evaluate_retrieval([example], _StaticRetriever(docs), top_k=3)
        self.assertLessEqual(result.metrics["ndcg@3"], 1.0)
        self.assertLessEqual(result.metrics["context_precision@3"], 1.0)

    def test_examples_are_not_mutated(self):
        example = RetrievalExample(query="q", expected_doc_ids={"doc"})
        evaluate_retrieval([example], _StaticRetriever([Document("text", doc_id="doc")]), top_k=1)
        self.assertEqual(example.graded_relevance, {})

    def test_perfect_retrieval_still_scores_one(self):
        docs = [Document("a", doc_id="d1"), Document("b", doc_id="d2")]
        example = RetrievalExample(query="q", expected_doc_ids={"d1", "d2"})
        result = evaluate_retrieval([example], _StaticRetriever(docs), top_k=2)
        self.assertEqual(result.metrics["ndcg@2"], 1.0)
        self.assertEqual(result.metrics["context_precision@2"], 1.0)


class CorrectiveRAGDedupTests(unittest.TestCase):
    def test_retries_do_not_duplicate_documents(self):
        docs = [Document("alpha", doc_id="d1"), Document("beta", doc_id="d2")]
        retriever = CorrectiveRAGRetriever(
            _StaticRetriever(docs), _StaticLLM("ambiguous"), max_retries=1, min_correct=3
        )
        results = retriever.retrieve("q", top_k=4)
        self.assertEqual([doc.doc_id for doc in results], ["d1", "d2"])

    def test_idless_documents_are_deduplicated_by_content(self):
        docs = [Document("alpha"), Document("beta")]
        retriever = CorrectiveRAGRetriever(
            _StaticRetriever(docs), _StaticLLM("ambiguous"), max_retries=2, min_correct=3
        )
        results = retriever.retrieve("q", top_k=6)
        self.assertEqual([doc.content for doc in results], ["alpha", "beta"])


class AdaptiveGateParsingTests(unittest.TestCase):
    def _decision(self, response: str) -> GateDecision:
        retriever = AdaptiveRetriever(_StaticRetriever([]), _StaticLLM(response))
        return retriever._decide("question")

    def test_explicit_retrieve_wins_over_no_prefix(self):
        self.assertEqual(
            self._decision("Non, cette question nécessite une recherche. RETRIEVE"),
            GateDecision.RETRIEVE,
        )
        self.assertEqual(self._decision("NOTE: RETRIEVE"), GateDecision.RETRIEVE)

    def test_no_retrieve_still_detected(self):
        self.assertEqual(self._decision("NO_RETRIEVE"), GateDecision.NO_RETRIEVE)
        self.assertEqual(self._decision("no"), GateDecision.NO_RETRIEVE)


class GraphMutationTests(unittest.TestCase):
    def test_build_knowledge_graph_does_not_mutate_documents(self):
        docs = [Document("Marie travaille avec Pierre."), Document("Pierre contient Bidule.")]
        build_knowledge_graph(docs)
        self.assertEqual([doc.doc_id for doc in docs], [None, None])

    def test_graph_retriever_does_not_mutate_documents(self):
        docs = [Document("Marie travaille avec Pierre.")]
        GraphRAGRetriever(docs, build_knowledge_graph(docs))
        self.assertIsNone(docs[0].doc_id)

    def test_triples_have_independent_metadata(self):
        triples = extract_triples(
            "Alice travaille avec Bob. Bob collabore avec Carol.", doc_id="d", metadata={"k": 1}
        )
        self.assertGreater(len(triples), 1)
        triples[0].metadata["injected"] = True
        self.assertNotIn("injected", triples[1].metadata)


class SelfRAGCriticTests(unittest.TestCase):
    def test_termless_claims_do_not_inflate_support_score(self):
        critic = LexicalEvidenceCritic()
        evidence = [Document("Le budget atteint plusieurs millions cette année.")]
        assessment = critic.assess_support("q", "Le budget atteint plusieurs millions. 42.", evidence)
        if assessment.unsupported_claims:
            self.assertLess(assessment.score, 1.0)
            self.assertFalse(assessment.supported)


class SemanticChunkerCosineTests(unittest.TestCase):
    def test_distances_invariant_to_embedding_scale(self):
        import numpy as np

        from cheragh.semantic_chunker import SemanticChunker

        embeddings = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        base = SemanticChunker._consecutive_distances(embeddings)
        scaled = SemanticChunker._consecutive_distances(embeddings * 10.0)
        np.testing.assert_allclose(base, scaled)
        self.assertAlmostEqual(base[0], 1.0 - 1.0 / np.sqrt(2.0))


class SQLValidationTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE public_t (a TEXT)")
        self.conn.execute("CREATE TABLE secret_t (ssn TEXT)")
        self.conn.execute("INSERT INTO secret_t VALUES ('123-45-6789')")
        self.conn.execute("INSERT INTO public_t VALUES ('update me')")
        self.conn.commit()
        self.engine = SQLRAGEngine(connection=self.conn, table_allowlist=["public_t"], read_only=True)

    def test_comma_join_cannot_bypass_allowlist(self):
        for query in (
            "SELECT secret_t.ssn FROM public_t, secret_t",
            "SELECT * FROM public_t, secret_t",
            "SELECT * FROM public_t , secret_t s",
        ):
            with self.assertRaises(ValueError):
                self.engine.validate_sql(query)

    def test_keywords_inside_string_literals_are_allowed(self):
        result = self.engine.execute_sql("SELECT * FROM public_t WHERE a = 'update me'")
        self.assertEqual(result.rows, [{"a": "update me"}])
        self.engine.validate_sql("SELECT * FROM public_t WHERE a = 'it''s a delete; test'")

    def test_case_insensitive_allowlist_and_cte_names(self):
        self.engine.validate_sql("SELECT * FROM Public_T")
        self.engine.validate_sql("WITH c AS (SELECT * FROM public_t) SELECT * FROM c")
        with self.assertRaises(ValueError):
            self.engine.validate_sql("WITH c AS (SELECT * FROM secret_t) SELECT * FROM c")

    def test_existing_protections_still_hold(self):
        for query in (
            "SELECT * FROM public_t JOIN secret_t ON 1=1",
            "SELECT * FROM (SELECT * FROM secret_t)",
            "DELETE FROM public_t",
            "SELECT 1; DROP TABLE public_t",
        ):
            with self.assertRaises(ValueError):
                self.engine.validate_sql(query)


if __name__ == "__main__":
    unittest.main()
