from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cheragh import Document, EmbeddingModel, RAGEngine, StaticLLMClient
from cheragh.hybrid_search import BM25Retriever
from cheragh.reranking import ReciprocalRankFusionReranker, ReciprocalRankFusionRetriever


class _NeverEmbed(EmbeddingModel):
    def embed_documents(self, texts):
        raise AssertionError("BM25 must not embed documents")

    def embed_query(self, text):
        raise AssertionError("BM25 must not embed queries")


class _RankedRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def retrieve(self, query: str, top_k: int = 5):
        self.calls.append(top_k)
        return self.documents[:top_k]


class BM25RetrieverTests(unittest.TestCase):
    def test_sparse_only_retrieval_filters_and_snapshots_documents(self):
        source = Document("python packaging wheel", doc_id="python", metadata={"lang": "en"})
        retriever = BM25Retriever(
            [source, Document("recette cuisine", doc_id="food", metadata={"lang": "fr"})]
        )
        source.content = "mutated"
        source.metadata["lang"] = "fr"

        results = retriever.retrieve("python wheel", top_k=2, filters={"lang": "en"})

        self.assertEqual([document.doc_id for document in results], ["python"])
        self.assertEqual(results[0].content, "python packaging wheel")
        self.assertGreater(results[0].score, 0.0)
        self.assertEqual(results[0].metadata["bm25_score"], results[0].score)

        source.metadata["nested"] = {"owner": "index"}
        snapshot_retriever = BM25Retriever([source])
        isolated = snapshot_retriever.retrieve("python", top_k=1)[0]
        isolated.metadata["nested"]["owner"] = "caller"
        self.assertEqual(snapshot_retriever.documents[0].metadata["nested"]["owner"], "index")

    def test_engine_factory_has_a_real_bm25_mode_without_embeddings(self):
        engine = RAGEngine.from_documents(
            [Document("alpha exact token", doc_id="a"), Document("beta", doc_id="b")],
            embedding_model=_NeverEmbed(),
            llm_client=StaticLLMClient("answer [source: a]"),
            retriever_type="bm25",
            top_k=1,
        )

        response = engine.ask("exact token")

        self.assertIsInstance(engine.retriever, BM25Retriever)
        self.assertEqual(response.sources[0].doc_id, "a")

    def test_bm25_parameters_and_top_k_are_strict(self):
        with self.assertRaises(ValueError):
            BM25Retriever([], k1=0)
        with self.assertRaises(ValueError):
            BM25Retriever([], b=1.1)
        with self.assertRaises(TypeError):
            BM25Retriever([], k1=True)
        with self.assertRaises(TypeError):
            BM25Retriever([], b="0.5")
        retriever = BM25Retriever([Document("one", doc_id="one")])
        for value, error in ((0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)):
            with self.subTest(value=value), self.assertRaises(error):
                retriever.retrieve("one", top_k=value)

    def test_bm25_config_does_not_construct_an_unused_embedding_provider(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "rag.json"
            config.write_text(
                '{"embedding":{"provider":"openai","model":"unused"},'
                '"retriever":{"type":"bm25","top_k":1},'
                '"generation":{"provider":"extractive"}}',
                encoding="utf-8",
            )
            with patch("cheragh.engine._embedding_from_config", side_effect=AssertionError("unused")):
                engine = RAGEngine.from_config(
                    config,
                    documents=[Document("sparse token", doc_id="sparse")],
                )

        self.assertIsInstance(engine.retriever, BM25Retriever)


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_multi_source_retriever_performs_canonical_rrf(self):
        a = Document("a", doc_id="a")
        b = Document("b", doc_id="b")
        c = Document("c", doc_id="c")
        first = _RankedRetriever([a, b, c])
        second = _RankedRetriever([b, c, a])
        retriever = ReciprocalRankFusionRetriever([first, second], candidate_top_k=3, k=60)

        results = retriever.retrieve("query", top_k=3)

        self.assertEqual([document.doc_id for document in results], ["b", "a", "c"])
        self.assertEqual(first.calls, [3])
        self.assertEqual(second.calls, [3])
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual(results[0].metadata["retrieval_method"], "reciprocal-rank-fusion")
        self.assertEqual(results[0].metadata["rrf_sources"], 2)

    def test_rrf_deduplicates_per_source_and_qualifies_identity_keys(self):
        duplicate = Document("same", doc_id="shared")
        id_looking_like_content = Document("other", doc_id="plain")
        anonymous = Document("plain")
        first = _RankedRetriever([duplicate, duplicate, id_looking_like_content, anonymous])
        second = _RankedRetriever([Document("same", doc_id="shared")])
        retriever = ReciprocalRankFusionRetriever([first, second], candidate_top_k=4, k=60)

        results = retriever.retrieve("query", top_k=3)
        by_id = {document.doc_id: document for document in results}

        self.assertAlmostEqual(by_id["shared"].score, 1 / 61 + 1 / 61)
        self.assertEqual(by_id["shared"].metadata["rrf_sources"], 2)
        self.assertEqual(by_id["plain"].metadata["rrf_sources"], 1)
        self.assertIn(None, by_id)

    def test_single_list_reranker_assigns_rrf_scores_and_validates_inputs(self):
        reranker = ReciprocalRankFusionReranker(k=10)
        results = reranker.rerank(
            "unused",
            [Document("a", doc_id="a", score=99.0), Document("b", doc_id="b", score=1.0)],
            top_k=2,
        )

        self.assertEqual([document.doc_id for document in results], ["a", "b"])
        self.assertAlmostEqual(results[0].score, 1 / 11)
        self.assertEqual(results[0].metadata["first_stage_score"], 99.0)
        with self.assertRaises(ValueError):
            ReciprocalRankFusionRetriever([], candidate_top_k=5)
        with self.assertRaises(ValueError):
            ReciprocalRankFusionReranker(k=0)


if __name__ == "__main__":
    unittest.main()
