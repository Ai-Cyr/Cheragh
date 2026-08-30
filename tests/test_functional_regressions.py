from __future__ import annotations

import unittest
from pathlib import Path

from cheragh import Document, FeedbackLoop, LongContextPacker, RAGEngine, RAGResponse, StaticLLMClient
from cheragh.base import BaseRetriever
from cheragh.cache import MemoryCache
from cheragh.evaluation import evaluate_retrieval
from cheragh.security import AccessControlledRAGEngine, AccessControlledRetriever, AccessPolicy, Principal


class _RankedPrefixRetriever(BaseRetriever):
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.top_k_calls: list[int] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        self.top_k_calls.append(top_k)
        return self.documents[:top_k]


class ResponseFeedbackContractTests(unittest.TestCase):
    def test_feedback_export_round_trips_into_retrieval_evaluation(self) -> None:
        retriever = _RankedPrefixRetriever([Document("Evidence", doc_id="doc-1", score=1.0)])
        response = RAGEngine(
            retriever,
            llm_client=StaticLLMClient("Answer [source: doc-1]"),
        ).ask("Question")

        self.assertIsNotNone(response.trace)
        assert response.trace is not None
        self.assertEqual(response.response_id, response.trace.request_id)
        self.assertEqual(response.to_dict()["response_id"], response.response_id)

        loop = FeedbackLoop()
        record = loop.log_feedback(
            query=response.query,
            rating="bad",
            response=response,
            correct_source_ids=["doc-1"],
        )
        dataset = loop.export_evalset(only_negative=True)

        self.assertEqual(record.response_id, response.response_id)
        self.assertEqual(dataset[0]["query"], "Question")
        self.assertNotIn("question", dataset[0])
        self.assertEqual(dataset[0]["metadata"]["response_id"], response.response_id)
        result = evaluate_retrieval(dataset, retriever, top_k=1)
        self.assertEqual(result.metrics["hit_rate@1"], 1.0)

    def test_retrieval_evaluation_accepts_legacy_question_key(self) -> None:
        retriever = _RankedPrefixRetriever([Document("Evidence", doc_id="doc-1")])

        result = evaluate_retrieval(
            [{"question": "Legacy question", "expected_doc_ids": ["doc-1"]}],
            retriever,
            top_k=1,
        )

        self.assertEqual(result.rows[0]["query"], "Legacy question")
        self.assertEqual(result.metrics["hit_rate@1"], 1.0)

    def test_response_without_trace_still_has_a_stable_identifier(self) -> None:
        response = RAGResponse(query="q", answer="a", sources=[], retrieved_documents=[], prompt="")

        self.assertIsInstance(response.response_id, str)
        self.assertTrue(response.response_id)
        self.assertEqual(response.to_dict()["response_id"], response.response_id)


class AccessControlRetrievalTests(unittest.TestCase):
    @staticmethod
    def _principal() -> Principal:
        return Principal(user_id="user-1", tenant_ids={"mine"})

    @staticmethod
    def _documents() -> list[Document]:
        denied = [
            Document(f"Denied {index}", metadata={"tenant_id": "other"}, doc_id=f"denied-{index}")
            for index in range(30)
        ]
        return denied + [Document("Allowed", metadata={"tenant_id": "mine"}, doc_id="allowed")]

    def test_acl_progressively_scans_until_it_finds_authorized_documents(self) -> None:
        ranked = _RankedPrefixRetriever(self._documents())
        retriever = AccessControlledRetriever(
            ranked,
            principal=self._principal(),
            policy=AccessPolicy(require_tenant_match=True),
            overfetch_factor=4,
            max_candidates=64,
        )

        documents = retriever.retrieve("query", top_k=1)

        self.assertEqual([document.doc_id for document in documents], ["allowed"])
        self.assertEqual(ranked.top_k_calls, [4, 8, 16, 32])
        self.assertEqual(retriever.last_scanned_count, 31)
        self.assertEqual(retriever.last_denied_count, 30)
        self.assertFalse(retriever.last_candidate_limit_reached)

    def test_acl_stops_at_configured_candidate_limit(self) -> None:
        ranked = _RankedPrefixRetriever(self._documents())
        retriever = AccessControlledRetriever(
            ranked,
            principal=self._principal(),
            policy=AccessPolicy(require_tenant_match=True),
            overfetch_factor=4,
            max_candidates=16,
        )

        documents = retriever.retrieve("query", top_k=1)

        self.assertEqual(documents, [])
        self.assertEqual(ranked.top_k_calls, [4, 8, 16])
        self.assertEqual(retriever.last_scanned_count, 16)
        self.assertEqual(retriever.last_denied_count, 16)
        self.assertTrue(retriever.last_candidate_limit_reached)

    def test_acl_engine_preserves_complete_base_engine_configuration(self) -> None:
        ranked = _RankedPrefixRetriever(
            [Document("Allowed", metadata={"tenant_id": "mine"}, doc_id="allowed")]
        )
        cache = MemoryCache(max_entries=8)
        packer = LongContextPacker(128)
        engine = RAGEngine(
            ranked,
            llm_client=StaticLLMClient("Answer [source: allowed]"),
            context_packer=packer,
            cache_backend=cache,
            cache_config={"retrieval_ttl": 30},
            trace_export_path=Path("traces.jsonl"),
            trace_include_prompt=True,
            trace_pricing={"input_per_1k": 0.01},
        )
        guarded = AccessControlledRAGEngine(
            engine,
            policy=AccessPolicy(require_tenant_match=True),
            max_candidates=32,
        )

        scoped = guarded.for_principal(self._principal())

        self.assertIsNot(scoped, engine)
        self.assertIsInstance(scoped.retriever, AccessControlledRetriever)
        self.assertIs(engine.retriever, ranked)
        self.assertIs(scoped.context_packer, packer)
        self.assertIs(scoped.cache_backend, cache)
        self.assertEqual(scoped.cache_config, engine.cache_config)
        self.assertEqual(scoped.trace_export_path, engine.trace_export_path)
        self.assertEqual(scoped.trace_include_prompt, engine.trace_include_prompt)
        self.assertEqual(scoped.trace_pricing, engine.trace_pricing)

    def test_acl_engine_reports_scan_diagnostics(self) -> None:
        ranked = _RankedPrefixRetriever(self._documents())
        guarded = AccessControlledRAGEngine(
            RAGEngine(ranked, llm_client=StaticLLMClient("Answer [source: allowed]")),
            policy=AccessPolicy(require_tenant_match=True),
            max_candidates=64,
        )

        response = guarded.ask("query", principal=self._principal(), top_k=1)

        diagnostics = response.metadata["access_control"]
        self.assertEqual(diagnostics["scanned_documents"], 31)
        self.assertEqual(diagnostics["denied_documents"], 30)
        self.assertFalse(diagnostics["candidate_limit_reached"])


if __name__ == "__main__":
    unittest.main()
