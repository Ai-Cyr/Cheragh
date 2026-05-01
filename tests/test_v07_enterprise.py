import tempfile
import unittest
from pathlib import Path

from cheragh import (
    AccessControlledRAGEngine,
    AccessPolicy,
    Document,
    FeedbackLoop,
    HashingEmbedding,
    MultiTenantRAGEngine,
    Principal,
    RAGEngine,
    SQLRAGEngine,
    StaticLLMClient,
    StructuredRAG,
)


class V07EnterpriseTests(unittest.TestCase):
    def test_sql_rag_engine_rule_based_sum(self):
        engine = SQLRAGEngine.from_records(
            "sales",
            [
                {"client": "Alpha", "revenue": 100, "quarter": "Q1"},
                {"client": "Beta", "revenue": 180, "quarter": "Q1"},
            ],
        )
        response = engine.ask("Quel est le revenu total ?")
        self.assertEqual(response.metadata["architecture"], "sql_rag")
        self.assertIn("SUM", response.metadata["sql"].upper())
        self.assertIn("280", response.answer)

    def test_structured_rag_from_tables(self):
        rag = StructuredRAG.from_tables({"customers": [{"name": "Alice", "score": 10}, {"name": "Bob", "score": 5}]})
        response = rag.ask("Qui a le score le plus grand ?")
        self.assertEqual(response.metadata["architecture"], "structured_rag")
        self.assertIn("Alice", response.answer)

    def test_access_control_filters_documents(self):
        docs = [
            Document("Doc client A public", metadata={"tenant_id": "a", "classification": "public"}, doc_id="a-public"),
            Document("Doc client B secret", metadata={"tenant_id": "b", "classification": "restricted"}, doc_id="b-secret"),
        ]
        engine = RAGEngine.from_documents(
            docs,
            embedding_model=HashingEmbedding(64),
            retriever_type="memory",
            llm_client=StaticLLMClient("Réponse [source: a-public]"),
        )
        guarded = AccessControlledRAGEngine(engine, policy=AccessPolicy(require_tenant_match=True))
        response = guarded.ask("client", principal=Principal(user_id="u1", tenant_ids={"a"}, max_classification="internal"))
        self.assertEqual([src.doc_id for src in response.sources], ["a-public"])
        self.assertEqual(response.metadata["access_control"]["denied_documents"], 1)

    def test_multi_tenant_engine_routes_collection_and_enforces_acl(self):
        docs_a = [Document("Contrat tenant A", metadata={"tenant_id": "tenant-a", "classification": "internal"}, doc_id="a1")]
        docs_b = [Document("Contrat tenant B", metadata={"tenant_id": "tenant-b", "classification": "internal"}, doc_id="b1")]
        embedder = HashingEmbedding(64)
        engine_a = RAGEngine.from_documents(docs_a, embedding_model=embedder, retriever_type="memory", llm_client=StaticLLMClient("A [source: a1]"))
        engine_b = RAGEngine.from_documents(docs_b, embedding_model=embedder, retriever_type="memory", llm_client=StaticLLMClient("B [source: b1]"))
        mt = MultiTenantRAGEngine()
        mt.add_collection("tenant-a", "contracts", engine_a, default=True)
        mt.add_collection("tenant-b", "contracts", engine_b, default=True)
        response = mt.ask("contrat", tenant_id="tenant-a", collection_id="contracts", principal={"user_id": "u", "tenant_ids": ["tenant-a"]})
        self.assertEqual(response.metadata["tenant"]["tenant_id"], "tenant-a")
        self.assertEqual(response.sources[0].doc_id, "a1")
        self.assertEqual(mt.stats()["tenant_count"], 2)

    def test_feedback_loop_jsonl_and_eval_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feedback.jsonl"
            loop = FeedbackLoop.from_jsonl(path) if hasattr(FeedbackLoop, "from_jsonl") else FeedbackLoop()
            # support both direct store and helper method if present
            if not hasattr(FeedbackLoop, "from_jsonl"):
                from cheragh.feedback import JSONLFeedbackStore
                loop = FeedbackLoop(JSONLFeedbackStore(path))
            record = loop.log_feedback(
                query="Question",
                rating="bad",
                answer="Mauvaise réponse",
                correct_answer="Bonne réponse",
                correct_source_ids=["doc-1"],
                tenant_id="tenant-a",
            )
            self.assertEqual(record.rating, "negative")
            summary = loop.summary()
            self.assertEqual(summary.total, 1)
            dataset = loop.export_evalset(only_negative=True)
            self.assertEqual(dataset[0]["expected_doc_ids"], ["doc-1"])


if __name__ == "__main__":
    unittest.main()
