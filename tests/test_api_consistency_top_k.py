import unittest

from cheragh.adaptive import AdaptiveRetriever
from cheragh.base import BaseRetriever, Document
from cheragh.corrective_rag import CorrectiveRAGRetriever
from cheragh.graph.engine import GraphRAGRetriever, KnowledgeGraph
from cheragh.pipeline import AdvancedRAGPipeline
from cheragh.raptor_engine.engine import RAPTORIndex, RAPTORRetrieverV2
from cheragh.reranking import KeywordOverlapReranker, RerankingRetriever
from cheragh.retrieval.parent_child import ParentChildRetriever
from cheragh.router import EnsembleRetriever
from cheragh.routing.router import QueryRouter as ApplicationQueryRouter
from cheragh.security.access_control import AccessControlledRetriever
from cheragh.tenancy.engine import MultiTenantRAGEngine
from cheragh.workflow.nodes import RetrieveNode


class _RecordingRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.calls: list[int] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        self.calls.append(top_k)
        return [Document("alpha", doc_id="doc-1", score=1.0)][:top_k]


class _StaticLLM:
    def generate(self, prompt: str, **kwargs: object) -> str:
        return "RETRIEVE"


class TopKContractTests(unittest.TestCase):
    def test_public_top_k_apis_share_strict_positive_integer_contract(self) -> None:
        base = _RecordingRetriever()
        llm = _StaticLLM()
        keyword = KeywordOverlapReranker()
        parent_child = ParentChildRetriever(
            [Document("parent", doc_id="parent-1")],
            child_documents=[Document("child", metadata={"parent_doc_id": "parent-1"})],
            child_retriever=base,
        )
        workflow_node = RetrieveNode(base)
        application_router = ApplicationQueryRouter({"default": base})

        cases = {
            "legacy_retriever": lambda value: AdaptiveRetriever(base, llm).retrieve("query", top_k=value),
            "legacy_corrective_retriever": lambda value: CorrectiveRAGRetriever(base, llm).retrieve(
                "query", top_k=value
            ),
            "graph_retriever": lambda value: GraphRAGRetriever(
                [], KnowledgeGraph(), fallback_retriever=base
            ).retrieve("query", top_k=value),
            "raptor_retriever": lambda value: RAPTORRetrieverV2(RAPTORIndex()).retrieve(
                "query", top_k=value
            ),
            "reranker": lambda value: keyword.rerank("query", [], top_k=value),
            "reranking_retriever": lambda value: RerankingRetriever(
                base, reranker=keyword
            ).retrieve("query", top_k=value),
            "ensemble_retriever": lambda value: EnsembleRetriever([base]).retrieve(
                "query", top_k=value
            ),
            "parent_child_retriever": lambda value: parent_child.retrieve("query", top_k=value),
            "access_controlled_retriever": lambda value: AccessControlledRetriever(
                base, principal=None
            ).retrieve("query", top_k=value),
            "tenant_retrieval": lambda value: MultiTenantRAGEngine().retrieve(
                "query", "missing-tenant", top_k=value
            ),
            "application_router": lambda value: application_router.ask("query", top_k=value),
            "workflow_node": lambda value: workflow_node.run({"query": "query", "top_k": value}),
            "pipeline_configuration": lambda value: AdvancedRAGPipeline(base, llm, top_k=value),
        }

        invalid_values = ((0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError))
        for case_name, invoke in cases.items():
            for value, expected_error in invalid_values:
                with self.subTest(case=case_name, value=value):
                    with self.assertRaises(expected_error):
                        invoke(value)

    def test_valid_top_k_is_forwarded_without_coercion(self) -> None:
        base = _RecordingRetriever()
        node = RetrieveNode(base)

        result = node.run({"query": "query", "top_k": 1})

        self.assertEqual(base.calls, [1])
        self.assertEqual([doc.doc_id for doc in result["documents"]], ["doc-1"])

    def test_legacy_corrective_limits_and_metadata_are_request_local(self) -> None:
        source = Document("alpha", metadata={"tenant": "a"}, doc_id="doc-1")
        base = _RecordingRetriever()
        base.retrieve = lambda query, top_k=5: [source]  # type: ignore[method-assign]

        retriever = CorrectiveRAGRetriever(base, _StaticLLM(), max_retries=0)
        result = retriever.retrieve("query", top_k=1)

        self.assertNotIn("crag_label", source.metadata)
        self.assertEqual(result[0].metadata["crag_label"], "ambiguous")
        for kwargs, expected in (
            ({"max_retries": True}, TypeError),
            ({"max_retries": -1}, ValueError),
            ({"min_correct": 0}, ValueError),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(expected):
                CorrectiveRAGRetriever(base, _StaticLLM(), **kwargs)


if __name__ == "__main__":
    unittest.main()
