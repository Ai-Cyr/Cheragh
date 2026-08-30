import ast
import inspect
import unittest
from typing import TYPE_CHECKING, Sequence, get_type_hints

import cheragh
from cheragh import (
    BaseRetriever,
    Document,
    HashingEmbedding,
    HybridSearchRetriever,
    KeywordOverlapReranker,
    LexicalEvidenceCritic,
    MemoryVectorStore,
    RAGEngine,
    RerankerProtocol,
    RetrievalToolAdapter,
    SelfRAGEngine,
    Source,
    StaticLLMClient,
    ToolRegistry,
)

if TYPE_CHECKING:
    from cheragh import AgenticRAGEngine, FederatedRAGEngine, RAGResponse

    _typed_engine_class: type[RAGEngine] = RAGEngine
    _typed_response: RAGResponse
    _typed_agentic_engine: AgenticRAGEngine
    _typed_federated_engine: FederatedRAGEngine
    _typed_reranker: RerankerProtocol = KeywordOverlapReranker()


class EmptyRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return []


class RecordingRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.top_k_calls: list[int] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        self.top_k_calls.append(top_k)
        return [Document("context", doc_id="d1", score=1.0)]


class APICoreConsistencyTests(unittest.TestCase):
    def test_in_memory_indexes_snapshot_documents_and_nested_metadata(self):
        document = Document("alpha", metadata={"nested": {"version": 1}}, doc_id="a")
        documents = [document]
        hybrid = HybridSearchRetriever(documents, HashingEmbedding(dimension=32))
        memory = MemoryVectorStore(HashingEmbedding(dimension=32))
        memory.add_documents(documents)

        document.content = "mutated"
        document.metadata["nested"]["version"] = 2
        documents.append(Document("late addition", doc_id="late"))

        self.assertEqual(len(hybrid.documents), 1)
        self.assertEqual(hybrid.documents[0].content, "alpha")
        self.assertEqual(hybrid.documents[0].metadata["nested"]["version"], 1)
        self.assertEqual(hybrid.retrieve("alpha", top_k=1)[0].content, "alpha")
        self.assertEqual(len(memory.documents), 1)
        self.assertEqual(memory.documents[0].content, "alpha")
        self.assertEqual(memory.documents[0].metadata["nested"]["version"], 1)
        self.assertEqual(memory.similarity_search("alpha", top_k=1)[0].content, "alpha")

    def test_strict_no_context_response_matches_stream_response(self):
        llm = StaticLLMClient("must not be generated")
        engine = RAGEngine(
            EmptyRetriever(),
            llm_client=llm,
            strict_grounding=True,
            require_citations=True,
            flag_unsourced_sentences=True,
        )

        direct = engine.ask("question")
        stream = engine.stream_with_response("question")
        self.assertEqual("".join(stream), direct.answer)
        streamed = stream.response
        self.assertIsNotNone(streamed)
        assert streamed is not None

        direct_payload = direct.to_dict()
        streamed_payload = streamed.to_dict()
        direct_payload.pop("trace")
        streamed_payload.pop("trace")
        direct_payload.pop("response_id")
        streamed_payload.pop("response_id")
        self.assertEqual(streamed_payload, direct_payload)
        self.assertEqual(direct.grounded_score, 0.0)
        self.assertEqual(direct.metadata["top_k"], engine.top_k)
        self.assertEqual(direct.citation_validation.grounded_score, direct.grounded_score)
        self.assertEqual(direct.trace.metadata["answer_chars"], len(direct.answer))
        self.assertEqual(streamed.trace.metadata["answer_chars"], len(streamed.answer))
        self.assertIn("no_relevant_documents", direct.trace.warnings)
        self.assertIn("no_relevant_documents", streamed.trace.warnings)
        self.assertEqual(llm.prompts, [])

    def test_rag_engine_enforces_positive_non_bool_integer_top_k(self):
        for invalid in (True, False, 1.5, "2"):
            with self.subTest(constructor_type=invalid):
                with self.assertRaises(TypeError):
                    RAGEngine(EmptyRetriever(), top_k=invalid)
        for invalid in (0, -1):
            with self.subTest(constructor_value=invalid):
                with self.assertRaises(ValueError):
                    RAGEngine(EmptyRetriever(), top_k=invalid)

        retriever = RecordingRetriever()
        engine = RAGEngine(retriever, llm_client=StaticLLMClient("answer"), top_k=3)
        engine.ask("question")
        self.assertEqual(retriever.top_k_calls[-1], 3)
        for invalid, exception in ((True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)):
            with self.subTest(ask=invalid):
                with self.assertRaises(exception):
                    engine.ask("question", top_k=invalid)
            with self.subTest(stream_with_response=invalid):
                with self.assertRaises(exception):
                    engine.stream_with_response("question", top_k=invalid)
            with self.subTest(stream=invalid):
                with self.assertRaises(exception):
                    list(engine.stream("question", top_k=invalid))

    def test_source_factory_preserves_location_and_copies_metadata(self):
        document = Document(
            "source text",
            metadata={
                "source": "guide.md",
                "page": 3,
                "source_char_start": 10,
                "source_char_end": 21,
                "nested": {"version": 1},
            },
            doc_id="guide",
            score=0.8,
        )
        source = Source.from_document(document)
        document.metadata["page"] = 9
        document.metadata["nested"]["version"] = 2

        self.assertEqual(source.preview, "source text")
        self.assertEqual(source.metadata["page"], 3)
        self.assertEqual(source.metadata["nested"]["version"], 1)
        self.assertEqual(source.location, "source=guide.md; page=3; chars=10-21")

    def test_lazy_root_exports_are_visible_to_dir(self):
        self.assertTrue(set(cheragh.__all__).issubset(dir(cheragh)))

    def test_all_lazy_root_exports_have_static_type_checking_imports(self):
        tree = ast.parse(inspect.getsource(cheragh))
        typed_exports: set[str] = set()
        for node in tree.body:
            if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"):
                continue
            for statement in node.body:
                if isinstance(statement, ast.ImportFrom):
                    typed_exports.update(alias.asname or alias.name for alias in statement.names)

        self.assertEqual(typed_exports, set(cheragh._LAZY_EXPORTS))

    def test_reranker_protocol_signature_matches_builtin_contract(self):
        hints = get_type_hints(RerankerProtocol.rerank)
        signature = inspect.signature(RerankerProtocol.rerank)

        self.assertEqual(hints["documents"], Sequence[Document])
        self.assertEqual(signature.parameters["top_k"].default, 5)
        self.assertIsInstance(KeywordOverlapReranker(), RerankerProtocol)

    def test_release_examples_use_the_supported_adapter_shapes(self):
        critic = LexicalEvidenceCritic()
        engine = SelfRAGEngine(None, evidence_critic=critic, retrieval_gate=None)
        self.assertIs(engine.evidence_critic, critic)

        registry = ToolRegistry()
        registry.register(RetrievalToolAdapter(EmptyRetriever()).as_tool())
        self.assertEqual([spec.name for spec in registry.specs], ["retrieve"])


if __name__ == "__main__":
    unittest.main()
