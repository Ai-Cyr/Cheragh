import unittest
from types import SimpleNamespace

from cheragh import Document, RAGEngine, StaticLLMClient
from cheragh.corrective import CorrectiveRAGEngine, RetrievalGrade
from cheragh.federated import FederatedRAGEngine, FederatedRetriever
from cheragh.multihop import MultiHopRAGEngine
from cheragh.schema import RAGResponse, Source
from cheragh.structured import SQLRAGEngine, StructuredRAG
from cheragh.tenancy import MultiTenantRAGEngine
from cheragh.workflow import RAGWorkflow


class RecordingLLM(StaticLLMClient):
    def __init__(self, response: str = "ok"):
        super().__init__(response)
        self.kwargs: list[dict[str, object]] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.kwargs.append(dict(kwargs))
        return super().generate(prompt, **kwargs)


class StaticRetriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.queries: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        self.queries.append((query, top_k))
        return self.documents[:top_k]


class QuerySensitiveGrader:
    def grade(self, query: str, documents: list[Document]) -> RetrievalGrade:
        passed = query == "requete corrigee"
        return RetrievalGrade(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="test",
            document_count=len(documents),
        )


class AskWithoutTopK:
    def __init__(self):
        self.calls: list[str] = []

    def ask(self, query: str):
        self.calls.append(query)
        inner = SimpleNamespace(
            answer="sans top_k",
            documents=[
                Document(
                    "preuve sans top_k",
                    metadata={"source": "memoire.db", "page": 2},
                    doc_id="shared",
                    score=0.9,
                )
            ],
        )
        return SimpleNamespace(response=inner)


class AskWithTopK:
    def __init__(self):
        self.top_k: int | None = None

    def ask(self, query: str, top_k: int | None = None):
        self.top_k = top_k
        return SimpleNamespace(
            answer="avec top_k",
            documents=[Document("preuve avec top_k", doc_id="shared", score=0.8)],
        )


class AskWithGenerationKwargsOnly:
    def __init__(self):
        self.kwargs: dict[str, object] | None = None

    def ask(self, query: str, **generate_kwargs: object):
        self.kwargs = dict(generate_kwargs)
        return SimpleNamespace(answer="sans contrat top_k", documents=[])


class AskRaisingTypeError:
    def __init__(self):
        self.calls = 0

    def ask(self, query: str, **generate_kwargs: object):
        self.calls += 1
        raise TypeError("provider failure")


class StructuredAPIConsistencyTests(unittest.TestCase):
    def test_uses_canonical_schema_query_alias_kwargs_and_complete_trace(self):
        llm = RecordingLLM("total")
        engine = SQLRAGEngine.from_records(
            "sales",
            [{"amount": 10}, {"amount": 20}],
            llm_client=llm,
        )

        response = engine.ask(question="Quel est le total ?", temperature=0.25)

        self.assertIsInstance(response, RAGResponse)
        self.assertIsInstance(response.sources[0], Source)
        self.assertEqual(response.query, "Quel est le total ?")
        self.assertEqual(llm.kwargs, [{"temperature": 0.25}])
        self.assertIn("Réponds en français", response.prompt)
        self.assertNotIn("Tu génères uniquement une requête SQL", response.prompt)
        self.assertEqual(response.trace.query, response.query)
        self.assertIsNotNone(response.trace.ended_at_unix)
        self.assertTrue(response.trace.retrieval)
        self.assertTrue(response.trace.token_usage)

        facade = StructuredRAG(engine)
        self.assertEqual(facade.ask(query="Combien de lignes ?").query, "Combien de lignes ?")
        with self.assertRaises(ValueError):
            engine.ask("une requete", question="une autre")
        for invalid, exception in ((True, TypeError), (0, ValueError), (-1, ValueError)):
            with self.subTest(top_k=invalid), self.assertRaises(exception):
                engine.retrieve("question", top_k=invalid)

    def test_max_rows_rejects_bool_non_integer_and_non_positive_values(self):
        for invalid in (True, False, 1.5, "5"):
            with self.subTest(invalid_type=invalid), self.assertRaises(TypeError):
                SQLRAGEngine(max_rows=invalid)
        for invalid in (0, -1):
            with self.subTest(invalid_value=invalid), self.assertRaises(ValueError):
                SQLRAGEngine(max_rows=invalid)


class MultiHopAPIConsistencyTests(unittest.TestCase):
    def test_constructor_rejects_non_integer_limits(self):
        retriever = StaticRetriever([])
        for field, invalid, exception in (
            ("max_steps", True, TypeError),
            ("max_steps", 1.5, TypeError),
            ("top_k_per_step", 0, ValueError),
            ("final_top_k", -1, ValueError),
        ):
            with self.subTest(field=field, invalid=invalid), self.assertRaises(exception):
                MultiHopRAGEngine(retriever, **{field: invalid})

    def test_retrieve_never_generates_and_ask_finishes_trace(self):
        retriever = StaticRetriever([Document("preuve", doc_id="doc-1", score=1.0)])
        llm = RecordingLLM("reponse [source: doc-1]")
        engine = MultiHopRAGEngine(retriever, llm_client=llm, max_steps=1)

        documents = engine.retrieve("question", top_k=1)

        self.assertEqual([document.doc_id for document in documents], ["doc-1"])
        self.assertEqual(llm.kwargs, [])

        result = engine.ask("question", temperature=0.1)
        self.assertEqual(llm.kwargs, [{"temperature": 0.1}])
        self.assertEqual(result.response.trace.query, "question")
        self.assertIsNotNone(result.response.trace.ended_at_unix)
        self.assertTrue(result.response.trace.retrieval)
        self.assertTrue(result.response.trace.token_usage)

        for invalid, exception in ((True, TypeError), (0, ValueError), (-1, ValueError)):
            with self.subTest(top_k=invalid), self.assertRaises(exception):
                engine.retrieve("question", top_k=invalid)


class CorrectiveAPIConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.retriever = StaticRetriever([Document("preuve corrigee", doc_id="doc-1", score=1.0)])
        self.llm = StaticLLMClient("reponse [source: doc-1]")
        self.base = RAGEngine(self.retriever, llm_client=self.llm)

    def test_preserves_original_query_after_internal_rewrite(self):
        engine = CorrectiveRAGEngine(
            base_engine=self.base,
            retrieval_grader=QuerySensitiveGrader(),
            query_rewriter=lambda query: ["requete corrigee"],
            max_retries=1,
        )

        response = engine.ask("requete originale")

        self.assertEqual(response.query, "requete originale")
        self.assertEqual(response.trace.query, "requete originale")
        self.assertEqual(response.trace.metadata["selected_query"], "requete corrigee")
        self.assertEqual(response.metadata["original_query"], "requete originale")
        self.assertEqual(response.metadata["selected_query"], "requete corrigee")

    def test_rejects_conflicting_explicit_dependencies(self):
        other_retriever = StaticRetriever([])
        with self.assertRaisesRegex(ValueError, "retriever conflicts"):
            CorrectiveRAGEngine(base_engine=self.base, retriever=other_retriever)
        with self.assertRaisesRegex(ValueError, "llm_client conflicts"):
            CorrectiveRAGEngine(base_engine=self.base, llm_client=StaticLLMClient("other"))

        compatible = CorrectiveRAGEngine(
            base_engine=self.base,
            retriever=self.retriever,
            llm_client=self.llm,
        )
        self.assertIs(compatible.retriever, self.retriever)

        for invalid, exception in ((True, TypeError), (1.5, TypeError), (-1, ValueError)):
            with self.subTest(max_retries=invalid), self.assertRaises(exception):
                CorrectiveRAGEngine(base_engine=self.base, max_retries=invalid)


class FederatedAPIConsistencyTests(unittest.TestCase):
    def test_constructor_rejects_unused_engine_options(self):
        with self.assertRaisesRegex(TypeError, "unexpected keyword"):
            FederatedRAGEngine({"source": AskWithoutTopK()}, strict_groundng=True)

    def test_adapts_ask_signature_wrappers_and_qualifies_document_ids(self):
        without_top_k = AskWithoutTopK()
        with_top_k = AskWithTopK()
        retriever = FederatedRetriever(
            {"plain": without_top_k, "configurable": with_top_k},
            top_k_per_source=3,
        )

        documents = retriever.retrieve("question", top_k=4)

        self.assertEqual(without_top_k.calls, ["question"])
        self.assertEqual(with_top_k.top_k, 3)
        self.assertEqual(
            {document.doc_id for document in documents},
            {"plain::shared", "configurable::shared"},
        )
        self.assertTrue(all(document.metadata["original_doc_id"] == "shared" for document in documents))

    def test_does_not_treat_generation_kwargs_as_a_top_k_contract(self):
        source = AskWithGenerationKwargsOnly()
        retriever = FederatedRetriever({"source": source})

        retriever.retrieve("question", top_k=2)

        self.assertEqual(source.kwargs, {})

    def test_ask_returns_locations_and_complete_trace(self):
        source = AskWithoutTopK()
        engine = FederatedRAGEngine(
            {"plain": source},
            llm_client=StaticLLMClient("reponse [source: plain::shared]"),
            top_k_per_source=2,
        )

        result = engine.ask("question")

        self.assertEqual(result.sources[0].doc_id, "plain::shared")
        self.assertEqual(result.sources[0].location, "source=memoire.db; page=2")
        self.assertEqual(result.response.trace.query, "question")
        self.assertIsNotNone(result.response.trace.ended_at_unix)
        self.assertTrue(result.response.trace.retrieval)
        self.assertTrue(result.response.trace.token_usage)

        for invalid, exception in ((True, TypeError), (0, ValueError), (-1, ValueError)):
            with self.subTest(top_k=invalid), self.assertRaises(exception):
                engine.retrieve("question", top_k=invalid)


class MultiTenantAPIConsistencyTests(unittest.TestCase):
    def test_generic_generation_kwargs_do_not_receive_principal(self):
        target = AskWithGenerationKwargsOnly()
        engine = MultiTenantRAGEngine(enforce_access_control=False)
        engine.add_collection("tenant-a", "collection-a", target, default=True)

        engine.ask("question", tenant_id="tenant-a", temperature=0.2)

        self.assertEqual(target.kwargs, {"temperature": 0.2})

    def test_internal_type_error_is_not_retried(self):
        target = AskRaisingTypeError()
        engine = MultiTenantRAGEngine(enforce_access_control=False)
        engine.add_collection("tenant-a", "collection-a", target, default=True)

        with self.assertRaisesRegex(TypeError, "provider failure"):
            engine.ask("question", tenant_id="tenant-a")

        self.assertEqual(target.calls, 1)


class WorkflowAPIConsistencyTests(unittest.TestCase):
    def test_keyword_state_dispatch_is_selected_before_execution(self):
        workflow = RAGWorkflow()
        workflow.add_node("keyword", lambda *, query: {"answer": query.upper()})

        result = workflow.ask("question")

        self.assertEqual(result.answer, "QUESTION")

    def test_internal_type_error_is_not_retried_with_other_call_style(self):
        calls = []

        def failing(state):
            calls.append(dict(state))
            raise TypeError("node failure")

        workflow = RAGWorkflow().add_node("failing", failing)

        with self.assertRaisesRegex(TypeError, "node failure"):
            workflow.ask("question")

        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
