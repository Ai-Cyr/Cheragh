import math
import unittest

from cheragh.base import BaseRetriever, Document, LLMClient, StaticLLMClient
from cheragh.corrective import (
    CorrectiveRAGEngine,
    CorrectiveRAGResult,
    LexicalKnowledgeRefiner,
    LexicalRetrievalGrader,
    RetrievalAction,
    RetrievalGrade,
)
from cheragh.engine import RAGEngine


class RecordingRetriever(BaseRetriever):
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        self.calls.append((query, top_k))
        return self.documents


class StatefulRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.calls = 0
        self.source = Document(
            "graded evidence",
            metadata={"nested": {"owner": "index"}},
            doc_id="graded",
            score=1.0,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        self.calls += 1
        if self.calls == 1:
            return [self.source]
        return [Document("incoherent second retrieval", doc_id="poison", score=1.0)]


class RecordingLLM(LLMClient):
    def __init__(self, answer: str):
        self.answer = answer
        self.prompts: list[str] = []
        self.kwargs: list[dict[str, object]] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        self.kwargs.append(dict(kwargs))
        return self.answer


class AlwaysCorrectGrader:
    def grade(self, query: str, documents: list[Document]) -> RetrievalGrade:
        return RetrievalGrade(
            score=1.0,
            passed=True,
            reason="test",
            document_count=len(documents),
            action=RetrievalAction.CORRECT,
        )


class MutatingGrader(AlwaysCorrectGrader):
    def grade(self, query: str, documents: list[Document]) -> RetrievalGrade:
        documents[0].content = "grader mutation"
        documents[0].metadata["nested"]["owner"] = "grader"
        return super().grade(query, documents)


class MappingActionGrader:
    def __init__(self, action: str):
        self.action = action

    def grade(self, query: str, documents: list[Document]) -> dict[str, object]:
        return {
            "score": {"incorrect": 0.0, "ambiguous": 0.5, "correct": 1.0}[self.action],
            "action": self.action,
            "reason": "mapping",
        }


class InvalidMappingGrader:
    def grade(self, query: str, documents: list[Document]) -> dict[str, object]:
        return {"score": True, "passed": "yes", "reason": 1}


class MutatingRefiner:
    def __init__(self):
        self.received: list[Document] = []

    def refine(self, query: str, documents: list[Document]) -> list[Document]:
        self.received = documents
        documents[0].metadata["nested"]["v"] = 999
        return [
            Document(
                "refined answer",
                metadata={"nested": {"v": 2}},
                doc_id=documents[0].doc_id,
                score=documents[0].score,
            )
        ]


class InvalidRefiner:
    def refine(self, query: str, documents: list[Document]):
        return ["not a document"]


class AdvancedCRAGTests(unittest.TestCase):
    def test_generation_uses_the_exact_graded_snapshot_without_second_retrieval(self) -> None:
        retriever = StatefulRetriever()
        llm = RecordingLLM("answer [source: graded]")
        base = RAGEngine(retriever, llm_client=llm, require_citations=True)
        engine = CorrectiveRAGEngine(base_engine=base, retrieval_grader=MutatingGrader(), max_retries=0)

        response = engine.ask("question", temperature=0.2)

        self.assertEqual(retriever.calls, 1)
        self.assertIn("graded evidence", llm.prompts[0])
        self.assertNotIn("grader mutation", llm.prompts[0])
        self.assertNotIn("incoherent second retrieval", llm.prompts[0])
        self.assertEqual(response.retrieved_documents[0].doc_id, "graded")
        self.assertEqual(response.retrieved_documents[0].metadata["nested"]["owner"], "index")
        self.assertEqual(retriever.source.metadata["nested"]["owner"], "index")
        self.assertEqual(llm.kwargs, [{"temperature": 0.2}])

    def test_correct_action_does_not_call_external_retriever(self) -> None:
        primary = RecordingRetriever([Document("Paris est la capitale de la France.", doc_id="local", score=0.0)])
        external = RecordingRetriever([Document("external", doc_id="web", score=0.0)])
        engine = CorrectiveRAGEngine(
            retriever=primary,
            llm_client=StaticLLMClient("Paris [source: local]"),
            retrieval_grader=AlwaysCorrectGrader(),
            external_retriever=external,
            max_retries=0,
        )

        response = engine.ask("Quelle est la capitale de la France ?", top_k=1)

        self.assertEqual(external.calls, [])
        self.assertEqual(response.metadata["retrieval_action"], "correct")
        self.assertFalse(response.metadata["external_retrieval_attempted"])
        self.assertEqual(response.retrieved_documents[0].metadata["corrective_provenance"]["origin"], "primary")

    def test_ambiguous_action_augments_with_external_evidence_and_logs_provenance(self) -> None:
        primary_doc = Document("La capitale est discutée.", metadata={"source": "local"}, doc_id="partial", score=0.0)
        external_doc = Document(
            "Paris est la capitale de la France.",
            metadata={"source": "web", "nested": {"rank": 1}},
            doc_id="web-1",
            score=0.0,
        )
        primary = RecordingRetriever([primary_doc])
        external = RecordingRetriever([external_doc, Document("extra", doc_id="web-2", score=0.0)])
        llm = RecordingLLM("Paris [source: web-1]")
        engine = CorrectiveRAGEngine(
            retriever=primary,
            llm_client=llm,
            retrieval_grader=LexicalRetrievalGrader(min_overlap=0.75, incorrect_overlap=0.0),
            external_retriever=external,
            external_top_k=1,
            max_retries=0,
            return_details=True,
        )

        result = engine.ask("capitale France", top_k=2)
        self.assertIsInstance(result, CorrectiveRAGResult)
        assert isinstance(result, CorrectiveRAGResult)

        self.assertEqual(external.calls, [("capitale France", 1)])
        self.assertEqual(result.metadata["retrieval_action"], "ambiguous")
        self.assertEqual(result.metadata["external_document_ids"], ["web-1"])
        self.assertEqual([document.doc_id for document in result.retrieved_documents], ["web-1", "partial"])
        provenance = result.retrieved_documents[0].metadata["corrective_provenance"]
        self.assertEqual(provenance["origin"], "external")
        self.assertEqual(provenance["trigger_action"], "ambiguous")
        self.assertEqual(provenance["retrieval_query"], "capitale France")
        self.assertEqual(result.attempts[0]["action"], "ambiguous")
        self.assertTrue(result.attempts[0]["post_correction_grade"]["passed"])
        self.assertIn("Paris est la capitale", llm.prompts[0])
        external_doc.metadata["nested"]["rank"] = 99
        self.assertEqual(result.retrieved_documents[0].metadata["nested"]["rank"], 1)

    def test_incorrect_action_discards_primary_and_uses_external_correction(self) -> None:
        primary = RecordingRetriever([Document("Météo et sport.", doc_id="wrong", score=0.0)])
        external = RecordingRetriever(
            [Document("Paris est la capitale de la France.", doc_id="web", score=0.0)]
        )
        llm = RecordingLLM("Paris [source: web]")
        engine = CorrectiveRAGEngine(
            retriever=primary,
            llm_client=llm,
            retrieval_grader=LexicalRetrievalGrader(min_overlap=0.75),
            external_retriever=external,
            max_retries=0,
        )

        response = engine.ask("capitale France", top_k=1)

        self.assertEqual(response.metadata["retrieval_action"], "incorrect")
        self.assertEqual([document.doc_id for document in response.retrieved_documents], ["web"])
        self.assertNotIn("Météo et sport", llm.prompts[0])
        self.assertIn("Paris est la capitale", llm.prompts[0])

    def test_lexical_refiner_decomposes_and_recomposes_relevant_sentences(self) -> None:
        source = Document(
            "La météo est pluvieuse. Paris est la capitale de la France. Le football se joue demain.",
            metadata={"source": "guide", "nested": {"v": 1}},
            doc_id="guide",
            score=0.0,
        )
        retriever = RecordingRetriever([source])
        llm = RecordingLLM("Paris [source: guide]")
        engine = CorrectiveRAGEngine(
            retriever=retriever,
            llm_client=llm,
            retrieval_grader=AlwaysCorrectGrader(),
            knowledge_refiner="lexical",
            max_retries=0,
        )

        response = engine.ask("capitale France", top_k=1)

        refined = response.retrieved_documents[0]
        self.assertEqual(refined.content, "Paris est la capitale de la France.")
        self.assertNotIn("météo", llm.prompts[0].lower())
        self.assertEqual(refined.metadata["nested"], {"v": 1})
        refinement = refined.metadata["corrective_provenance"]["refinement"]
        self.assertEqual(refinement["strategy"], "LexicalKnowledgeRefiner")
        self.assertEqual(refinement["retained_passages"], 1)
        self.assertEqual(source.content.split(".")[0], "La météo est pluvieuse")
        self.assertNotIn("corrective_provenance", source.metadata)

    def test_custom_refiner_receives_snapshots_and_output_inherits_origin_provenance(self) -> None:
        source = Document("raw answer", metadata={"nested": {"v": 1}}, doc_id="doc", score=1.0)
        refiner = MutatingRefiner()
        llm = RecordingLLM("refined [source: doc]")
        engine = CorrectiveRAGEngine(
            retriever=RecordingRetriever([source]),
            llm_client=llm,
            retrieval_grader=AlwaysCorrectGrader(),
            knowledge_refiner=refiner,
            max_retries=0,
        )

        response = engine.ask("answer", top_k=1)

        self.assertEqual(source.metadata["nested"], {"v": 1})
        self.assertEqual(response.retrieved_documents[0].content, "refined answer")
        provenance = response.retrieved_documents[0].metadata["corrective_provenance"]
        self.assertEqual(provenance["origin"], "primary")
        self.assertEqual(provenance["refinement"]["strategy"], "MutatingRefiner")
        self.assertTrue(response.metadata["knowledge_refined"])

    def test_mapping_grader_action_is_normalized_to_tri_state(self) -> None:
        retriever = RecordingRetriever([Document("partial", doc_id="p", score=0.0)])
        engine = CorrectiveRAGEngine(
            retriever=retriever,
            llm_client=StaticLLMClient("unused"),
            retrieval_grader=MappingActionGrader("ambiguous"),
            max_retries=0,
        )

        response = engine.ask("question")

        self.assertEqual(response.metadata["retrieval_action"], "ambiguous")
        self.assertEqual(response.metadata["retrieval_grade"]["action"], "ambiguous")
        self.assertIn("corrective_low_context", response.warnings)

    def test_advanced_failures_fall_back_without_generating(self) -> None:
        primary = RecordingRetriever([Document("wrong", doc_id="wrong", score=0.0)])
        external = RecordingRetriever([])
        llm = RecordingLLM("must not run")
        engine = CorrectiveRAGEngine(
            retriever=primary,
            llm_client=llm,
            retrieval_grader=MappingActionGrader("incorrect"),
            external_retriever=external,
            max_retries=0,
        )

        response = engine.ask("question", top_k=1)

        self.assertEqual(llm.prompts, [])
        self.assertEqual(external.calls, [("question", 1)])
        self.assertTrue(response.metadata["attempts"][0]["external_retrieval_attempted"])
        self.assertEqual(response.metadata["failed_stage"], "retrieval")

    def test_strict_threshold_and_component_validation(self) -> None:
        retriever = RecordingRetriever([])
        for kwargs, exception in (
            ({"min_context_score": True}, TypeError),
            ({"min_context_score": -0.1}, ValueError),
            ({"min_context_score": 1.1}, ValueError),
            ({"incorrect_context_score": math.nan}, ValueError),
            ({"min_context_score": 0.2, "incorrect_context_score": 0.3}, ValueError),
            ({"min_grounded_score": "0.5"}, TypeError),
            ({"external_top_k": True}, TypeError),
            ({"external_top_k": 0}, ValueError),
            ({"return_details": 1}, TypeError),
            ({"fallback_answer": 1}, TypeError),
            ({"external_retriever": object()}, TypeError),
            ({"retrieval_grader": object()}, TypeError),
            ({"query_rewriter": object()}, TypeError),
            ({"knowledge_refiner": "unknown"}, ValueError),
            ({"knowledge_refiner": object()}, TypeError),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(exception):
                    CorrectiveRAGEngine(retriever=retriever, **kwargs)  # type: ignore[arg-type]

        for invalid, exception in ((True, TypeError), (0, ValueError), (-1, ValueError)):
            with self.subTest(min_term_overlap=invalid):
                with self.assertRaises(exception):
                    LexicalKnowledgeRefiner(min_term_overlap=invalid)  # type: ignore[arg-type]

    def test_invalid_collaborator_outputs_are_rejected_before_generation(self) -> None:
        source = RecordingRetriever([Document("answer", doc_id="doc", score=1.0)])
        with self.assertRaises(TypeError):
            CorrectiveRAGEngine(
                retriever=source,
                retrieval_grader=AlwaysCorrectGrader(),
                knowledge_refiner=InvalidRefiner(),
                max_retries=0,
            ).ask("answer")

        invalid_score = CorrectiveRAGEngine(
            retriever=source,
            retrieval_grader=lambda query, documents: math.inf,
            max_retries=0,
        )
        with self.assertRaises(ValueError):
            invalid_score.ask("answer")

        invalid_mapping = CorrectiveRAGEngine(
            retriever=source,
            retrieval_grader=InvalidMappingGrader(),
            max_retries=0,
        )
        with self.assertRaises(TypeError):
            invalid_mapping.ask("answer")

        bad_documents = RecordingRetriever(["not a document"])  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            CorrectiveRAGEngine(
                retriever=bad_documents,
                retrieval_grader=AlwaysCorrectGrader(),
                max_retries=0,
            ).ask("answer")

    def test_primary_top_k_is_enforced_when_provider_over_returns(self) -> None:
        source = Document("valid evidence", doc_id="valid", score=1.0)

        class OverReturningRetriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 5):
                yield source
                yield "invalid but beyond the strict cap"

        response = CorrectiveRAGEngine(
            retriever=OverReturningRetriever(),
            llm_client=StaticLLMClient("answer [source: valid]"),
            retrieval_grader=AlwaysCorrectGrader(),
            max_retries=0,
        ).ask("question", top_k=1)

        self.assertEqual([document.doc_id for document in response.retrieved_documents], ["valid"])

    def test_anonymous_evidence_receives_a_stable_citable_snapshot_id(self) -> None:
        source = Document("anonymous evidence", doc_id=None, score=1.0)
        llm = RecordingLLM("answer")
        response = CorrectiveRAGEngine(
            retriever=RecordingRetriever([source]),
            llm_client=llm,
            retrieval_grader=AlwaysCorrectGrader(),
            max_retries=0,
        ).ask("question", top_k=1)

        document = response.retrieved_documents[0]
        self.assertTrue(document.doc_id.startswith("crag-anonymous-"))
        self.assertIn(f"[source: {document.doc_id}]", llm.prompts[0])
        self.assertIsNone(source.doc_id)
        provenance = document.metadata["corrective_provenance"]
        self.assertTrue(provenance["synthetic_doc_id"])
        self.assertIsNone(provenance["original_doc_id"])

    def test_infinite_duplicate_query_rewriter_is_consumed_with_a_strict_bound(self) -> None:
        class InfiniteDuplicateRewriter:
            def __init__(self) -> None:
                self.yields = 0

            def transform(self, query: str):
                while True:
                    self.yields += 1
                    yield query

        rewriter = InfiniteDuplicateRewriter()
        response = CorrectiveRAGEngine(
            retriever=RecordingRetriever([Document("evidence", doc_id="doc", score=1.0)]),
            llm_client=StaticLLMClient("answer [source: doc]"),
            retrieval_grader=AlwaysCorrectGrader(),
            query_rewriter=rewriter,
            max_retries=1,
        ).ask("question")

        self.assertEqual(response.metadata["selected_query"], "question")
        self.assertEqual(rewriter.yields, 8)

    def test_malformed_documents_and_empty_queries_fail_before_generation(self) -> None:
        malformed = (
            Document("", doc_id="empty"),
            Document("content", metadata=[], doc_id="metadata"),  # type: ignore[arg-type]
            Document("content", doc_id=3),  # type: ignore[arg-type]
            Document("content", doc_id="score", score=float("nan")),
            Document("content", doc_id="bool-score", score=True),
        )
        for document in malformed:
            with self.subTest(document=document):
                engine = CorrectiveRAGEngine(
                    retriever=RecordingRetriever([document]),
                    retrieval_grader=AlwaysCorrectGrader(),
                    max_retries=0,
                )
                with self.assertRaises((TypeError, ValueError)):
                    engine.ask("question")

        retriever = RecordingRetriever([Document("evidence", doc_id="doc")])
        engine = CorrectiveRAGEngine(
            retriever=retriever,
            retrieval_grader=AlwaysCorrectGrader(),
            max_retries=0,
        )
        for invalid, exception in ((None, TypeError), ("   ", ValueError)):
            with self.subTest(query=invalid):
                with self.assertRaises(exception):
                    engine.ask(invalid)  # type: ignore[arg-type]
        self.assertEqual(retriever.calls, [])


if __name__ == "__main__":
    unittest.main()
