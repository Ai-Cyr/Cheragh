from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import unittest

from cheragh.adaptive import (
    AdaptiveRAGEngine,
    AdaptiveRAGRoute,
    AdaptiveRoutingDecision,
    HeuristicComplexityClassifier,
    LLMComplexityClassifier,
    AdaptiveRetriever,
)
from cheragh.base import Document, LLMClient
from cheragh.schema import RAGResponse, Source


class _QueueLLM(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


class _FixedClassifier:
    def __init__(self, route: AdaptiveRAGRoute) -> None:
        self.route = route
        self.queries: list[str] = []

    def classify(self, query: str) -> AdaptiveRoutingDecision:
        self.queries.append(query)
        return AdaptiveRoutingDecision(self.route, confidence=0.91, rationale="test_classifier")


class _RecordingEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, int | None, dict[str, object]]] = []
        self.response = RAGResponse(
            query="provider-query",
            answer=f"answer-from-{name}",
            sources=[Source("doc", 0.8, "evidence")],
            retrieved_documents=[Document("evidence", doc_id="doc", score=0.8)],
            prompt="provider-prompt",
            metadata={"provider": {"name": name}},
        )

    def ask(self, query: str, top_k: int | None = None, **kwargs: object) -> RAGResponse:
        self.calls.append((query, top_k, kwargs))
        return self.response


@dataclass
class _WrappedResponse:
    response: RAGResponse


class _WrappedEngine(_RecordingEngine):
    def ask(self, query: str, top_k: int | None = None, **kwargs: object) -> _WrappedResponse:
        self.calls.append((query, top_k, kwargs))
        return _WrappedResponse(self.response)


class AdaptiveRAGTests(unittest.TestCase):
    def test_no_retrieval_route_calls_only_direct_generator(self) -> None:
        single = _RecordingEngine("single")
        iterative = _RecordingEngine("iterative")
        llm = _QueueLLM(["4"])
        engine = AdaptiveRAGEngine(
            single,
            iterative_engine=iterative,
            llm_client=llm,
            classifier=_FixedClassifier(AdaptiveRAGRoute.NO_RETRIEVAL),
        )

        response = engine.ask("  Combien font 2 + 2 ?  ", top_k=3)

        self.assertEqual(response.answer, "4")
        self.assertEqual(response.query, "Combien font 2 + 2 ?")
        self.assertEqual(response.sources, [])
        self.assertEqual(single.calls, [])
        self.assertEqual(iterative.calls, [])
        self.assertEqual(
            response.metadata["adaptive_rag"]["executed_route"],
            "no_retrieval",
        )
        self.assertIn("adaptive_rag_no_retrieval", response.warnings)

    def test_single_and_iterative_routes_use_distinct_engines(self) -> None:
        for route, expected_name in (
            (AdaptiveRAGRoute.SINGLE_STEP, "single"),
            (AdaptiveRAGRoute.ITERATIVE, "iterative"),
        ):
            with self.subTest(route=route):
                single = _RecordingEngine("single")
                iterative = _WrappedEngine("iterative")
                engine = AdaptiveRAGEngine(
                    single,
                    iterative_engine=iterative,
                    llm_client=_QueueLLM([]),
                    classifier=_FixedClassifier(route),
                )

                response = engine.ask("question documentée", top_k=7, temperature=0.0)

                expected = single if expected_name == "single" else iterative
                other = iterative if expected is single else single
                self.assertEqual(expected.calls, [("question documentée", 7, {"temperature": 0.0})])
                self.assertEqual(other.calls, [])
                self.assertEqual(response.answer, f"answer-from-{expected_name}")
                self.assertEqual(
                    response.metadata["adaptive_rag"]["executed_route"],
                    route.value,
                )
                # The provider-owned response is not mutated by route metadata.
                self.assertNotIn("adaptive_rag", expected.response.metadata)

    def test_missing_iterative_engine_has_explicit_bounded_fallback(self) -> None:
        single = _RecordingEngine("single")
        engine = AdaptiveRAGEngine(
            single,
            llm_client=_QueueLLM([]),
            classifier=_FixedClassifier(AdaptiveRAGRoute.ITERATIVE),
        )

        response = engine.ask("compare les deux politiques")

        route = response.metadata["adaptive_rag"]
        self.assertEqual(route["requested_route"], "iterative")
        self.assertEqual(route["executed_route"], "single_step")
        self.assertEqual(route["fallback_reason"], "iterative_engine_unavailable")
        self.assertIn("adaptive_rag_iterative_fallback", response.warnings)

        strict = AdaptiveRAGEngine(
            single,
            llm_client=_QueueLLM([]),
            classifier=_FixedClassifier(AdaptiveRAGRoute.ITERATIVE),
            fallback_to_single_step=False,
        )
        with self.assertRaises(RuntimeError):
            strict.ask("compare les deux politiques")

    def test_heuristic_classifier_covers_three_complexity_classes(self) -> None:
        classifier = HeuristicComplexityClassifier()
        cases = {
            "Bonjour !": AdaptiveRAGRoute.NO_RETRIEVAL,
            "Quel est le délai de remboursement ?": AdaptiveRAGRoute.SINGLE_STEP,
            "Compare les politiques et explique leurs impacts": AdaptiveRAGRoute.ITERATIVE,
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(classifier.classify(query).route, expected)

    def test_llm_classifier_falls_back_when_label_is_unrecognized(self) -> None:
        llm = _QueueLLM(["Je ne suis pas certain."])
        classifier = LLMComplexityClassifier(llm)

        decision = classifier.classify("Compare A versus B")

        self.assertEqual(decision.route, AdaptiveRAGRoute.ITERATIVE)
        self.assertTrue(decision.rationale.startswith("llm_output_unrecognized:"))

    def test_public_contract_validates_query_top_k_and_classifier_output(self) -> None:
        single = _RecordingEngine("single")
        engine = AdaptiveRAGEngine(
            single,
            llm_client=_QueueLLM([]),
            classifier=_FixedClassifier(AdaptiveRAGRoute.SINGLE_STEP),
        )
        for value, error in ((0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)):
            with self.subTest(top_k=value), self.assertRaises(error):
                engine.ask("question", top_k=value)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            engine.ask("   ")
        with self.assertRaises(TypeError):
            engine.ask(42)  # type: ignore[arg-type]

        class InvalidClassifier:
            def classify(self, query: str) -> str:
                return "single_step"

        invalid = AdaptiveRAGEngine(
            single,
            llm_client=_QueueLLM([]),
            classifier=InvalidClassifier(),  # type: ignore[arg-type]
        )
        with self.assertRaises(TypeError):
            invalid.ask("question")

        for confidence, error in (
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (True, TypeError),
        ):
            with self.subTest(confidence=confidence), self.assertRaises(error):
                AdaptiveRoutingDecision(
                    AdaptiveRAGRoute.SINGLE_STEP,
                    confidence=confidence,  # type: ignore[arg-type]
                )
        with self.assertRaises(TypeError):
            AdaptiveRoutingDecision(
                AdaptiveRAGRoute.SINGLE_STEP,
                rationale=1,  # type: ignore[arg-type]
            )

    def test_legacy_adaptive_retriever_enforces_top_k_and_result_boundary(self) -> None:
        class _OverReturningRetriever:
            def __init__(self) -> None:
                self.documents = [
                    Document("one", metadata={"nested": {"version": 1}}, doc_id="one", score=1.0),
                    Document("two", doc_id="two", score=0.5),
                ]

            def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
                del query, top_k
                return self.documents

        source = _OverReturningRetriever()
        adaptive = AdaptiveRetriever(source, _QueueLLM(["RETRIEVE", "RETRIEVE"]))

        first = adaptive.retrieve("  question  ", top_k=1)
        first[0].metadata["nested"]["version"] = 99
        second = adaptive.retrieve("question", top_k=1)

        self.assertEqual([document.doc_id for document in first], ["one"])
        self.assertEqual([document.doc_id for document in second], ["one"])
        self.assertEqual(second[0].metadata["nested"]["version"], 1)

        source.documents[0].score = math.nan
        invalid = AdaptiveRetriever(source, _QueueLLM(["RETRIEVE"]))
        with self.assertRaises(ValueError):
            invalid.retrieve("question", top_k=1)

    def test_async_api_preserves_routing_contract(self) -> None:
        single = _RecordingEngine("single")
        engine = AdaptiveRAGEngine(
            single,
            llm_client=_QueueLLM([]),
            classifier=_FixedClassifier(AdaptiveRAGRoute.SINGLE_STEP),
        )

        response = asyncio.run(engine.aask("question", top_k=2))

        self.assertEqual(response.answer, "answer-from-single")
        self.assertEqual(single.calls[0][:2], ("question", 2))


if __name__ == "__main__":
    unittest.main()
