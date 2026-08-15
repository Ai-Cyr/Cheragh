from __future__ import annotations

import unittest

from cheragh.base import BaseRetriever, Document, LLMClient
from cheragh.multihop import (
    LLMMultiHopPlanner,
    MultiHopRAGEngine,
    PlanningAction,
    PlanningContext,
    PlanningDecision,
    RuleBasedMultiHopPlanner,
)


class _RoutingRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.alpha = Document(
            "Alpha utilise le Produit Orion.",
            metadata={"nested": {"owner": "retriever"}},
            doc_id="alpha",
            score=0.7,
        )
        self.orion = Document(
            "Le Produit Orion dépend du fournisseur Delta.",
            doc_id="orion",
            score=0.9,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        self.calls.append((query, top_k))
        if "Delta" in query or "Orion" in query:
            return [self.orion]
        return [self.alpha]


class _RecordingLLM(LLMClient):
    def __init__(self, answer: str = "Réponse") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return self.answer


class _ObservationPlanner:
    def __init__(self) -> None:
        self.contexts: list[PlanningContext] = []

    def plan(self, context: PlanningContext) -> PlanningDecision:
        self.contexts.append(context)
        if not context.hops:
            return PlanningDecision.next("rechercher Alpha", rationale="seed")
        if len(context.hops) == 1:
            observed = context.hops[0].documents[0].content
            context.hops[0].documents[0].metadata["nested"]["owner"] = "planner"
            self.assert_observed(observed)
            return PlanningDecision.next("Produit Orion fournisseur Delta", rationale="bridge")
        return PlanningDecision.stop(rationale="evidence_sufficient")

    @staticmethod
    def assert_observed(observed: str) -> None:
        if "Produit Orion" not in observed:
            raise AssertionError("planner did not receive the previous observation")


class _AlwaysNextPlanner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, context: PlanningContext) -> PlanningDecision:
        self.calls += 1
        return PlanningDecision.next(f"hop {context.next_step}", rationale="continue")


class _DuplicateQueryPlanner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, context: PlanningContext) -> PlanningDecision:
        self.calls += 1
        return PlanningDecision.next("same query", rationale="repeat")


class _FixedDecomposer:
    def decompose(self, query: str, max_steps: int = 4) -> list[str]:
        return ["première requête", "deuxième requête", "troisième requête"]


class _QueueLLM(LLMClient):
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return self.outputs.pop(0)


class _InvalidFallbackPlanner:
    def plan(self, context: PlanningContext):
        return {"action": "next", "query": "not a decision"}


class MultiHopPlanningTests(unittest.TestCase):
    def test_dynamic_planner_observes_previous_evidence_and_stops_explicitly(self) -> None:
        retriever = _RoutingRetriever()
        planner = _ObservationPlanner()
        answer_llm = _RecordingLLM("Synthèse [source: orion]")
        engine = MultiHopRAGEngine(
            retriever,
            llm_client=answer_llm,
            planner=planner,
            max_steps=4,
            top_k_per_step=2,
        )

        result = engine.ask("Quel fournisseur dépend du produit utilisé par Alpha ?")

        self.assertEqual([query for query, _ in retriever.calls], [
            "rechercher Alpha",
            "Produit Orion fournisseur Delta",
        ])
        self.assertEqual(len(planner.contexts), 3)
        self.assertEqual([decision.action for decision in result.planning_decisions], [
            PlanningAction.NEXT,
            PlanningAction.NEXT,
            PlanningAction.STOP,
        ])
        self.assertEqual(result.response.metadata["planning_mode"], "dynamic")
        self.assertEqual(result.response.metadata["stop_reason"], "evidence_sufficient")
        self.assertEqual(result.hops[0].observation, "documents=1; evidence_ids=alpha")
        self.assertEqual(result.hops[0].documents[0].metadata["nested"]["owner"], "retriever")
        self.assertEqual(len(answer_llm.prompts), 1)

    def test_strict_max_steps_caps_planner_and_retrieval(self) -> None:
        retriever = _RoutingRetriever()
        planner = _AlwaysNextPlanner()
        engine = MultiHopRAGEngine(
            retriever,
            llm_client=_RecordingLLM(),
            planner=planner,
            max_steps=2,
        )

        result = engine.ask("question complexe")

        self.assertEqual(planner.calls, 2)
        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(len(result.hops), 2)
        self.assertEqual(result.planning_decisions[-1].action, PlanningAction.STOP)
        self.assertEqual(result.planning_decisions[-1].rationale, "max_steps_reached")

    def test_retrieve_never_calls_answer_synthesis_and_returns_snapshots(self) -> None:
        retriever = _RoutingRetriever()
        planner = _ObservationPlanner()
        answer_llm = _RecordingLLM()
        engine = MultiHopRAGEngine(
            retriever,
            llm_client=answer_llm,
            planner=planner,
            max_steps=3,
        )

        documents = engine.retrieve("question", top_k=2)

        self.assertEqual(answer_llm.prompts, [])
        self.assertEqual([document.doc_id for document in documents], ["orion", "alpha"])
        retriever.alpha.content = "changed later"
        retriever.alpha.metadata["nested"]["owner"] = "caller"
        self.assertEqual(documents[1].content, "Alpha utilise le Produit Orion.")
        self.assertEqual(documents[1].metadata["nested"]["owner"], "retriever")

    def test_duplicate_evidence_keeps_best_score_and_complete_chain(self) -> None:
        class DuplicateRetriever(BaseRetriever):
            def __init__(self) -> None:
                self.calls = 0

            def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
                self.calls += 1
                return [Document("preuve", doc_id="same", score=0.2 if self.calls == 1 else 0.8)]

        retriever = DuplicateRetriever()
        result = MultiHopRAGEngine(
            retriever,
            llm_client=_RecordingLLM(),
            planner=_AlwaysNextPlanner(),
            max_steps=2,
        ).ask("question")

        self.assertEqual(len(result.retrieved_documents), 1)
        merged = result.retrieved_documents[0]
        self.assertEqual(merged.score, 0.8)
        provenance = merged.metadata["multi_hop_provenance"]
        self.assertEqual(provenance["seen_steps"], [1, 2])
        self.assertEqual(provenance["retrieval_queries"], ["hop 1", "hop 2"])
        self.assertEqual(len(provenance["occurrences"]), 2)
        result.hops[0].documents[0].metadata["multi_hop_provenance"]["seen_steps"].append(99)
        self.assertEqual(provenance["seen_steps"], [1, 2])

    def test_rule_based_planner_conditions_follow_up_on_observation(self) -> None:
        retriever = _RoutingRetriever()
        engine = MultiHopRAGEngine(
            retriever,
            planner=RuleBasedMultiHopPlanner(bridge_terms=2),
            max_steps=3,
        )

        engine.retrieve("Compare les risques Alpha et ceux du fournisseur", top_k=2)

        self.assertGreaterEqual(len(retriever.calls), 2)
        first_query, second_query = retriever.calls[0][0], retriever.calls[1][0]
        self.assertEqual(first_query, "Compare les risques Alpha et ceux du fournisseur")
        self.assertNotEqual(second_query, "ceux du fournisseur")
        self.assertTrue(any(term in second_query for term in ("produit", "orion", "utilise")))

    def test_static_decomposition_path_remains_available_and_budgeted(self) -> None:
        retriever = _RoutingRetriever()
        engine = MultiHopRAGEngine(
            retriever,
            decomposer=_FixedDecomposer(),
            max_steps=2,
        )

        result = engine.ask("question")

        self.assertEqual(result.response.metadata["planning_mode"], "static")
        self.assertEqual(result.decomposed_queries, ["première requête", "deuxième requête"])
        self.assertEqual(len(retriever.calls), 2)
        self.assertIn("Contexte découvert précédemment", retriever.calls[1][0])
        self.assertEqual(result.planning_decisions[-1].rationale, "max_steps_reached")

    def test_llm_planner_uses_json_contract_and_deterministic_fallback(self) -> None:
        planner_llm = _QueueLLM([
            '{"action":"next","query":"chercher Orion","rationale":"premier pont"}',
            '{"action":"stop","rationale":"preuves suffisantes"}',
        ])
        planner = LLMMultiHopPlanner(planner_llm)
        first = planner.plan(PlanningContext("question", next_step=1, max_steps=2))
        hop_engine = MultiHopRAGEngine(_RoutingRetriever(), planner=_AlwaysNextPlanner(), max_steps=1)
        hop_result = hop_engine.ask("question")
        second = planner.plan(
            PlanningContext(
                "question",
                next_step=2,
                max_steps=2,
                hops=(hop_result.hops[0],),
                evidence=tuple(hop_result.retrieved_documents),
            )
        )

        self.assertEqual(first.query, "chercher Orion")
        self.assertEqual(second.action, PlanningAction.STOP)
        self.assertIn("Observations", planner_llm.prompts[1])
        self.assertIn("evidence_ids", planner_llm.prompts[1])

        invalid_llm = _QueueLLM(["not-json"])
        fallback = LLMMultiHopPlanner(invalid_llm, fallback=RuleBasedMultiHopPlanner())
        decision = fallback.plan(PlanningContext("question", next_step=1, max_steps=1))
        self.assertEqual(decision.action, PlanningAction.NEXT)
        self.assertTrue(decision.rationale.startswith("llm_output_invalid:"))

    def test_decision_dependencies_and_public_limits_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            PlanningDecision(PlanningAction.NEXT)
        with self.assertRaises(ValueError):
            PlanningDecision(PlanningAction.STOP, query="not allowed")
        with self.assertRaises(ValueError):
            MultiHopRAGEngine(
                _RoutingRetriever(),
                planner=_AlwaysNextPlanner(),
                decomposer=_FixedDecomposer(),
            )

        retriever = _RoutingRetriever()
        planner = _AlwaysNextPlanner()
        engine = MultiHopRAGEngine(retriever, planner=planner)
        with self.assertRaises(ValueError):
            engine.retrieve("question", top_k=0)
        self.assertEqual(planner.calls, 0)
        self.assertEqual(retriever.calls, [])

    def test_retriever_top_k_is_enforced_even_when_provider_ignores_it(self) -> None:
        class OverReturningRetriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
                return [
                    Document("first", doc_id="first", score=1.0),
                    Document("second", doc_id="second", score=0.9),
                    Document("third", doc_id="third", score=0.8),
                ]

        result = MultiHopRAGEngine(
            OverReturningRetriever(),
            planner=_AlwaysNextPlanner(),
            max_steps=1,
            top_k_per_step=1,
        ).retrieve("question", top_k=3)

        self.assertEqual([document.doc_id for document in result], ["first"])

    def test_duplicate_planned_query_stops_before_duplicate_retrieval(self) -> None:
        retriever = _RoutingRetriever()
        planner = _DuplicateQueryPlanner()
        result = MultiHopRAGEngine(
            retriever,
            planner=planner,
            max_steps=4,
        ).ask("question")

        self.assertEqual(planner.calls, 2)
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(result.response.metadata["stop_reason"], "duplicate_planned_query")
        self.assertEqual(result.planning_decisions[-1].metadata["duplicate_query"], "same query")

    def test_anonymous_documents_receive_stable_citable_ids_and_deduplicate(self) -> None:
        class AnonymousRetriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
                return [Document("same anonymous evidence", doc_id=None, score=0.5)]

        llm = _RecordingLLM()
        result = MultiHopRAGEngine(
            AnonymousRetriever(),
            llm_client=llm,
            planner=_AlwaysNextPlanner(),
            max_steps=2,
        ).ask("question")

        self.assertEqual(len(result.retrieved_documents), 1)
        document = result.retrieved_documents[0]
        self.assertTrue(document.doc_id.startswith("multi-hop-anonymous-"))
        self.assertIn(f"[source: {document.doc_id}]", llm.prompts[0])
        provenance = document.metadata["multi_hop_provenance"]
        self.assertEqual(provenance["seen_steps"], [1, 2])
        self.assertTrue(provenance["synthetic_doc_id"])

    def test_false_first_hop_stop_and_stop_with_query_use_safe_fallback(self) -> None:
        for payload in (
            '{"action":"stop","rationale":"unsupported certainty"}',
            '{"action":"stop","query":"hidden next","rationale":"malformed"}',
        ):
            with self.subTest(payload=payload):
                planner = LLMMultiHopPlanner(_QueueLLM([payload]))
                decision = planner.plan(PlanningContext("question", next_step=1, max_steps=2))
                self.assertEqual(decision.action, PlanningAction.NEXT)
                self.assertEqual(decision.query, "question")
                self.assertTrue(decision.rationale.startswith("llm_output_invalid:"))

    def test_malformed_fallback_planner_is_rejected(self) -> None:
        planner = LLMMultiHopPlanner(
            _QueueLLM(["not-json"]),
            fallback=_InvalidFallbackPlanner(),  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(TypeError, "fallback.plan"):
            planner.plan(PlanningContext("question", next_step=1, max_steps=1))

    def test_empty_decomposition_and_incoherent_context_are_rejected_before_retrieval(self) -> None:
        class EmptyDecomposer:
            def decompose(self, query: str, max_steps: int = 4) -> list[str]:
                return []

        retriever = _RoutingRetriever()
        engine = MultiHopRAGEngine(retriever, decomposer=EmptyDecomposer())
        with self.assertRaisesRegex(ValueError, "at least one"):
            engine.retrieve("question")
        self.assertEqual(retriever.calls, [])

        with self.assertRaisesRegex(ValueError, r"len\(hops\)"):
            PlanningContext("question", next_step=2, max_steps=2)

        for invalid, exception in ((True, TypeError), (0, ValueError)):
            with self.subTest(max_steps=invalid):
                with self.assertRaises(exception):
                    RuleBasedMultiHopPlanner().decomposer.decompose("question", max_steps=invalid)

    def test_malformed_retrieval_documents_and_generation_are_rejected(self) -> None:
        invalid_documents = (
            Document("", doc_id="empty"),
            Document("content", metadata=[], doc_id="metadata"),  # type: ignore[arg-type]
            Document("content", doc_id=3),  # type: ignore[arg-type]
            Document("content", doc_id="score", score=float("nan")),
            Document("content", doc_id="bool-score", score=True),
        )
        for document in invalid_documents:
            with self.subTest(document=document):
                retriever = _RoutingRetriever()
                retriever.retrieve = lambda query, top_k, value=document: [value]  # type: ignore[method-assign]
                engine = MultiHopRAGEngine(retriever, planner=_AlwaysNextPlanner(), max_steps=1)
                with self.assertRaises((TypeError, ValueError)):
                    engine.retrieve("question")

        class InvalidAnswerLLM(LLMClient):
            def generate(self, prompt: str, **kwargs: object):
                return {"not": "text"}

        with self.assertRaisesRegex(TypeError, "must return a string"):
            MultiHopRAGEngine(
                _RoutingRetriever(),
                llm_client=InvalidAnswerLLM(),  # type: ignore[arg-type]
                planner=_AlwaysNextPlanner(),
                max_steps=1,
            ).ask("question")


if __name__ == "__main__":
    unittest.main()
