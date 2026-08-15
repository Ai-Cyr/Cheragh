from __future__ import annotations

import math
import unittest

from cheragh.base import BaseRetriever, Document, LLMClient
from cheragh.flare import (
    DraftUncertainty,
    FLAREPipeline,
    TokenConfidence,
    TokenConfidenceUncertaintyEstimator,
)


class _QueueLLM(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def generate(self, prompt: str, **kwargs: object) -> str:
        del prompt, kwargs
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


class _RecordingRetriever(BaseRetriever):
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        self.calls.append((query, top_k))
        return self.documents[:top_k]


class _FixedEstimator:
    def __init__(self, assessment: DraftUncertainty) -> None:
        self.assessment = assessment
        self.calls: list[tuple[str, str, str]] = []

    def assess(self, query: str, partial_answer: str, draft: str) -> DraftUncertainty:
        self.calls.append((query, partial_answer, draft))
        return self.assessment


class FLAREUncertaintyTests(unittest.TestCase):
    def test_low_confidence_spans_drive_active_retrieval(self) -> None:
        retriever = _RecordingRetriever(
            [Document("Le contrat fixe le délai à 14 jours.", doc_id="policy", score=0.9)]
        )
        provider_calls: list[str] = []

        def confidence_provider(draft: str) -> list[TokenConfidence]:
            provider_calls.append(draft)
            return [
                TokenConfidence("Le délai est", 0.95),
                TokenConfidence("peut-être 30 jours", 0.12),
            ]

        pipeline = FLAREPipeline(
            retriever,
            _QueueLLM(
                [
                    "Le délai est peut-être 30 jours.",
                    "Le délai contractuel est de 14 jours. [source: policy]",
                    "[DONE]",
                ]
            ),
            uncertainty_estimator=TokenConfidenceUncertaintyEstimator(
                confidence_provider,
                threshold=0.5,
            ),
        )

        result = pipeline.run("Quel est le délai contractuel ?")

        self.assertEqual(provider_calls, ["Le délai est peut-être 30 jours."])
        self.assertEqual(len(retriever.calls), 1)
        self.assertIn("peut-être 30 jours", retriever.calls[0][0])
        self.assertNotIn("Le délai est\n", retriever.calls[0][0])
        self.assertEqual(result["iterations"][0]["draft_confidence"], 0.12)
        self.assertEqual(
            result["iterations"][0]["low_confidence_spans"],
            ["peut-être 30 jours"],
        )
        self.assertTrue(result["iterations"][0]["retrieval_triggered"])

    def test_high_confidence_draft_skips_retrieval_even_when_long(self) -> None:
        retriever = _RecordingRetriever([Document("unused", doc_id="unused")])
        estimator = TokenConfidenceUncertaintyEstimator(
            lambda draft: [TokenConfidence(draft, 0.99)],
            threshold=0.5,
        )
        draft = "Une phrase suffisamment longue mais déjà certaine."
        pipeline = FLAREPipeline(
            retriever,
            _QueueLLM([draft, "[DONE]"]),
            min_draft_length=1,
            uncertainty_estimator=estimator,
        )

        result = pipeline.run("Question simple")

        self.assertEqual(result["answer"], draft)
        self.assertEqual(retriever.calls, [])
        self.assertFalse(result["iterations"][0]["retrieval_triggered"])

    def test_custom_estimator_sees_partial_answer_at_each_iteration(self) -> None:
        retriever = _RecordingRetriever([])
        estimator = _FixedEstimator(
            DraftUncertainty(False, confidence=0.8, rationale="provider_confident")
        )
        pipeline = FLAREPipeline(
            retriever,
            _QueueLLM(["Première phrase.", "Deuxième phrase.", "[DONE]"]),
            uncertainty_estimator=estimator,
        )

        result = pipeline.run("Question")

        self.assertEqual(result["answer"], "Première phrase. Deuxième phrase.")
        self.assertEqual(estimator.calls[0][1], "")
        self.assertEqual(estimator.calls[1][1], "Première phrase.")
        self.assertEqual(
            [item["uncertainty_rationale"] for item in result["iterations"]],
            ["provider_confident", "provider_confident"],
        )

    def test_confidence_contract_is_strict(self) -> None:
        for value, error in (
            (-0.1, ValueError),
            (1.1, ValueError),
            (float("nan"), ValueError),
            (True, TypeError),
        ):
            with self.subTest(value=value), self.assertRaises(error):
                TokenConfidence("token", value)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            TokenConfidenceUncertaintyEstimator(lambda draft: [], threshold=True)
        with self.assertRaises(ValueError):
            TokenConfidenceUncertaintyEstimator(lambda draft: [], threshold=2.0)
        with self.assertRaises(TypeError):
            TokenConfidenceUncertaintyEstimator(
                lambda draft: [],
                min_low_confidence_tokens=True,
            )
        with self.assertRaises(TypeError):
            DraftUncertainty(True, low_confidence_spans="token")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            DraftUncertainty(True, rationale=1)  # type: ignore[arg-type]
        normalized = DraftUncertainty(True, confidence=1, low_confidence_spans=[" token "])  # type: ignore[arg-type]
        self.assertEqual(normalized.confidence, 1.0)
        self.assertEqual(normalized.low_confidence_spans, ("token",))

        generator_estimator = TokenConfidenceUncertaintyEstimator(
            lambda _draft: (item for item in [TokenConfidence("token", 0.1)])  # type: ignore[arg-type]
        )
        with self.assertRaises(TypeError):
            generator_estimator.assess("query", "", "draft")

    def test_invalid_estimator_output_fails_at_trust_boundary(self) -> None:
        class InvalidEstimator:
            def assess(self, query: str, partial_answer: str, draft: str) -> object:
                del query, partial_answer, draft
                return {"requires_retrieval": True}

        pipeline = FLAREPipeline(
            _RecordingRetriever([]),
            _QueueLLM(["draft"]),
            uncertainty_estimator=InvalidEstimator(),  # type: ignore[arg-type]
        )
        with self.assertRaises(TypeError):
            pipeline.run("Question")

    def test_ask_exposes_shared_response_contract_and_top_k_override(self) -> None:
        source = Document("Valeur vérifiée.", metadata={"page": 2}, doc_id="source", score=0.8)
        retriever = _RecordingRetriever([source])
        pipeline = FLAREPipeline(
            retriever,
            _QueueLLM(["Valeur incertaine.", "Valeur vérifiée. [source: source]", "[DONE]"]),
            uncertainty_estimator=_FixedEstimator(DraftUncertainty(True, confidence=0.1)),
        )

        response = pipeline.ask("Quelle valeur ?", top_k=1)

        self.assertEqual(response.query, "Quelle valeur ?")
        self.assertEqual(response.metadata["architecture"], "flare")
        self.assertTrue(response.metadata["multi_prompt_generation"])
        self.assertEqual(response.retrieved_documents[0].doc_id, "source")
        self.assertEqual(response.sources[0].location, "page=2")
        self.assertEqual(response.citations, ["source"])
        self.assertEqual(retriever.calls[0][1], 1)
        for value, error in ((0, ValueError), (True, TypeError), (1.5, TypeError)):
            with self.subTest(top_k=value), self.assertRaises(error):
                pipeline.ask("Question", top_k=value)  # type: ignore[arg-type]

    def test_retriever_over_return_is_capped_snapshotted_and_validated(self) -> None:
        class _OverReturningRetriever(_RecordingRetriever):
            def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
                self.calls.append((query, top_k))
                return self.documents

        source = Document(
            "first evidence",
            metadata={"nested": {"version": 1}},
            doc_id="first",
            score=1.0,
        )
        retriever = _OverReturningRetriever(
            [source, Document("second evidence", doc_id="second", score=0.5)]
        )
        pipeline = FLAREPipeline(
            retriever,
            _QueueLLM(["uncertain draft", "grounded [source: first]", "[DONE]"]),
            uncertainty_estimator=_FixedEstimator(DraftUncertainty(True)),
        )

        response = pipeline.ask("Question", top_k=1)

        self.assertEqual([document.doc_id for document in response.retrieved_documents], ["first"])
        response.retrieved_documents[0].metadata["nested"]["version"] = 99
        self.assertEqual(source.metadata["nested"]["version"], 1)

        source.score = math.nan
        invalid = FLAREPipeline(
            retriever,
            _QueueLLM(["uncertain draft"]),
            uncertainty_estimator=_FixedEstimator(DraftUncertainty(True)),
        )
        with self.assertRaises(ValueError):
            invalid.run("Question")

    def test_no_retrieval_answer_does_not_report_perfect_grounding(self) -> None:
        pipeline = FLAREPipeline(
            _RecordingRetriever([]),
            _QueueLLM(["Parametric answer.", "[DONE]"]),
            uncertainty_estimator=_FixedEstimator(DraftUncertainty(False)),
        )

        response = pipeline.ask("Question")

        self.assertEqual(response.grounded_score, 0.0)
        self.assertIn("flare_no_retrieval", response.warnings)


if __name__ == "__main__":
    unittest.main()
