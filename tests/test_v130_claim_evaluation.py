from __future__ import annotations

import unittest

from cheragh.base import Document
from cheragh.evaluation import (
    CitationAlignment,
    Claim,
    ClaimDiagnostic,
    ClaimEvaluationResult,
    ClaimEvaluator,
    ClaimStatus,
    EntailmentScore,
    EvidenceAssessment,
    LexicalEntailmentScorer,
    SentenceClaimSegmenter,
    evaluate_claims,
)


class _ControlledScorer:
    def score(self, claim: str, evidence: Document) -> EntailmentScore:
        if evidence.doc_id == "support":
            return EntailmentScore(0.95, 0.02, method="controlled")
        if evidence.doc_id == "conflict":
            return EntailmentScore(0.05, 0.97, method="controlled")
        return EntailmentScore(0.05, 0.01, method="controlled")


class _MutatingScorer:
    def score(self, claim: str, evidence: Document) -> EntailmentScore:
        evidence.content = "mutated by scorer"
        evidence.metadata["nested"]["owner"] = "scorer"
        return EntailmentScore(0.8)


class ClaimEvaluationTests(unittest.TestCase):
    def test_sentence_segmentation_attaches_citation_only_fragments(self) -> None:
        claims = SentenceClaimSegmenter().segment(
            "Paris est en France. [source: france]\nLa Lune orbite la Terre [source: moon]."
        )

        self.assertEqual([claim.text for claim in claims], ["Paris est en France.", "La Lune orbite la Terre."])
        self.assertEqual(claims[0].citations, ("france",))
        self.assertEqual(claims[1].citations, ("moon",))

    def test_lexical_fallback_is_robust_to_irrelevant_evidence(self) -> None:
        documents = [
            Document("Paris est la capitale de la France.", doc_id="france"),
            Document("Les octets, les volcans et le jazz sont sans rapport.", doc_id="noise"),
        ]
        result = evaluate_claims(
            "Paris est la capitale de la France [source: france]. "
            "La Lune est faite de fromage [source: noise].",
            documents,
        )

        self.assertEqual(result.supported_claims, ("Paris est la capitale de la France.",))
        self.assertEqual(result.unsupported_claims, ("La Lune est faite de fromage.",))
        self.assertAlmostEqual(result.faithfulness, 0.5)
        self.assertAlmostEqual(result.citation_alignment, 0.5)
        self.assertAlmostEqual(result.citation_precision, 0.5)
        self.assertEqual(result.contradiction_rate, 0.0)
        self.assertAlmostEqual(result.unsupported_rate, 0.5)

    def test_faithfulness_and_citation_alignment_are_separate(self) -> None:
        result = evaluate_claims(
            "Paris est la capitale de la France [source: noise].",
            [
                Document("Paris est la capitale de la France.", doc_id="support"),
                Document("Les abeilles produisent du miel.", doc_id="noise"),
            ],
        )

        self.assertEqual(result.faithfulness, 1.0)
        self.assertEqual(result.citation_alignment, 0.0)
        self.assertEqual(result.diagnostics[0].status, ClaimStatus.SUPPORTED)
        self.assertTrue(result.diagnostics[0].citation_alignments[0].known)
        self.assertFalse(result.diagnostics[0].citation_alignments[0].aligned)

    def test_unknown_citation_is_reported_even_when_claim_has_support(self) -> None:
        result = evaluate_claims(
            "Paris est la capitale de la France [source: ghost].",
            [Document("Paris est la capitale de la France.", doc_id="france")],
        )

        self.assertEqual(result.faithfulness, 1.0)
        self.assertEqual(result.citation_alignment, 0.0)
        self.assertEqual(result.citation_validity, 0.0)
        self.assertEqual(result.unknown_citations, ("ghost",))
        self.assertEqual(result.diagnostics[0].unknown_citations, ("ghost",))

    def test_injected_entailment_scorer_reports_contradiction(self) -> None:
        evaluator = ClaimEvaluator(
            segmenter=lambda _answer: [Claim("La Terre est plate", ("conflict",))],
            scorer=_ControlledScorer(),
        )
        result = evaluator.evaluate(
            "ignored",
            [Document("La Terre est sphérique.", doc_id="conflict")],
        )

        diagnostic = result.diagnostics[0]
        self.assertEqual(diagnostic.status, ClaimStatus.CONTRADICTED)
        self.assertEqual(result.contradicted_claims, ("La Terre est plate",))
        self.assertEqual(result.contradiction_rate, 1.0)
        self.assertTrue(diagnostic.citation_alignments[0].contradicted)
        self.assertFalse(diagnostic.citation_aligned)

    def test_supported_claim_without_citation_is_flagged_as_uncited(self) -> None:
        result = evaluate_claims(
            "Paris est la capitale de la France.",
            [Document("Paris est la capitale de la France.", doc_id="france")],
        )

        self.assertEqual(result.faithfulness, 1.0)
        self.assertEqual(result.citation_alignment, 0.0)
        self.assertEqual(result.citation_completeness, 0.0)
        self.assertEqual(result.uncited_claims, ("Paris est la capitale de la France.",))
        self.assertEqual(result.citation_precision, 0.0)

    def test_documents_scorer_and_returned_snapshots_are_isolated(self) -> None:
        source = Document(
            "original evidence",
            metadata={"nested": {"owner": "caller"}},
            doc_id="source",
        )
        result = ClaimEvaluator(
            segmenter=lambda _answer: [Claim("claim", ("source",))],
            scorer=_MutatingScorer(),
            top_k=1,
        ).evaluate("ignored", [source])

        retained = result.diagnostics[0].evidence[0].document
        self.assertEqual(source.content, "original evidence")
        self.assertEqual(source.metadata["nested"]["owner"], "caller")
        self.assertEqual(retained.content, "original evidence")
        self.assertEqual(retained.metadata["nested"]["owner"], "caller")

        source.content = "changed later"
        source.metadata["nested"]["owner"] = "changed later"
        self.assertEqual(retained.content, "original evidence")
        self.assertEqual(retained.metadata["nested"]["owner"], "caller")

        copied = result.snapshot()
        copied.diagnostics[0].evidence[0].document.metadata["nested"]["owner"] = "copy"
        self.assertEqual(retained.metadata["nested"]["owner"], "caller")

        serialized = result.to_dict()
        serialized["diagnostics"][0]["evidence"][0]["metadata"]["nested"]["owner"] = "dict"
        self.assertEqual(retained.metadata["nested"]["owner"], "caller")

    def test_top_k_limits_diagnostic_evidence_and_is_validated(self) -> None:
        evaluator = ClaimEvaluator(
            segmenter=lambda _answer: ["claim"],
            scorer=lambda _claim, evidence: 0.9 if evidence.doc_id == "one" else 0.2,
            top_k=1,
        )
        documents = [Document("one", doc_id="one"), Document("two", doc_id="two")]

        result = evaluator.evaluate("ignored", documents)
        self.assertEqual(len(result.diagnostics[0].evidence), 1)
        self.assertEqual(result.diagnostics[0].evidence[0].document.doc_id, "one")

        for invalid in (0, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    ClaimEvaluator(top_k=invalid)
                with self.assertRaises(ValueError):
                    evaluator.evaluate("ignored", documents, top_k=invalid)
        for invalid in (True, 1.5, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    ClaimEvaluator(top_k=invalid)  # type: ignore[arg-type]

    def test_lexical_fallback_does_not_pretend_to_detect_negation(self) -> None:
        score = LexicalEntailmentScorer().score(
            "Paris is not the capital of France",
            Document("Paris is the capital of France"),
        )

        self.assertGreater(score.entailment, 0.0)
        self.assertEqual(score.contradiction, 0.0)
        self.assertEqual(score.method, "lexical_token_recall")
        self.assertIn("no semantic", score.rationale or "")

    def test_empty_answer_has_no_positive_quality_signal_and_invalid_scores_fail(self) -> None:
        result = evaluate_claims("", [])
        self.assertEqual(result.metrics["faithfulness"], 0.0)
        self.assertEqual(result.metrics["citation_alignment"], 0.0)
        self.assertTrue(all(score == 0.0 for score in result.metrics.values()))
        self.assertEqual(result.diagnostics, ())

        with self.assertRaises(ValueError):
            EntailmentScore(1.1)
        with self.assertRaises(TypeError):
            EntailmentScore(True)  # type: ignore[arg-type]

    def test_claim_and_adapter_boundaries_reject_ambiguous_values(self) -> None:
        with self.assertRaises(TypeError):
            Claim("claim", "source")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ClaimEvaluator(segmenter=object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ClaimEvaluator(scorer=object())  # type: ignore[arg-type]

        for score, expected_error in ((True, TypeError), (float("nan"), ValueError)):
            with self.subTest(score=score):
                with self.assertRaises(expected_error):
                    evaluate_claims(
                        "claim",
                        [Document("evidence", doc_id="source", score=score)],  # type: ignore[arg-type]
                    )

        with self.assertRaises(ValueError):
            evaluate_claims(
                "claim",
                [Document("one", doc_id="same"), Document("two", doc_id="same")],
            )

    def test_direct_diagnostics_cannot_claim_inconsistent_citation_state(self) -> None:
        assessment = EvidenceAssessment(
            document=Document("evidence", doc_id="source"),
            rank=1,
            entailment_score=0.9,
            contradiction_score=0.0,
            cited=True,
        )
        alignment = CitationAlignment(
            citation_id="source",
            known=True,
            aligned=True,
            contradicted=False,
            entailment_score=0.9,
            contradiction_score=0.0,
        )
        diagnostic = ClaimDiagnostic(
            claim=Claim("claim", ("source",)),
            status=ClaimStatus.SUPPORTED,
            support_score=0.9,
            contradiction_score=0.0,
            evidence=(assessment,),
            citation_alignments=(alignment,),
        )
        result = ClaimEvaluationResult(
            diagnostics=(diagnostic,),
            faithfulness_score=1.0,
            citation_alignment_score=1.0,
            citation_precision=1.0,
            citation_validity=1.0,
            citation_completeness=1.0,
            contradiction_rate=0.0,
            unsupported_rate=0.0,
        )
        self.assertEqual(result.diagnostics[0].claim.citations, ("source",))

        with self.assertRaises(ValueError):
            CitationAlignment(
                citation_id="missing",
                known=False,
                aligned=True,
                contradicted=False,
            )
        with self.assertRaises(ValueError):
            ClaimDiagnostic(
                claim=Claim("claim", ("other",)),
                status=ClaimStatus.SUPPORTED,
                support_score=0.9,
                contradiction_score=0.0,
                evidence=(assessment,),
                citation_alignments=(alignment,),
            )


if __name__ == "__main__":
    unittest.main()
