"""Claim-level faithfulness and citation-alignment diagnostics.

This module provides an offline, framework-neutral evaluation boundary inspired
by the *kind* of diagnostics exposed by RAGAS and RAGChecker.  It does not claim
to reproduce either framework or to replace a semantic NLI/LLM judge.  The
default :class:`LexicalEntailmentScorer` only measures content-token recall and
cannot reliably detect paraphrases, world knowledge, or contradictions.  Pass
an application-specific :class:`EvidenceEntailmentScorer` for those capabilities.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
import re
from typing import Any, Protocol, runtime_checkable

from ..base import Document, _snapshot_document, _validate_top_k
from ..citations import CITATION_PATTERN
from ..tokenization import RetrievalTokenizer


class ClaimStatus(str, Enum):
    """Outcome of checking one claim against the supplied evidence."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class Claim:
    """A normalized factual claim and the source ids attached to it."""

    text: str
    citations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("claim text must be a non-empty string")
        if isinstance(self.citations, (str, bytes)) or not isinstance(
            self.citations,
            Sequence,
        ):
            raise TypeError("claim citations must be a sequence of strings")
        normalized: list[str] = []
        for citation in self.citations:
            if not isinstance(citation, str) or not citation.strip():
                raise ValueError("claim citations must be non-empty strings")
            value = citation.strip()
            if value not in normalized:
                normalized.append(value)
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "citations", tuple(normalized))


@runtime_checkable
class ClaimSegmenter(Protocol):
    """Adapter boundary for deterministic or model-backed claim segmentation."""

    def segment(self, answer: str) -> Iterable[Claim | str]:
        ...


class SentenceClaimSegmenter:
    """Split an answer deterministically on sentence and line boundaries.

    Citation-only fragments are attached to the preceding sentence.  This is a
    transparent baseline, not semantic atomic-fact decomposition; callers can
    inject a domain-aware segmenter through :class:`ClaimEvaluator`.
    """

    _BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

    def segment(self, answer: str) -> list[Claim]:
        if not isinstance(answer, str):
            raise TypeError("answer must be a string")
        claims: list[Claim] = []
        for fragment in self._BOUNDARY.split(answer):
            raw = fragment.strip(" \t\r\n-*•")
            if not raw:
                continue
            parsed = _parse_claim(raw)
            if parsed is None:
                citations = _citation_ids(raw)
                if citations and claims:
                    previous = claims[-1]
                    claims[-1] = Claim(previous.text, (*previous.citations, *citations))
                continue
            claims.append(parsed)
        return claims


@dataclass(frozen=True)
class EntailmentScore:
    """Calibrated evidence scores returned by an entailment adapter.

    ``entailment`` and ``contradiction`` are independent values in ``[0, 1]``;
    they need not sum to one because adapters may not expose an NLI softmax.
    """

    entailment: float
    contradiction: float = 0.0
    rationale: str | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entailment", _probability(self.entailment, name="entailment"))
        object.__setattr__(
            self,
            "contradiction",
            _probability(self.contradiction, name="contradiction"),
        )
        if self.rationale is not None and not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string or None")
        if self.method is not None and not isinstance(self.method, str):
            raise TypeError("method must be a string or None")


@runtime_checkable
class EvidenceEntailmentScorer(Protocol):
    """Adapter boundary for judging whether one evidence item supports a claim."""

    def score(self, claim: str, evidence: Document) -> EntailmentScore:
        ...


class LexicalEntailmentScorer:
    """Dependency-free content-token recall baseline.

    The score is the share of distinct claim tokens found in an evidence item.
    It deliberately reports zero contradiction probability: token overlap is
    not a semantic contradiction detector.  Negation, paraphrase and numerical
    consistency therefore require an injected NLI/LLM or domain scorer.
    """

    def __init__(self, tokenizer: RetrievalTokenizer | None = None) -> None:
        self.tokenizer = tokenizer if tokenizer is not None else RetrievalTokenizer(ngram_range=(1, 1))

    def score(self, claim: str, evidence: Document) -> EntailmentScore:
        claim_terms = set(self.tokenizer.tokenize(claim))
        evidence_terms = set(self.tokenizer.tokenize(evidence.content))
        recall = len(claim_terms & evidence_terms) / len(claim_terms) if claim_terms else 0.0
        return EntailmentScore(
            entailment=recall,
            contradiction=0.0,
            rationale="content-token recall only; no semantic entailment or contradiction detection",
            method="lexical_token_recall",
        )


@dataclass(frozen=True)
class EvidenceAssessment:
    """Scores and an independent snapshot for one evidence item."""

    document: Document
    rank: int
    entailment_score: float
    contradiction_score: float
    cited: bool
    rationale: str | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("evidence rank must be an int")
        if self.rank <= 0:
            raise ValueError("evidence rank must be > 0")
        _validate_document(self.document, name="evidence document")
        if not isinstance(self.cited, bool):
            raise TypeError("cited must be a boolean")
        if self.rationale is not None and not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string or None")
        if self.method is not None and not isinstance(self.method, str):
            raise TypeError("method must be a string or None")
        object.__setattr__(self, "document", _snapshot_document(self.document))
        object.__setattr__(
            self,
            "entailment_score",
            _probability(self.entailment_score, name="entailment_score"),
        )
        object.__setattr__(
            self,
            "contradiction_score",
            _probability(self.contradiction_score, name="contradiction_score"),
        )

    def snapshot(self) -> "EvidenceAssessment":
        return EvidenceAssessment(
            document=self.document,
            rank=self.rank,
            entailment_score=self.entailment_score,
            contradiction_score=self.contradiction_score,
            cited=self.cited,
            rationale=self.rationale,
            method=self.method,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.document.doc_id,
            "content": self.document.content,
            "metadata": deepcopy(self.document.metadata or {}),
            "document_score": self.document.score,
            "rank": self.rank,
            "entailment_score": self.entailment_score,
            "contradiction_score": self.contradiction_score,
            "cited": self.cited,
            "rationale": self.rationale,
            "method": self.method,
        }


@dataclass(frozen=True)
class CitationAlignment:
    """Claim-level diagnostic for a single citation id."""

    citation_id: str
    known: bool
    aligned: bool
    contradicted: bool
    entailment_score: float | None = None
    contradiction_score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.citation_id, str) or not self.citation_id.strip():
            raise ValueError("citation_id must be a non-empty string")
        if not all(isinstance(value, bool) for value in (self.known, self.aligned, self.contradicted)):
            raise TypeError("known, aligned and contradicted must be booleans")
        object.__setattr__(self, "citation_id", self.citation_id.strip())
        if self.entailment_score is not None:
            object.__setattr__(
                self,
                "entailment_score",
                _probability(self.entailment_score, name="citation entailment_score"),
            )
        if self.contradiction_score is not None:
            object.__setattr__(
                self,
                "contradiction_score",
                _probability(self.contradiction_score, name="citation contradiction_score"),
            )
        if self.aligned and self.contradicted:
            raise ValueError("a citation cannot be both aligned and contradicted")
        if not self.known:
            if self.aligned or self.contradicted:
                raise ValueError("an unknown citation cannot be aligned or contradicted")
            if self.entailment_score is not None or self.contradiction_score is not None:
                raise ValueError("an unknown citation cannot expose evidence scores")
        elif self.entailment_score is None or self.contradiction_score is None:
            raise ValueError("a known citation must expose both evidence scores")


@dataclass(frozen=True)
class ClaimDiagnostic:
    """Support, contradiction, and citation diagnostics for one claim."""

    claim: Claim
    status: ClaimStatus
    support_score: float
    contradiction_score: float
    evidence: tuple[EvidenceAssessment, ...]
    citation_alignments: tuple[CitationAlignment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.claim, Claim):
            raise TypeError("claim must be a Claim")
        try:
            status = self.status if isinstance(self.status, ClaimStatus) else ClaimStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ValueError("status must be a supported ClaimStatus") from exc
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "support_score", _probability(self.support_score, name="support_score"))
        object.__setattr__(
            self,
            "contradiction_score",
            _probability(self.contradiction_score, name="contradiction_score"),
        )
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, EvidenceAssessment) for item in self.evidence
        ):
            raise TypeError("evidence must be a tuple of EvidenceAssessment values")
        if not isinstance(self.citation_alignments, tuple):
            raise TypeError("citation_alignments must be a tuple of CitationAlignment values")
        if any(not isinstance(item, CitationAlignment) for item in self.citation_alignments):
            raise TypeError("citation_alignments must contain only CitationAlignment values")
        if tuple(item.citation_id for item in self.citation_alignments) != self.claim.citations:
            raise ValueError("citation_alignments must correspond exactly to claim citations")
        ranks = [item.rank for item in self.evidence]
        if len(ranks) != len(set(ranks)):
            raise ValueError("evidence ranks must be unique")
        for item in self.evidence:
            expected_cited = item.document.doc_id in self.claim.citations
            if item.cited != expected_cited:
                raise ValueError("evidence cited flags must match claim citations")
        object.__setattr__(self, "evidence", tuple(item.snapshot() for item in self.evidence))
        object.__setattr__(self, "citation_alignments", tuple(self.citation_alignments))

    @property
    def citation_aligned(self) -> bool:
        """Whether every attached citation is known and supports this claim."""

        return bool(self.citation_alignments) and all(item.aligned for item in self.citation_alignments)

    @property
    def uncited(self) -> bool:
        return not self.claim.citations

    @property
    def unknown_citations(self) -> tuple[str, ...]:
        return tuple(item.citation_id for item in self.citation_alignments if not item.known)

    def snapshot(self) -> "ClaimDiagnostic":
        return ClaimDiagnostic(
            claim=Claim(self.claim.text, self.claim.citations),
            status=self.status,
            support_score=self.support_score,
            contradiction_score=self.contradiction_score,
            evidence=self.evidence,
            citation_alignments=self.citation_alignments,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.text,
            "citations": list(self.claim.citations),
            "status": self.status.value,
            "support_score": self.support_score,
            "contradiction_score": self.contradiction_score,
            "citation_aligned": self.citation_aligned,
            "unknown_citations": list(self.unknown_citations),
            "evidence": [item.to_dict() for item in self.evidence],
            "citation_alignments": [
                {
                    "citation_id": item.citation_id,
                    "known": item.known,
                    "aligned": item.aligned,
                    "contradicted": item.contradicted,
                    "entailment_score": item.entailment_score,
                    "contradiction_score": item.contradiction_score,
                }
                for item in self.citation_alignments
            ],
        }


@dataclass(frozen=True)
class ClaimEvaluationResult:
    """Aggregate claim metrics plus immutable-looking diagnostic snapshots."""

    diagnostics: tuple[ClaimDiagnostic, ...]
    faithfulness_score: float
    citation_alignment_score: float
    citation_precision: float
    citation_validity: float
    citation_completeness: float
    contradiction_rate: float
    unsupported_rate: float

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics, tuple) or any(
            not isinstance(item, ClaimDiagnostic) for item in self.diagnostics
        ):
            raise TypeError("diagnostics must be a tuple of ClaimDiagnostic values")
        object.__setattr__(self, "diagnostics", tuple(item.snapshot() for item in self.diagnostics))
        for name in (
            "faithfulness_score",
            "citation_alignment_score",
            "citation_precision",
            "citation_validity",
            "citation_completeness",
            "contradiction_rate",
            "unsupported_rate",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name=name))

    @property
    def faithfulness(self) -> float:
        return self.faithfulness_score

    @property
    def citation_alignment(self) -> float:
        return self.citation_alignment_score

    @property
    def metrics(self) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness_score,
            "citation_alignment": self.citation_alignment_score,
            "citation_precision": self.citation_precision,
            "citation_validity": self.citation_validity,
            "citation_completeness": self.citation_completeness,
            "contradiction_rate": self.contradiction_rate,
            "unsupported_rate": self.unsupported_rate,
        }

    @property
    def supported_claims(self) -> tuple[str, ...]:
        return tuple(item.claim.text for item in self.diagnostics if item.status is ClaimStatus.SUPPORTED)

    @property
    def unsupported_claims(self) -> tuple[str, ...]:
        return tuple(item.claim.text for item in self.diagnostics if item.status is ClaimStatus.UNSUPPORTED)

    @property
    def contradicted_claims(self) -> tuple[str, ...]:
        return tuple(item.claim.text for item in self.diagnostics if item.status is ClaimStatus.CONTRADICTED)

    @property
    def uncited_claims(self) -> tuple[str, ...]:
        return tuple(item.claim.text for item in self.diagnostics if item.uncited)

    @property
    def unknown_citations(self) -> tuple[str, ...]:
        values: list[str] = []
        for diagnostic in self.diagnostics:
            for citation in diagnostic.unknown_citations:
                if citation not in values:
                    values.append(citation)
        return tuple(values)

    def snapshot(self) -> "ClaimEvaluationResult":
        """Return a deep diagnostic snapshot safe for independent mutation."""

        return ClaimEvaluationResult(
            diagnostics=self.diagnostics,
            faithfulness_score=self.faithfulness_score,
            citation_alignment_score=self.citation_alignment_score,
            citation_precision=self.citation_precision,
            citation_validity=self.citation_validity,
            citation_completeness=self.citation_completeness,
            contradiction_rate=self.contradiction_rate,
            unsupported_rate=self.unsupported_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "supported_claims": list(self.supported_claims),
            "unsupported_claims": list(self.unsupported_claims),
            "contradicted_claims": list(self.contradicted_claims),
            "uncited_claims": list(self.uncited_claims),
            "unknown_citations": list(self.unknown_citations),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


SegmenterLike = ClaimSegmenter | Callable[[str], Iterable[Claim | str]]
ScorerLike = EvidenceEntailmentScorer | Callable[[str, Document], EntailmentScore]


class ClaimEvaluator:
    """Evaluate claim support separately from citation-source alignment.

    All supplied documents are scored so aggregate metrics and cited-source
    checks are complete.  ``top_k`` limits only the ranked evidence details
    retained on each :class:`ClaimDiagnostic`.
    """

    def __init__(
        self,
        *,
        segmenter: SegmenterLike | None = None,
        scorer: ScorerLike | None = None,
        support_threshold: float = 0.6,
        contradiction_threshold: float = 0.6,
        top_k: int = 5,
    ) -> None:
        self.segmenter = segmenter if segmenter is not None else SentenceClaimSegmenter()
        self.scorer = scorer if scorer is not None else LexicalEntailmentScorer()
        if not callable(self.segmenter) and not callable(getattr(self.segmenter, "segment", None)):
            raise TypeError("segmenter must be callable or define segment()")
        if not callable(self.scorer) and not callable(getattr(self.scorer, "score", None)):
            raise TypeError("scorer must be callable or define score()")
        self.support_threshold = _probability(support_threshold, name="support_threshold")
        self.contradiction_threshold = _probability(
            contradiction_threshold,
            name="contradiction_threshold",
        )
        self.top_k = _validate_top_k(top_k)

    def evaluate(
        self,
        answer: str,
        documents: Iterable[Document],
        *,
        top_k: int | None = None,
    ) -> ClaimEvaluationResult:
        if not isinstance(answer, str):
            raise TypeError("answer must be a string")
        retained_top_k = self.top_k if top_k is None else _validate_top_k(top_k)
        evidence = _validated_snapshots(documents)
        claims = _run_segmenter(self.segmenter, answer)
        diagnostics = tuple(
            self._evaluate_claim(claim, evidence, top_k=retained_top_k) for claim in claims
        )
        return _aggregate(diagnostics)

    def _evaluate_claim(
        self,
        claim: Claim,
        documents: Sequence[Document],
        *,
        top_k: int,
    ) -> ClaimDiagnostic:
        assessments: list[EvidenceAssessment] = []
        cited_ids = set(claim.citations)
        for rank, document in enumerate(documents, start=1):
            # The adapter gets its own copy so an impure scorer cannot corrupt
            # the evidence snapshot retained in diagnostics.
            raw_score = _run_scorer(self.scorer, claim.text, _snapshot_document(document))
            assessments.append(
                EvidenceAssessment(
                    document=document,
                    rank=rank,
                    entailment_score=raw_score.entailment,
                    contradiction_score=raw_score.contradiction,
                    cited=document.doc_id is not None and str(document.doc_id) in cited_ids,
                    rationale=raw_score.rationale,
                    method=raw_score.method,
                )
            )

        support = max((item.entailment_score for item in assessments), default=0.0)
        contradiction = max((item.contradiction_score for item in assessments), default=0.0)
        if contradiction >= self.contradiction_threshold and contradiction > support:
            status = ClaimStatus.CONTRADICTED
        elif support >= self.support_threshold and support > contradiction:
            status = ClaimStatus.SUPPORTED
        else:
            status = ClaimStatus.UNSUPPORTED

        alignments = tuple(self._align_citation(citation, assessments) for citation in claim.citations)
        ranked = sorted(
            assessments,
            key=lambda item: (
                max(item.entailment_score, item.contradiction_score),
                item.entailment_score,
                item.contradiction_score,
                item.cited,
                -item.rank,
            ),
            reverse=True,
        )
        return ClaimDiagnostic(
            claim=claim,
            status=status,
            support_score=support,
            contradiction_score=contradiction,
            evidence=tuple(ranked[:top_k]),
            citation_alignments=alignments,
        )

    def _align_citation(
        self,
        citation: str,
        assessments: Sequence[EvidenceAssessment],
    ) -> CitationAlignment:
        matching = [
            item
            for item in assessments
            if item.document.doc_id is not None and str(item.document.doc_id) == citation
        ]
        if not matching:
            return CitationAlignment(
                citation_id=citation,
                known=False,
                aligned=False,
                contradicted=False,
            )
        entailment = max(item.entailment_score for item in matching)
        contradiction = max(item.contradiction_score for item in matching)
        contradicted = contradiction >= self.contradiction_threshold and contradiction > entailment
        aligned = entailment >= self.support_threshold and entailment > contradiction
        return CitationAlignment(
            citation_id=citation,
            known=True,
            aligned=aligned,
            contradicted=contradicted,
            entailment_score=entailment,
            contradiction_score=contradiction,
        )


def evaluate_claims(
    answer: str,
    documents: Iterable[Document],
    *,
    segmenter: SegmenterLike | None = None,
    scorer: ScorerLike | None = None,
    support_threshold: float = 0.6,
    contradiction_threshold: float = 0.6,
    top_k: int = 5,
) -> ClaimEvaluationResult:
    """Convenience function for one claim-level answer evaluation."""

    evaluator = ClaimEvaluator(
        segmenter=segmenter,
        scorer=scorer,
        support_threshold=support_threshold,
        contradiction_threshold=contradiction_threshold,
        top_k=top_k,
    )
    return evaluator.evaluate(answer, documents)


def _parse_claim(value: str) -> Claim | None:
    citations = _citation_ids(value)
    text = CITATION_PATTERN.sub("", value).strip(" \t\r\n-*•")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    if not text:
        return None
    return Claim(text=text, citations=citations)


def _citation_ids(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(match.strip() for match in CITATION_PATTERN.findall(value) if match.strip())
    )


def _run_segmenter(segmenter: SegmenterLike, answer: str) -> tuple[Claim, ...]:
    method = getattr(segmenter, "segment", None)
    raw_claims = method(answer) if callable(method) else segmenter(answer)  # type: ignore[operator]
    if isinstance(raw_claims, (str, bytes)) or not isinstance(raw_claims, Iterable):
        raise TypeError("claim segmenter must return an iterable of Claim or str values")
    claims: list[Claim] = []
    for raw in raw_claims:
        if isinstance(raw, Claim):
            parsed = _parse_claim(raw.text)
            if parsed is None:
                raise ValueError("claim segmenter returned an empty claim")
            claims.append(Claim(parsed.text, (*raw.citations, *parsed.citations)))
        elif isinstance(raw, str):
            parsed = _parse_claim(raw)
            if parsed is None:
                raise ValueError("claim segmenter returned an empty claim")
            claims.append(parsed)
        else:
            raise TypeError("claim segmenter must return only Claim or str values")
    return tuple(claims)


def _run_scorer(scorer: ScorerLike, claim: str, evidence: Document) -> EntailmentScore:
    method = getattr(scorer, "score", None)
    raw = method(claim, evidence) if callable(method) else scorer(claim, evidence)  # type: ignore[operator]
    if isinstance(raw, EntailmentScore):
        return raw
    if isinstance(raw, Mapping):
        return EntailmentScore(
            entailment=raw.get("entailment", raw.get("support", 0.0)),
            contradiction=raw.get("contradiction", 0.0),
            rationale=raw.get("rationale"),
            method=raw.get("method"),
        )
    if isinstance(raw, Real) and not isinstance(raw, bool):
        return EntailmentScore(float(raw))
    raise TypeError("entailment scorer must return EntailmentScore, a score mapping, or a number")


def _validated_snapshots(documents: Iterable[Document]) -> tuple[Document, ...]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Iterable):
        raise TypeError("documents must be an iterable of Document instances")
    snapshots: list[Document] = []
    seen_ids: set[str] = set()
    for index, document in enumerate(documents):
        _validate_document(document, name=f"documents[{index}]")
        if document.doc_id:
            if document.doc_id in seen_ids:
                raise ValueError(f"document IDs must be unique: {document.doc_id!r}")
            seen_ids.add(document.doc_id)
        snapshots.append(_snapshot_document(document))
    return tuple(snapshots)


def _validate_document(document: Any, *, name: str) -> None:
    if not isinstance(document, Document):
        raise TypeError(f"{name} must be a Document")
    if not isinstance(document.content, str):
        raise TypeError(f"{name}.content must be a string")
    if not isinstance(document.metadata, dict):
        raise TypeError(f"{name}.metadata must be a dict")
    if document.doc_id is not None and not isinstance(document.doc_id, str):
        raise TypeError(f"{name}.doc_id must be a string or None")
    if document.score is not None:
        if isinstance(document.score, bool) or not isinstance(document.score, Real):
            raise TypeError(f"{name}.score must be a real number or None")
        if not math.isfinite(float(document.score)):
            raise ValueError(f"{name}.score must be finite")


def _aggregate(diagnostics: tuple[ClaimDiagnostic, ...]) -> ClaimEvaluationResult:
    total = len(diagnostics)
    if not total:
        return ClaimEvaluationResult(
            diagnostics=(),
            # No claims means no evaluated quality signal. Reporting perfect
            # scores here makes empty answers look better than grounded ones.
            faithfulness_score=0.0,
            citation_alignment_score=0.0,
            citation_precision=0.0,
            citation_validity=0.0,
            citation_completeness=0.0,
            contradiction_rate=0.0,
            unsupported_rate=0.0,
        )

    alignments = [alignment for item in diagnostics for alignment in item.citation_alignments]
    aligned_count = sum(alignment.aligned for alignment in alignments)
    known_count = sum(alignment.known for alignment in alignments)
    return ClaimEvaluationResult(
        diagnostics=diagnostics,
        faithfulness_score=sum(item.status is ClaimStatus.SUPPORTED for item in diagnostics) / total,
        citation_alignment_score=sum(item.citation_aligned for item in diagnostics) / total,
        citation_precision=aligned_count / len(alignments) if alignments else 0.0,
        citation_validity=known_count / len(alignments) if alignments else 0.0,
        citation_completeness=sum(not item.uncited for item in diagnostics) / total,
        contradiction_rate=sum(item.status is ClaimStatus.CONTRADICTED for item in diagnostics) / total,
        unsupported_rate=sum(item.status is ClaimStatus.UNSUPPORTED for item in diagnostics) / total,
    )


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return normalized


__all__ = [
    "CitationAlignment",
    "Claim",
    "ClaimDiagnostic",
    "ClaimEvaluationResult",
    "ClaimEvaluator",
    "ClaimSegmenter",
    "ClaimStatus",
    "EntailmentScore",
    "EvidenceAssessment",
    "EvidenceEntailmentScorer",
    "LexicalEntailmentScorer",
    "SentenceClaimSegmenter",
    "evaluate_claims",
]
