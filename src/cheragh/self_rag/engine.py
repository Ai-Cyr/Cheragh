"""Experimental, inference-time Self-RAG orchestration.

This module implements the *inference loop* commonly associated with Self-RAG:
decide whether retrieval is useful, grade retrieved evidence, generate an answer,
and refine it while a support critic finds unsupported claims.

Maturity and limitations
------------------------
This API is experimental.  It does not train a model, add reflection tokens, or
claim to reproduce the original Self-RAG training procedure.  Quality depends on
the injected retriever, generator, and critics.  The dependency-free lexical
critic is a deterministic baseline intended for tests and small demonstrations,
not a substitute for a calibrated entailment model or human review.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Protocol

from ..base import (
    BaseRetriever,
    Document,
    ExtractiveLLMClient,
    LLMClient,
    _validate_non_negative_int,
    _validate_top_k,
)


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class RetrievalDecision:
    """Decision made before retrieval."""

    should_retrieve: bool
    confidence: float = 1.0
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", _clamp_score(self.confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_retrieve": self.should_retrieve,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class RetrievalGate(Protocol):
    """Strategy deciding whether a query needs external evidence."""

    def decide(self, query: str) -> RetrievalDecision:
        """Return a structured retrieval decision."""


@dataclass(frozen=True)
class StaticRetrievalGate:
    """Deterministic gate useful in tests and explicitly configured flows."""

    should_retrieve: bool = True
    confidence: float = 1.0
    reason: str = "configured_policy"

    def decide(self, query: str) -> RetrievalDecision:
        del query
        return RetrievalDecision(self.should_retrieve, self.confidence, self.reason)


class AlwaysRetrieveGate(StaticRetrievalGate):
    """Default conservative gate: always obtain evidence before answering."""

    def __init__(self) -> None:
        super().__init__(should_retrieve=True, confidence=1.0, reason="always_retrieve")


@dataclass(frozen=True)
class EvidenceRelevance:
    """Relevance grade for one retrieved document."""

    document_index: int
    doc_id: str | None
    score: float
    relevant: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if self.document_index < 0:
            raise ValueError("document_index must be >= 0")
        object.__setattr__(self, "score", _clamp_score(self.score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_index": self.document_index,
            "doc_id": self.doc_id,
            "score": self.score,
            "relevant": self.relevant,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RelevanceAssessment:
    """Collection of per-document relevance grades."""

    evidence: tuple[EvidenceRelevance, ...] = ()
    reason: str = ""

    @property
    def passed(self) -> bool:
        return any(item.relevant for item in self.evidence)

    @property
    def score(self) -> float:
        if not self.evidence:
            return 0.0
        return max(item.score for item in self.evidence)

    @property
    def relevant_indices(self) -> tuple[int, ...]:
        return tuple(item.document_index for item in self.evidence if item.relevant)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class SupportAssessment:
    """Critique of whether an answer is supported by selected evidence."""

    score: float
    supported: bool
    reason: str = ""
    unsupported_claims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _clamp_score(self.score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "supported": self.supported,
            "reason": self.reason,
            "unsupported_claims": list(self.unsupported_claims),
        }


class EvidenceCritic(Protocol):
    """Grades both evidence relevance and answer support."""

    def assess_relevance(self, query: str, documents: Iterable[Document]) -> RelevanceAssessment:
        """Grade every retrieved document for relevance to ``query``."""

    def assess_support(
        self,
        query: str,
        answer: str,
        documents: Iterable[Document],
    ) -> SupportAssessment:
        """Grade whether ``answer`` is supported by ``documents``."""


class LexicalEvidenceCritic:
    """Dependency-free, deterministic lexical relevance/support baseline.

    Token overlap is transparent and reproducible, but it cannot detect logical
    contradiction or reliably establish entailment.  Use a stronger injected
    critic for production factuality decisions.
    """

    def __init__(self, relevance_threshold: float = 0.1, support_threshold: float = 0.55):
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0 and 1")
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError("support_threshold must be between 0 and 1")
        self.relevance_threshold = relevance_threshold
        self.support_threshold = support_threshold

    def assess_relevance(self, query: str, documents: Iterable[Document]) -> RelevanceAssessment:
        query_terms = _content_terms(query)
        grades: list[EvidenceRelevance] = []
        for index, document in enumerate(documents):
            document_terms = _content_terms(document.content)
            score = _coverage(query_terms, document_terms)
            relevant = bool(document_terms) and (not query_terms or score >= self.relevance_threshold)
            grades.append(
                EvidenceRelevance(
                    document_index=index,
                    doc_id=document.doc_id,
                    score=score,
                    relevant=relevant,
                    reason="lexical_overlap" if relevant else "low_lexical_overlap",
                )
            )
        reason = "relevant_evidence_found" if any(item.relevant for item in grades) else "no_relevant_evidence"
        return RelevanceAssessment(tuple(grades), reason=reason)

    def assess_support(
        self,
        query: str,
        answer: str,
        documents: Iterable[Document],
    ) -> SupportAssessment:
        del query
        evidence_terms: set[str] = set()
        for document in documents:
            evidence_terms.update(_content_terms(document.content))

        claims = _claims(answer)
        if not claims:
            return SupportAssessment(0.0, False, "empty_answer", ())
        if not evidence_terms:
            return SupportAssessment(0.0, False, "no_evidence", tuple(claims))

        claim_scores: list[float] = []
        unsupported: list[str] = []
        for claim in claims:
            terms = _content_terms(claim)
            score = _coverage(terms, evidence_terms)
            claim_scores.append(score)
            if not terms or score < self.support_threshold:
                unsupported.append(claim)
        overall = sum(claim_scores) / len(claim_scores)
        supported = not unsupported and overall >= self.support_threshold
        return SupportAssessment(
            score=overall,
            supported=supported,
            reason="lexical_support" if supported else "insufficient_lexical_support",
            unsupported_claims=tuple(unsupported),
        )


class ScriptedEvidenceCritic:
    """Deterministic critic with configured grades for tests and simulations.

    Support assessments are consumed in order.  Once the sequence is exhausted,
    the last assessment is reused.  A new instance should be used for each
    independent scenario.
    """

    def __init__(
        self,
        *,
        relevant: bool = True,
        relevance_score: float = 1.0,
        support_assessments: Iterable[SupportAssessment] | None = None,
    ):
        self.relevant = relevant
        self.relevance_score = _clamp_score(relevance_score)
        self._support = (
            tuple(support_assessments)
            if support_assessments is not None
            else (SupportAssessment(1.0, True, "configured_support", ()),)
        )
        if not self._support:
            raise ValueError("support_assessments must not be empty")
        self._support_index = 0

    def assess_relevance(self, query: str, documents: Iterable[Document]) -> RelevanceAssessment:
        del query
        grades = tuple(
            EvidenceRelevance(
                document_index=index,
                doc_id=document.doc_id,
                score=self.relevance_score,
                relevant=self.relevant,
                reason="configured_relevance",
            )
            for index, document in enumerate(documents)
        )
        return RelevanceAssessment(grades, reason="configured_relevance")

    def assess_support(
        self,
        query: str,
        answer: str,
        documents: Iterable[Document],
    ) -> SupportAssessment:
        del query, answer, documents
        index = min(self._support_index, len(self._support) - 1)
        assessment = self._support[index]
        self._support_index += 1
        return assessment


@dataclass
class SelfRAGIteration:
    """One generation or refinement attempt and its support critique."""

    number: int
    kind: str
    prompt: str
    answer: str
    support: SupportAssessment | None = None

    def to_dict(self, *, include_prompts: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "number": self.number,
            "kind": self.kind,
            "answer": self.answer,
            "support": self.support.to_dict() if self.support else None,
        }
        if include_prompts:
            data["prompt"] = self.prompt
        return data


@dataclass
class SelfRAGTrace:
    """Structured trace of retrieval, criticism, and refinement decisions."""

    query: str
    retrieval: RetrievalDecision
    retrieved_count: int = 0
    relevance: RelevanceAssessment | None = None
    iterations: list[SelfRAGIteration] = field(default_factory=list)
    stop_reason: str = ""

    def to_dict(self, *, include_prompts: bool = False) -> dict[str, Any]:
        return {
            "query": self.query,
            "retrieval": self.retrieval.to_dict(),
            "retrieved_count": self.retrieved_count,
            "relevance": self.relevance.to_dict() if self.relevance else None,
            "iterations": [item.to_dict(include_prompts=include_prompts) for item in self.iterations],
            "stop_reason": self.stop_reason,
        }


@dataclass
class SelfRAGResult:
    """Final answer and auditable Self-RAG inference trace."""

    query: str
    answer: str
    documents: list[Document]
    trace: SelfRAGTrace
    status: str
    maturity: str = "experimental"

    @property
    def supported(self) -> bool | None:
        if not self.trace.iterations:
            return False if self.trace.retrieval.should_retrieve else None
        assessment = self.trace.iterations[-1].support
        return assessment.supported if assessment is not None else None

    def to_dict(self, *, include_prompts: bool = False) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "status": self.status,
            "maturity": self.maturity,
            "supported": self.supported,
            "documents": [
                {
                    "doc_id": document.doc_id,
                    "score": document.score,
                    "content": document.content,
                    "metadata": dict(document.metadata),
                }
                for document in self.documents
            ],
            "trace": self.trace.to_dict(include_prompts=include_prompts),
        }


class SelfRAGEngine:
    """Bounded, modular Self-RAG-style inference loop.

    The engine deliberately separates the retrieval gate, retriever, evidence
    critic, and generator so applications can inject calibrated implementations.
    It performs at most ``1 + max_refinements`` generation calls.
    """

    def __init__(
        self,
        retriever: BaseRetriever | None,
        llm_client: LLMClient | None = None,
        *,
        retrieval_gate: RetrievalGate | None = None,
        evidence_critic: EvidenceCritic | None = None,
        top_k: int = 5,
        max_refinements: int = 2,
        insufficient_evidence_answer: str = (
            "Je ne sais pas : les éléments récupérés ne sont pas suffisamment pertinents."
        ),
    ):
        top_k = _validate_top_k(top_k)
        max_refinements = _validate_non_negative_int(max_refinements, name="max_refinements")
        self.retriever = retriever
        self.llm_client = llm_client or ExtractiveLLMClient()
        self.retrieval_gate = retrieval_gate or AlwaysRetrieveGate()
        self.evidence_critic = evidence_critic or LexicalEvidenceCritic()
        self.top_k = top_k
        self.max_refinements = max_refinements
        self.insufficient_evidence_answer = insufficient_evidence_answer

    def ask(self, query: str, *, top_k: int | None = None, **generate_kwargs: Any) -> SelfRAGResult:
        query = " ".join(query.split())
        if not query:
            raise ValueError("query must not be empty")
        selected_top_k = self.top_k if top_k is None else _validate_top_k(top_k)

        retrieval = self.retrieval_gate.decide(query)
        if not isinstance(retrieval, RetrievalDecision):
            raise TypeError("retrieval_gate.decide() must return RetrievalDecision")
        trace = SelfRAGTrace(query=query, retrieval=retrieval)
        documents: list[Document] = []

        if retrieval.should_retrieve:
            if self.retriever is None:
                raise RuntimeError("the retrieval gate requested evidence, but no retriever is configured")
            retrieved = list(self.retriever.retrieve(query, top_k=selected_top_k))
            if not all(isinstance(document, Document) for document in retrieved):
                raise TypeError("retriever.retrieve() must return Document objects")
            trace.retrieved_count = len(retrieved)
            relevance = self.evidence_critic.assess_relevance(query, retrieved)
            if not isinstance(relevance, RelevanceAssessment):
                raise TypeError("evidence_critic.assess_relevance() must return RelevanceAssessment")
            trace.relevance = relevance
            valid_indices = {index for index in relevance.relevant_indices if index < len(retrieved)}
            documents = [document for index, document in enumerate(retrieved) if index in valid_indices]
            if not documents:
                trace.stop_reason = "insufficient_relevant_evidence"
                return SelfRAGResult(
                    query=query,
                    answer=self.insufficient_evidence_answer,
                    documents=[],
                    trace=trace,
                    status="insufficient_evidence",
                )

        context = _format_context(documents)
        answer = ""
        previous_support: SupportAssessment | None = None
        for attempt in range(self.max_refinements + 1):
            if attempt == 0:
                kind = "generation"
                prompt = _generation_prompt(query, context, retrieval.should_retrieve)
            else:
                kind = "refinement"
                prompt = _refinement_prompt(query, context, answer, previous_support)
            answer = str(self.llm_client.generate(prompt, **generate_kwargs))
            support: SupportAssessment | None = None
            if retrieval.should_retrieve:
                support = self.evidence_critic.assess_support(query, answer, documents)
                if not isinstance(support, SupportAssessment):
                    raise TypeError("evidence_critic.assess_support() must return SupportAssessment")
            trace.iterations.append(SelfRAGIteration(attempt + 1, kind, prompt, answer, support))
            previous_support = support
            if support is None:
                trace.stop_reason = "retrieval_not_requested"
                return SelfRAGResult(query, answer, documents, trace, "completed_without_retrieval")
            if support.supported:
                trace.stop_reason = "answer_supported"
                return SelfRAGResult(query, answer, documents, trace, "supported")

        trace.stop_reason = "refinement_limit_reached"
        return SelfRAGResult(query, answer, documents, trace, "unsupported")

    def run(self, query: str, **kwargs: Any) -> SelfRAGResult:
        """Alias for :meth:`ask`."""
        return self.ask(query, **kwargs)


def _generation_prompt(query: str, context: str, has_evidence: bool) -> str:
    if has_evidence:
        instruction = (
            "Réponds uniquement avec les faits étayés par les éléments ci-dessous. "
            "Si les éléments ne suffisent pas, dis-le explicitement."
        )
    else:
        instruction = (
            "La politique de récupération a décidé de ne pas consulter de source externe. "
            "Réponds prudemment et signale toute incertitude."
        )
    return f"{instruction}\n\nQuestion:\n{query}\n\nÉléments:\n{context or '(aucun)'}\n\nRéponse:"


def _refinement_prompt(
    query: str,
    context: str,
    answer: str,
    support: SupportAssessment | None,
) -> str:
    unsupported = "\n".join(f"- {claim}" for claim in (support.unsupported_claims if support else ()))
    critique = unsupported or (support.reason if support else "support_non_evalue")
    return (
        "Révise la réponse afin que chaque affirmation soit directement étayée par les éléments. "
        "Supprime les affirmations non vérifiables; n'invente aucun fait.\n\n"
        f"Question:\n{query}\n\nÉléments:\n{context}\n\nRéponse précédente:\n{answer}\n\n"
        f"Critique:\n{critique}\n\nRéponse révisée:"
    )


def _format_context(documents: Iterable[Document]) -> str:
    blocks = []
    for index, document in enumerate(documents, start=1):
        label = document.doc_id or f"document-{index}"
        blocks.append(f"[{label}] {document.content}")
    return "\n\n".join(blocks)


def _claims(answer: str) -> list[str]:
    return [part.strip(" -\t\n") for part in re.split(r"(?<=[.!?])\s+|\n+", answer) if part.strip(" -\t\n")]


def _coverage(required: set[str], available: set[str]) -> float:
    if not required:
        return 1.0 if available else 0.0
    return len(required & available) / len(required)


def _content_terms(text: str) -> set[str]:
    stop_words = {
        "afin",
        "avec",
        "cette",
        "dans",
        "des",
        "elle",
        "est",
        "les",
        "pour",
        "que",
        "qui",
        "sur",
        "the",
        "and",
        "are",
        "for",
        "from",
        "that",
        "this",
        "with",
    }
    return {
        token
        for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]{3,}", text.lower())
        if token not in stop_words
    }
