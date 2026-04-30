"""Corrective RAG engine.

This module implements a lightweight Corrective-RAG/Self-RAG pattern:
retrieve, grade context quality, optionally rewrite and retry, generate, then
optionally retry once more when grounding looks weak.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Iterable

from ..base import BaseRetriever, Document, LLMClient, ExtractiveLLMClient
from ..engine import RAGEngine, RAGResponse
from ..query import MultiQueryTransformer, QueryTransformer


@dataclass
class RetrievalGrade:
    """Quality assessment for retrieved context."""

    score: float
    passed: bool
    reason: str
    document_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "reason": self.reason,
            "document_count": self.document_count,
        }


@dataclass
class CorrectiveRAGResult:
    """Detailed result from :class:`CorrectiveRAGEngine`."""

    response: RAGResponse
    attempts: list[dict[str, Any]] = field(default_factory=list)
    corrected: bool = False

    @property
    def answer(self) -> str:
        return self.response.answer

    @property
    def sources(self):
        return self.response.sources

    @property
    def metadata(self) -> dict[str, Any]:
        return self.response.metadata

    def to_dict(self) -> dict[str, Any]:
        data = self.response.to_dict()
        data["corrective"] = {"corrected": self.corrected, "attempts": self.attempts}
        return data


class LexicalRetrievalGrader:
    """Dependency-free retrieval grader based on query/document token overlap."""

    def __init__(self, min_overlap: float = 0.08):
        self.min_overlap = min_overlap

    def grade(self, query: str, documents: Iterable[Document]) -> RetrievalGrade:
        docs = list(documents)
        if not docs:
            return RetrievalGrade(score=0.0, passed=False, reason="no_documents", document_count=0)
        query_terms = _content_terms(query)
        if not query_terms:
            return RetrievalGrade(score=1.0, passed=True, reason="empty_query_terms", document_count=len(docs))
        doc_terms: set[str] = set()
        for doc in docs:
            doc_terms.update(_content_terms(doc.content))
        overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
        best_score = max((doc.score or 0.0) for doc in docs)
        # Mix lexical overlap and retriever score without assuming score scale.
        normalized_score = max(overlap, min(max(best_score, 0.0), 1.0) * 0.5)
        passed = normalized_score >= self.min_overlap
        return RetrievalGrade(
            score=float(normalized_score),
            passed=passed,
            reason="lexical_overlap" if passed else "low_overlap",
            document_count=len(docs),
        )


class CorrectiveRAGEngine:
    """Wrap a RAG engine with retrieval grading and retry/correction logic."""

    def __init__(
        self,
        base_engine: RAGEngine | None = None,
        retriever: BaseRetriever | None = None,
        llm_client: LLMClient | None = None,
        retrieval_grader: LexicalRetrievalGrader | Any | None = None,
        query_rewriter: QueryTransformer | Callable[[str], list[str] | str] | None = None,
        max_retries: int = 2,
        min_context_score: float = 0.12,
        min_grounded_score: float | None = None,
        fallback_answer: str = "Je ne sais pas : le contexte disponible n'est pas suffisamment fiable.",
        return_details: bool = False,
    ):
        if base_engine is None and retriever is None:
            raise ValueError("Provide either base_engine or retriever")
        if base_engine is None:
            base_engine = RAGEngine(retriever=retriever, llm_client=llm_client or ExtractiveLLMClient(), strict_grounding=True)
        self.base_engine = base_engine
        self.retriever = retriever or base_engine.retriever
        self.llm_client = llm_client or base_engine.llm_client
        self.retrieval_grader = retrieval_grader or LexicalRetrievalGrader(min_overlap=min_context_score)
        self.query_rewriter = query_rewriter or MultiQueryTransformer(num_queries=max(2, max_retries + 1))
        self.max_retries = max(0, max_retries)
        self.min_context_score = min_context_score
        self.min_grounded_score = min_grounded_score
        self.fallback_answer = fallback_answer
        self.return_details = return_details

    def ask(self, query: str, top_k: int | None = None, **kwargs: Any) -> RAGResponse | CorrectiveRAGResult:
        attempts: list[dict[str, Any]] = []
        candidate_queries = self._candidate_queries(query)
        candidate_queries = candidate_queries[: max(1, self.max_retries + 1)]
        best_query = candidate_queries[0]
        best_grade = RetrievalGrade(0.0, False, "not_evaluated", 0)

        for idx, candidate in enumerate(candidate_queries):
            docs = self.retriever.retrieve(candidate, top_k=top_k or self.base_engine.top_k)
            grade = self._grade(candidate, docs)
            attempts.append({"attempt": idx + 1, "query": candidate, "retrieval_grade": grade.to_dict()})
            if grade.score > best_grade.score:
                best_query, best_grade = candidate, grade
            if grade.passed:
                best_query, best_grade = candidate, grade
                break

        if not best_grade.passed:
            response = RAGResponse(
                query=query,
                answer=self.fallback_answer,
                sources=[],
                retrieved_documents=[],
                prompt="",
                metadata={
                    "corrective": True,
                    "retrieval_grade": best_grade.to_dict(),
                    "attempts": attempts,
                    "failed_stage": "retrieval",
                },
                warnings=["corrective_low_context"],
                grounded_score=0.0,
            )
            return CorrectiveRAGResult(response=response, attempts=attempts, corrected=True) if self.return_details else response

        response = self.base_engine.ask(best_query, top_k=top_k, **kwargs)
        response.metadata.setdefault("corrective", True)
        response.metadata["original_query"] = query
        response.metadata["selected_query"] = best_query
        response.metadata["retrieval_grade"] = best_grade.to_dict()
        response.metadata["attempts"] = attempts
        corrected = best_query != query

        if self.min_grounded_score is not None and response.grounded_score < self.min_grounded_score and len(candidate_queries) > 1:
            for candidate in candidate_queries:
                if candidate == best_query:
                    continue
                retry_response = self.base_engine.ask(candidate, top_k=top_k, **kwargs)
                attempts.append(
                    {
                        "attempt": len(attempts) + 1,
                        "query": candidate,
                        "grounded_score": retry_response.grounded_score,
                        "stage": "answer_grounding_retry",
                    }
                )
                if retry_response.grounded_score > response.grounded_score:
                    response = retry_response
                    response.metadata.setdefault("corrective", True)
                    response.metadata["original_query"] = query
                    response.metadata["selected_query"] = candidate
                    response.metadata["attempts"] = attempts
                    corrected = True
                if response.grounded_score >= self.min_grounded_score:
                    break

        if response.trace is not None:
            response.trace.warnings.extend([warning for warning in response.warnings if warning not in response.trace.warnings])
        return CorrectiveRAGResult(response=response, attempts=attempts, corrected=corrected) if self.return_details else response

    def run(self, query: str, **kwargs: Any):
        return self.ask(query, **kwargs)

    def _candidate_queries(self, query: str) -> list[str]:
        if self.query_rewriter is None:
            return [query]
        if hasattr(self.query_rewriter, "transform"):
            variants = self.query_rewriter.transform(query)
        else:
            variants = self.query_rewriter(query)
        if isinstance(variants, str):
            variants = [variants]
        output: list[str] = []
        for variant in [query, *list(variants)]:
            if variant and variant not in output:
                output.append(str(variant))
        return output or [query]

    def _grade(self, query: str, documents: list[Document]) -> RetrievalGrade:
        grader = self.retrieval_grader
        if hasattr(grader, "grade"):
            grade = grader.grade(query, documents)
            if isinstance(grade, RetrievalGrade):
                return grade
            if isinstance(grade, dict):
                return RetrievalGrade(
                    score=float(grade.get("score", 0.0)),
                    passed=bool(grade.get("passed", False)),
                    reason=str(grade.get("reason", "custom")),
                    document_count=int(grade.get("document_count", len(documents))),
                )
        score = float(grader(query, documents)) if callable(grader) else 0.0
        return RetrievalGrade(score=score, passed=score >= self.min_context_score, reason="custom", document_count=len(documents))


def _content_terms(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "what", "which", "who", "how",
        "est", "une", "des", "les", "dans", "pour", "que", "qui", "quoi", "comment",
        "quelle", "quel", "quels", "quelles", "sur", "avec", "aux", "du", "de", "la", "le",
    }
    return {token for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]{3,}", text.lower()) if token not in stop}
