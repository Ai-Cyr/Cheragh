"""Corrective Retrieval-Augmented Generation (CRAG).

Reference: https://arxiv.org/abs/2401.15884

The advanced path follows the three retrieval actions described by CRAG
(``correct``, ``ambiguous`` and ``incorrect``), can augment weak evidence with
an injected external retriever, and supports knowledge decomposition and
recomposition before generation.  The dependency-free lexical components are
deliberately baselines, not replicas of the paper's trained evaluator.

Existing construction remains valid: when no external retriever or knowledge
refiner is configured, query rewrite/retry and fallback behaviour is preserved.
Generation always consumes the exact document snapshots that were graded, so a
stateful retriever cannot silently return a different context on a second call.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
import numbers
import re
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

from ..base import (
    BaseRetriever,
    Document,
    ExtractiveLLMClient,
    LLMClient,
    _snapshot_document,
    _snapshot_documents,
    _validate_non_negative_int,
    _validate_top_k,
)
from ..engine import RAGEngine, RAGResponse
from ..query import MultiQueryTransformer, QueryTransformer
from ..tracing import RAGTrace


class RetrievalAction(str, Enum):
    """CRAG action selected from retrieval confidence."""

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class RetrievalGrade:
    """Quality assessment for retrieved context."""

    score: float
    passed: bool
    reason: str
    document_count: int = 0
    action: RetrievalAction | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "reason": self.reason,
            "document_count": self.document_count,
            "action": self.action.value if self.action is not None else None,
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

    @property
    def query(self) -> str:
        return self.response.query

    @property
    def retrieved_documents(self) -> list[Document]:
        return self.response.retrieved_documents

    @property
    def warnings(self) -> list[str]:
        return self.response.warnings

    @property
    def trace(self) -> RAGTrace | None:
        return self.response.trace

    def to_dict(self, *, include_prompt: bool = False) -> dict[str, Any]:
        data = self.response.to_dict(include_prompt=include_prompt)
        data["corrective"] = {"corrected": self.corrected, "attempts": self.attempts}
        return data


@runtime_checkable
class KnowledgeRefiner(Protocol):
    """Composable boundary for CRAG knowledge decomposition/recomposition."""

    def refine(self, query: str, documents: Sequence[Document]) -> list[Document]:
        """Return corrected document snapshots for final generation."""


class LexicalKnowledgeRefiner:
    """Sentence-level lexical decomposition/recomposition baseline.

    Sentences containing at least ``min_term_overlap`` distinct content terms
    from the query are retained in their original order.  This deterministic
    baseline makes the CRAG refinement boundary useful offline; an application
    can inject a learned or LLM-based :class:`KnowledgeRefiner` instead.
    """

    def __init__(self, min_term_overlap: int = 1, max_sentences_per_document: int | None = None):
        self.min_term_overlap = _validate_top_k(min_term_overlap, name="min_term_overlap")
        self.max_sentences_per_document = (
            None
            if max_sentences_per_document is None
            else _validate_top_k(max_sentences_per_document, name="max_sentences_per_document")
        )

    def decompose(self, document: Document) -> list[str]:
        """Split a document into non-empty sentence/passages."""

        if not isinstance(document, Document):
            raise TypeError("document must be a Document")
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", document.content) if part.strip()]

    def recompose(self, document: Document, passages: Sequence[str]) -> Document | None:
        """Recompose retained passages while preserving source provenance."""

        if not isinstance(document, Document):
            raise TypeError("document must be a Document")
        if isinstance(passages, (str, bytes)) or not isinstance(passages, Sequence):
            raise TypeError("passages must be a Sequence[str]")
        retained: list[str] = []
        for index, passage in enumerate(passages):
            if not isinstance(passage, str):
                raise TypeError(f"passages[{index}] must be a str")
            if passage.strip():
                retained.append(passage.strip())
        if not retained:
            return None

        refined = _snapshot_document(document)
        original_content = refined.content
        refined.content = " ".join(retained)
        provenance = _corrective_provenance(refined)
        provenance["refinement"] = {
            "strategy": self.__class__.__name__,
            "original_characters": len(original_content),
            "retained_characters": len(refined.content),
            "retained_passages": len(retained),
        }
        return refined

    def refine(self, query: str, documents: Sequence[Document]) -> list[Document]:
        query_terms = _content_terms(query)
        refined: list[Document] = []
        for source in _validated_snapshots(documents, name="documents"):
            passages = self.decompose(source)
            if query_terms:
                ranked = [
                    (index, passage, len(query_terms & _content_terms(passage)))
                    for index, passage in enumerate(passages)
                ]
                ranked = [item for item in ranked if item[2] >= self.min_term_overlap]
                if self.max_sentences_per_document is not None:
                    ranked = sorted(ranked, key=lambda item: (-item[2], item[0]))[
                        : self.max_sentences_per_document
                    ]
                passages = [item[1] for item in sorted(ranked, key=lambda item: item[0])]
            recomposed = self.recompose(source, passages)
            if recomposed is not None:
                refined.append(recomposed)
        return refined


class LexicalRetrievalGrader:
    """Dependency-free retrieval grader based on query/document token overlap."""

    def __init__(self, min_overlap: float = 0.08, incorrect_overlap: float = 0.0):
        self.min_overlap = _probability(min_overlap, name="min_overlap")
        self.incorrect_overlap = _probability(incorrect_overlap, name="incorrect_overlap")
        if self.incorrect_overlap > self.min_overlap:
            raise ValueError("incorrect_overlap must be <= min_overlap")

    def grade(self, query: str, documents: Iterable[Document]) -> RetrievalGrade:
        docs = list(documents)
        if not docs:
            return RetrievalGrade(
                score=0.0,
                passed=False,
                reason="no_documents",
                document_count=0,
                action=RetrievalAction.INCORRECT,
            )
        query_terms = _content_terms(query)
        if not query_terms:
            return RetrievalGrade(
                score=1.0,
                passed=True,
                reason="empty_query_terms",
                document_count=len(docs),
                action=RetrievalAction.CORRECT,
            )
        doc_terms: set[str] = set()
        for doc in docs:
            doc_terms.update(_content_terms(doc.content))
        overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
        best_score = max((doc.score or 0.0) for doc in docs)
        # Mix lexical overlap and retriever score without assuming score scale.
        normalized_score = max(overlap, min(max(best_score, 0.0), 1.0) * 0.5)
        passed = normalized_score >= self.min_overlap
        action = (
            RetrievalAction.CORRECT
            if passed
            else RetrievalAction.INCORRECT
            if normalized_score <= self.incorrect_overlap
            else RetrievalAction.AMBIGUOUS
        )
        return RetrievalGrade(
            score=float(normalized_score),
            passed=passed,
            reason="lexical_overlap" if passed else "low_overlap",
            document_count=len(docs),
            action=action,
        )


@dataclass(frozen=True)
class _CorrectionOutcome:
    query: str
    documents: tuple[Document, ...]
    initial_grade: RetrievalGrade
    final_grade: RetrievalGrade
    initial_document_ids: tuple[str | None, ...]
    external_document_ids: tuple[str | None, ...]
    external_attempted: bool
    refined: bool

    @property
    def usable(self) -> bool:
        return bool(self.documents) and self.final_grade.action is RetrievalAction.CORRECT

    def log(self, *, attempt: int, stage: str = "retrieval_correction") -> dict[str, Any]:
        return {
            "attempt": attempt,
            "stage": stage,
            "query": self.query,
            "action": self.initial_grade.action.value if self.initial_grade.action is not None else None,
            "retrieval_grade": self.initial_grade.to_dict(),
            "post_correction_grade": self.final_grade.to_dict(),
            "initial_document_ids": list(self.initial_document_ids),
            "external_document_ids": list(self.external_document_ids),
            "external_retrieval_attempted": self.external_attempted,
            "corrected_document_ids": [document.doc_id for document in self.documents],
            "refined": self.refined,
        }


class _SnapshotRetriever(BaseRetriever):
    """One-request retriever preventing generation-time re-retrieval."""

    def __init__(self, documents: Sequence[Document]):
        self._documents = _validated_snapshots(documents, name="documents")

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        top_k = _validate_top_k(top_k)
        return _snapshot_documents(self._documents[:top_k])


class CorrectiveRAGEngine:
    """Wrap a RAG engine with graded retrieval and bounded CRAG correction.

    ``external_retriever`` is queried only for ``ambiguous`` or ``incorrect``
    actions.  ``knowledge_refiner`` may be ``"lexical"``, an object exposing
    ``refine(query, documents)``, or a callable with that signature.  All
    collaborator boundaries receive snapshots and all returned documents are
    validated and copied before use.
    """

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
        external_retriever: BaseRetriever | None = None,
        external_top_k: int | None = None,
        knowledge_refiner: KnowledgeRefiner | Callable[[str, Sequence[Document]], list[Document]] | str | None = None,
        incorrect_context_score: float = 0.0,
    ):
        max_retries = _validate_non_negative_int(max_retries, name="max_retries")
        min_context_score = _probability(min_context_score, name="min_context_score")
        incorrect_context_score = _probability(incorrect_context_score, name="incorrect_context_score")
        if incorrect_context_score > min_context_score:
            raise ValueError("incorrect_context_score must be <= min_context_score")
        if min_grounded_score is not None:
            min_grounded_score = _probability(min_grounded_score, name="min_grounded_score")
        if external_top_k is not None:
            external_top_k = _validate_top_k(external_top_k, name="external_top_k")
        if not isinstance(fallback_answer, str):
            raise TypeError("fallback_answer must be a str")
        if not isinstance(return_details, bool):
            raise TypeError("return_details must be a bool")
        if retrieval_grader is not None and not (
            callable(retrieval_grader) or callable(getattr(retrieval_grader, "grade", None))
        ):
            raise TypeError("retrieval_grader must expose grade(query, documents), be callable, or be None")
        if query_rewriter is not None and not (
            callable(query_rewriter) or callable(getattr(query_rewriter, "transform", None))
        ):
            raise TypeError("query_rewriter must expose transform(query), be callable, or be None")
        if base_engine is None and retriever is None:
            raise ValueError("Provide either base_engine or retriever")
        if base_engine is not None and retriever is not None and retriever is not base_engine.retriever:
            raise ValueError("retriever conflicts with base_engine.retriever")
        if base_engine is not None and llm_client is not None and llm_client is not base_engine.llm_client:
            raise ValueError("llm_client conflicts with base_engine.llm_client")
        if base_engine is None:
            assert retriever is not None  # narrowed by the validation above
            base_engine = RAGEngine(
                retriever=retriever,
                llm_client=llm_client or ExtractiveLLMClient(),
                strict_grounding=True,
            )
        if external_retriever is not None and not callable(getattr(external_retriever, "retrieve", None)):
            raise TypeError("external_retriever must expose retrieve(query, top_k)")
        if isinstance(knowledge_refiner, str):
            if knowledge_refiner.lower().replace("_", "-") not in {"lexical", "sentence", "sentence-lexical"}:
                raise ValueError("knowledge_refiner string must be 'lexical'")
            knowledge_refiner = LexicalKnowledgeRefiner()
        if knowledge_refiner is not None and not (
            callable(knowledge_refiner) or callable(getattr(knowledge_refiner, "refine", None))
        ):
            raise TypeError("knowledge_refiner must expose refine(query, documents), be callable, or be None")
        self.base_engine = base_engine
        self.retriever = base_engine.retriever if retriever is None else retriever
        self.llm_client = base_engine.llm_client if llm_client is None else llm_client
        self.retrieval_grader = (
            LexicalRetrievalGrader(
                min_overlap=min_context_score,
                incorrect_overlap=incorrect_context_score,
            )
            if retrieval_grader is None
            else retrieval_grader
        )
        self.query_rewriter = (
            MultiQueryTransformer(num_queries=max(2, max_retries + 1))
            if query_rewriter is None
            else query_rewriter
        )
        self.max_retries = max_retries
        self.min_context_score = min_context_score
        self.incorrect_context_score = incorrect_context_score
        self.min_grounded_score = min_grounded_score
        self.fallback_answer = fallback_answer
        self.return_details = return_details
        self.external_retriever = external_retriever
        self.external_top_k = external_top_k
        self.knowledge_refiner = knowledge_refiner

    def ask(self, query: str, top_k: int | None = None, **kwargs: Any) -> RAGResponse | CorrectiveRAGResult:
        query = _validate_user_query(query)
        effective_top_k = self.base_engine.top_k if top_k is None else _validate_top_k(top_k)
        attempts: list[dict[str, Any]] = []
        candidate_queries = self._candidate_queries(query)
        candidate_queries = candidate_queries[: max(1, self.max_retries + 1)]
        outcomes: dict[str, _CorrectionOutcome] = {}
        best: _CorrectionOutcome | None = None

        for candidate in candidate_queries:
            outcome = self._correct_candidate(candidate, effective_top_k)
            outcomes[candidate] = outcome
            attempts.append(outcome.log(attempt=len(attempts) + 1))
            if best is None or outcome.final_grade.score > best.final_grade.score:
                best = outcome
            if outcome.usable:
                best = outcome
                break

        assert best is not None  # candidate_queries always contains the original query
        if not best.usable:
            attempt_snapshot = deepcopy(attempts)
            response = RAGResponse(
                query=query,
                answer=self.fallback_answer,
                sources=[],
                retrieved_documents=[],
                prompt="",
                metadata={
                    "corrective": True,
                    "retrieval_grade": best.initial_grade.to_dict(),
                    "post_correction_grade": best.final_grade.to_dict(),
                    "retrieval_action": best.initial_grade.action.value,
                    "external_document_ids": list(best.external_document_ids),
                    "external_retrieval_attempted": best.external_attempted,
                    "knowledge_refined": best.refined,
                    "attempts": attempt_snapshot,
                    "failed_stage": "retrieval",
                },
                warnings=["corrective_low_context"],
                grounded_score=0.0,
            )
            result = CorrectiveRAGResult(response=response, attempts=deepcopy(attempts), corrected=True)
            return result if self.return_details else response

        selected = best
        response = self._generate_from_documents(selected.query, selected.documents, effective_top_k, kwargs)
        for attempt in reversed(attempts):
            if attempt["query"] == selected.query:
                attempt["grounded_score"] = response.grounded_score
                break

        if (
            self.min_grounded_score is not None
            and response.grounded_score < self.min_grounded_score
            and len(candidate_queries) > 1
        ):
            for candidate in candidate_queries:
                if candidate == selected.query:
                    continue
                outcome = outcomes.get(candidate)
                if outcome is None:
                    outcome = self._correct_candidate(candidate, effective_top_k)
                    outcomes[candidate] = outcome
                    attempts.append(outcome.log(attempt=len(attempts) + 1, stage="answer_grounding_retry"))
                if not outcome.usable:
                    continue
                retry_response = self._generate_from_documents(candidate, outcome.documents, effective_top_k, kwargs)
                attempts[-1]["grounded_score"] = retry_response.grounded_score
                if retry_response.grounded_score > response.grounded_score:
                    response = retry_response
                    selected = outcome
                if response.grounded_score >= self.min_grounded_score:
                    break

        corrected = (
            selected.query != query
            or selected.initial_grade.action is not RetrievalAction.CORRECT
            or bool(selected.external_document_ids)
            or selected.refined
        )
        response.metadata.setdefault("corrective", True)
        response.metadata["original_query"] = query
        response.metadata["selected_query"] = selected.query
        response.metadata["retrieval_action"] = selected.initial_grade.action.value
        response.metadata["retrieval_grade"] = selected.initial_grade.to_dict()
        response.metadata["post_correction_grade"] = selected.final_grade.to_dict()
        response.metadata["corrected_document_ids"] = [document.doc_id for document in selected.documents]
        response.metadata["generation_document_ids"] = [
            document.doc_id for document in response.retrieved_documents
        ]
        response.metadata["external_document_ids"] = list(selected.external_document_ids)
        response.metadata["external_retrieval_attempted"] = selected.external_attempted
        response.metadata["knowledge_refined"] = selected.refined
        response.metadata["attempts"] = deepcopy(attempts)

        # Query rewriting is an internal corrective detail. The public response
        # continues to represent the request made by the caller.
        response.query = query
        if response.trace is not None:
            response.trace.query = query
            response.trace.metadata["selected_query"] = selected.query
            response.trace.metadata["corrective_action"] = selected.initial_grade.action.value
            response.trace.metadata["corrected_document_ids"] = [document.doc_id for document in selected.documents]
            response.trace.warnings.extend(
                warning for warning in response.warnings if warning not in response.trace.warnings
            )
        result = CorrectiveRAGResult(response=response, attempts=deepcopy(attempts), corrected=corrected)
        return result if self.return_details else response

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
        if not isinstance(variants, Iterable):
            raise TypeError("query_rewriter must return a string or iterable of strings")
        output: list[str] = [query]
        seen = {query.casefold()}
        inspected = 0
        inspection_budget = max(4, (self.max_retries + 1) * 4)
        for variant in variants:
            inspected += 1
            if not isinstance(variant, str):
                raise TypeError("query_rewriter variants must be strings")
            normalized = _validate_user_query(variant, name="query_rewriter variant")
            if normalized.casefold() not in seen:
                output.append(normalized)
                seen.add(normalized.casefold())
            if len(output) >= max(1, self.max_retries + 1):
                break
            if inspected >= inspection_budget:
                break
        return output

    def _correct_candidate(self, query: str, top_k: int) -> _CorrectionOutcome:
        primary = self._retrieve(self.retriever, query, top_k, origin="primary", trigger_action=None)
        initial_grade = self._grade(query, primary)
        action = initial_grade.action
        assert action is not None
        for document in primary:
            provenance = _corrective_provenance(document)
            provenance["evaluation"] = {
                "action": action.value,
                "score": initial_grade.score,
                "reason": initial_grade.reason,
            }

        external: list[Document] = []
        if action in {RetrievalAction.AMBIGUOUS, RetrievalAction.INCORRECT} and self.external_retriever is not None:
            external_k = self.external_top_k or top_k
            external = self._retrieve(
                self.external_retriever,
                query,
                external_k,
                origin="external",
                trigger_action=action,
            )

        if action is RetrievalAction.CORRECT:
            corrected = primary
        elif action is RetrievalAction.AMBIGUOUS:
            # The corrective source goes first so it is not silently discarded
            # when a caller requests a one-document context.
            corrected = _interleave_unique(external, primary, limit=top_k)
        else:
            corrected = external[:top_k]

        refined = self.knowledge_refiner is not None and bool(corrected)
        if refined:
            corrected = self._refine(query, corrected, limit=top_k)
        else:
            corrected = _snapshot_documents(corrected[:top_k])

        changed = refined or bool(external)
        final_grade = self._grade(query, corrected) if changed else initial_grade
        for document in corrected:
            provenance = _corrective_provenance(document)
            provenance["post_correction"] = {
                "action": final_grade.action.value if final_grade.action is not None else None,
                "score": final_grade.score,
            }
        return _CorrectionOutcome(
            query=query,
            documents=tuple(_snapshot_documents(corrected)),
            initial_grade=initial_grade,
            final_grade=final_grade,
            initial_document_ids=tuple(document.doc_id for document in primary),
            external_document_ids=tuple(document.doc_id for document in external),
            external_attempted=(
                action in {RetrievalAction.AMBIGUOUS, RetrievalAction.INCORRECT}
                and self.external_retriever is not None
            ),
            refined=refined,
        )

    def _retrieve(
        self,
        retriever: BaseRetriever,
        query: str,
        top_k: int,
        *,
        origin: str,
        trigger_action: RetrievalAction | None,
    ) -> list[Document]:
        raw = retriever.retrieve(query, top_k=top_k)
        documents = _validated_snapshots(
            raw,
            name=f"{origin}_retriever results",
            limit=top_k,
        )
        for document in documents:
            provenance = _corrective_provenance(document)
            original_doc_id = document.doc_id
            if not document.doc_id or not document.doc_id.strip():
                document.doc_id = _anonymous_document_id(document)
                provenance["synthetic_doc_id"] = True
                provenance["original_doc_id"] = original_doc_id
            provenance.update(
                {
                    "origin": origin,
                    "retrieval_query": query,
                    "trigger_action": trigger_action.value if trigger_action is not None else None,
                }
            )
        return documents

    def _refine(self, query: str, documents: Sequence[Document], *, limit: int) -> list[Document]:
        refiner = self.knowledge_refiner
        snapshots = _validated_snapshots(documents, name="documents")
        if refiner is None:
            return snapshots
        if callable(getattr(refiner, "refine", None)):
            output = refiner.refine(query, _snapshot_documents(snapshots))
        elif callable(refiner):
            output = refiner(query, _snapshot_documents(snapshots))
        else:  # pragma: no cover - constructor validation guards this path
            raise TypeError("knowledge_refiner must be callable")
        refined = _validated_snapshots(output, name="knowledge_refiner result", limit=limit)
        source_provenance = {
            document.doc_id: deepcopy(document.metadata.get("corrective_provenance"))
            for document in snapshots
            if document.doc_id is not None and document.metadata.get("corrective_provenance") is not None
        }
        for index, document in enumerate(refined):
            inherited = source_provenance.get(document.doc_id) if document.doc_id is not None else None
            if inherited is None and index < len(snapshots):
                inherited = snapshots[index].metadata.get("corrective_provenance")
            if "corrective_provenance" not in document.metadata and inherited is not None:
                document.metadata["corrective_provenance"] = deepcopy(inherited)
            provenance = _corrective_provenance(document)
            original_doc_id = document.doc_id
            if not document.doc_id or not document.doc_id.strip():
                document.doc_id = _anonymous_document_id(document)
                provenance["synthetic_doc_id"] = True
                provenance["original_doc_id"] = original_doc_id
            provenance.setdefault("refinement", {"strategy": refiner.__class__.__name__})
        return refined

    def _generate_from_documents(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: int,
        generate_kwargs: dict[str, Any],
    ) -> RAGResponse:
        engine = RAGEngine(
            retriever=_SnapshotRetriever(documents),
            llm_client=self.llm_client,
            answer_prompt=self.base_engine.answer_prompt,
            top_k=top_k,
            strict_grounding=self.base_engine.strict_grounding,
            min_score=self.base_engine.min_score,
            require_citations=self.base_engine.require_citations,
            flag_unsourced_sentences=self.base_engine.flag_unsourced_sentences,
            compressor=self.base_engine.compressor,
            query_transformer=None,
            trace_enabled=self.base_engine.trace_enabled,
            cache_backend=self.base_engine.cache_backend,
            cache_config=self.base_engine.cache_config,
            trace_export_path=self.base_engine.trace_export_path,
            trace_include_prompt=self.base_engine.trace_include_prompt,
            trace_pricing=self.base_engine.trace_pricing,
        )
        return engine.ask(query, top_k=top_k, **generate_kwargs)

    def _grade(self, query: str, documents: Sequence[Document]) -> RetrievalGrade:
        grader = self.retrieval_grader
        snapshots = _validated_snapshots(documents, name="documents")
        if hasattr(grader, "grade"):
            grade = grader.grade(query, _snapshot_documents(snapshots))
            if isinstance(grade, RetrievalGrade):
                return self._normalize_grade(grade, len(snapshots))
            if isinstance(grade, dict):
                raw_action = grade.get("action")
                parsed_action = _retrieval_action(raw_action) if raw_action is not None else None
                default_passed = parsed_action is RetrievalAction.CORRECT
                return self._normalize_grade(
                    RetrievalGrade(
                        score=grade.get("score", 0.0),  # type: ignore[arg-type]
                        passed=grade.get("passed", default_passed),  # type: ignore[arg-type]
                        reason=grade.get("reason", "custom"),  # type: ignore[arg-type]
                        document_count=grade.get("document_count", len(snapshots)),  # type: ignore[arg-type]
                        action=parsed_action,
                    ),
                    len(snapshots),
                )
        score = grader(query, _snapshot_documents(snapshots)) if callable(grader) else 0.0
        return self._normalize_grade(
            RetrievalGrade(
                score=score,  # type: ignore[arg-type]
                passed=(
                    isinstance(score, numbers.Real)
                    and not isinstance(score, bool)
                    and float(score) >= self.min_context_score
                ),
                reason="custom",
                document_count=len(snapshots),
            ),
            len(snapshots),
        )

    def _normalize_grade(self, grade: RetrievalGrade, actual_document_count: int) -> RetrievalGrade:
        if isinstance(grade.score, bool) or not isinstance(grade.score, numbers.Real):
            raise TypeError("retrieval grade score must be a real number")
        if not isinstance(grade.passed, bool):
            raise TypeError("retrieval grade passed must be a bool")
        if not isinstance(grade.reason, str):
            raise TypeError("retrieval grade reason must be a str")
        score = float(grade.score)
        if not math.isfinite(score):
            raise ValueError("retrieval grade score must be finite")
        if isinstance(grade.document_count, bool) or not isinstance(grade.document_count, int):
            raise TypeError("retrieval grade document_count must be an int")
        if grade.document_count < 0:
            raise ValueError("retrieval grade document_count must be >= 0")
        action = grade.action
        if action is not None:
            action = _retrieval_action(action)
        elif grade.passed:
            action = RetrievalAction.CORRECT
        elif score <= self.incorrect_context_score:
            action = RetrievalAction.INCORRECT
        else:
            action = RetrievalAction.AMBIGUOUS
        return RetrievalGrade(
            score=score,
            passed=action is RetrievalAction.CORRECT,
            reason=str(grade.reason),
            document_count=actual_document_count,
            action=action,
        )


def _content_terms(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "what", "which", "who", "how",
        "est", "une", "des", "les", "dans", "pour", "que", "qui", "quoi", "comment",
        "quelle", "quel", "quels", "quelles", "sur", "avec", "aux", "du", "de", "la", "le",
    }
    return {token for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]{3,}", text.lower()) if token not in stop}


def _validate_user_query(value: object, *, name: str = "query") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return normalized


def _retrieval_action(value: object) -> RetrievalAction:
    if isinstance(value, RetrievalAction):
        return value
    if not isinstance(value, str):
        raise TypeError("retrieval grade action must be a string or RetrievalAction")
    try:
        return RetrievalAction(value.strip().lower())
    except ValueError as exc:
        raise ValueError("retrieval grade action must be correct, ambiguous, or incorrect") from exc


def _validated_snapshots(
    documents: Iterable[Document],
    *,
    name: str,
    limit: int | None = None,
) -> list[Document]:
    if isinstance(documents, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of Document")
    if limit is not None:
        limit = _validate_top_k(limit, name="limit")
    try:
        iterator = iter(documents)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of Document") from exc
    snapshots: list[Document] = []
    index = 0
    while limit is None or len(snapshots) < limit:
        try:
            document = next(iterator)
        except StopIteration:
            break
        if not isinstance(document, Document):
            raise TypeError(f"{name}[{index}] must be a Document")
        if not isinstance(document.content, str):
            raise TypeError(f"{name}[{index}].content must be a str")
        if not document.content.strip():
            raise ValueError(f"{name}[{index}].content must be non-empty")
        if not isinstance(document.metadata, dict):
            raise TypeError(f"{name}[{index}].metadata must be a dict")
        if document.doc_id is not None and not isinstance(document.doc_id, str):
            raise TypeError(f"{name}[{index}].doc_id must be a str or None")
        if document.score is not None:
            if isinstance(document.score, bool) or not isinstance(document.score, numbers.Real):
                raise TypeError(f"{name}[{index}].score must be a real number or None")
            if not math.isfinite(float(document.score)):
                raise ValueError(f"{name}[{index}].score must be finite")
        snapshots.append(_snapshot_document(document))
        index += 1
    return snapshots


def _corrective_provenance(document: Document) -> dict[str, Any]:
    existing = document.metadata.get("corrective_provenance")
    if isinstance(existing, dict):
        provenance = deepcopy(existing)
    elif existing is None:
        provenance = {}
    else:
        provenance = {"previous_value": deepcopy(existing)}
    document.metadata["corrective_provenance"] = provenance
    return provenance


def _interleave_unique(
    primary: Sequence[Document],
    external: Sequence[Document],
    *,
    limit: int,
) -> list[Document]:
    """Round-robin ambiguous local and external evidence without duplicates."""

    limit = _validate_top_k(limit, name="limit")
    merged: list[Document] = []
    seen: set[tuple[str, str]] = set()
    for index in range(max(len(primary), len(external))):
        for collection in (primary, external):
            if index >= len(collection):
                continue
            document = collection[index]
            key = (
                "doc_id",
                document.doc_id,
            ) if document.doc_id and document.doc_id.strip() else (
                "content",
                " ".join(document.content.split()).casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(_snapshot_document(document))
            if len(merged) >= limit:
                return merged
    return merged


def _anonymous_document_id(document: Document) -> str:
    normalized = " ".join(document.content.split()).casefold()
    digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=10).hexdigest()
    return f"crag-anonymous-{digest}"
