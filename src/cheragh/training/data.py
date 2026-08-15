"""Retrieval-aware training data and orchestration primitives.

Cheragh does not bundle a deep-learning framework.  This module owns the
framework-neutral part of retriever and RAG adaptation: defensive training
examples, hard-negative mining, teacher-score distillation, a reference
contrastive loss, RAFT-style open-book records, and an injectable trainer
boundary.  Applications remain responsible for model weights, optimizers and
distributed training.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import math
import random
from typing import Any, Protocol, runtime_checkable

from ..base import BaseRetriever, Document, _numpy, _snapshot_document, _validate_top_k


def _document_key(document: Document) -> str:
    if document.doc_id:
        return document.doc_id
    digest = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
    return f"content::{digest}"


def _snapshots(documents: Iterable[Document]) -> tuple[Document, ...]:
    return tuple(_snapshot_document(document) for document in documents)


@dataclass(frozen=True)
class RetrievalTrainingExample:
    """One query with positive evidence and mined or curated negatives."""

    query: str
    positive_documents: tuple[Document, ...]
    negative_documents: tuple[Document, ...] = ()
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("Training query must be a non-empty string")
        if self.answer is not None and not isinstance(self.answer, str):
            raise TypeError("Training answer must be a string or None")
        positives = _snapshots(self.positive_documents)
        negatives = _snapshots(self.negative_documents)
        if not positives:
            raise ValueError("A retrieval training example requires at least one positive document")
        positive_keys = {_document_key(document) for document in positives}
        if len(positive_keys) != len(positives):
            raise ValueError("Positive training documents must be unique")
        negative_keys = {_document_key(document) for document in negatives}
        if len(negative_keys) != len(negatives):
            raise ValueError("Negative training documents must be unique")
        overlap = positive_keys & negative_keys
        if overlap:
            raise ValueError(f"Documents cannot be both positive and negative: {sorted(overlap)}")
        object.__setattr__(self, "positive_documents", positives)
        object.__setattr__(self, "negative_documents", negatives)
        object.__setattr__(self, "metadata", deepcopy(self.metadata or {}))

    @property
    def positive_doc_ids(self) -> tuple[str, ...]:
        return tuple(_document_key(document) for document in self.positive_documents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "positive_documents": [_serialize_document(document) for document in self.positive_documents],
            "negative_documents": [_serialize_document(document) for document in self.negative_documents],
            "answer": self.answer,
            "metadata": deepcopy(self.metadata),
        }


class HardNegativeMiner:
    """Mine highly ranked non-positive documents from an existing retriever."""

    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        candidate_top_k: int = 50,
        negatives_per_query: int = 4,
        exclusion_filter: Callable[[str, Document], bool] | None = None,
    ):
        self.retriever = retriever
        self.candidate_top_k = _validate_top_k(candidate_top_k, name="candidate_top_k")
        self.negatives_per_query = _validate_top_k(negatives_per_query, name="negatives_per_query")
        if self.candidate_top_k < self.negatives_per_query:
            raise ValueError("candidate_top_k must be >= negatives_per_query")
        self.exclusion_filter = exclusion_filter

    def mine(
        self,
        query: str,
        positive_documents: Sequence[Document],
    ) -> RetrievalTrainingExample:
        positives = _snapshots(positive_documents)
        if not positives:
            raise ValueError("Hard-negative mining requires positive documents")
        positive_keys = {_document_key(document) for document in positives}
        candidates = self.retriever.retrieve(query, top_k=self.candidate_top_k)
        negatives: list[Document] = []
        seen = set(positive_keys)
        for candidate in candidates:
            key = _document_key(candidate)
            if key in seen:
                continue
            if self.exclusion_filter is not None and self.exclusion_filter(query, candidate):
                continue
            negatives.append(_snapshot_document(candidate))
            seen.add(key)
            if len(negatives) >= self.negatives_per_query:
                break
        return RetrievalTrainingExample(
            query=query,
            positive_documents=positives,
            negative_documents=tuple(negatives),
            metadata={
                "mining_method": self.retriever.__class__.__name__,
                "candidate_top_k": self.candidate_top_k,
                "requested_negatives": self.negatives_per_query,
                "mined_negatives": len(negatives),
            },
        )


@dataclass(frozen=True)
class DistilledRetrievalExample:
    """Training example with normalized teacher relevance probabilities."""

    example: RetrievalTrainingExample
    document_probabilities: tuple[float, ...]
    teacher_scores: tuple[float, ...]
    temperature: float

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise TypeError("Distillation temperature must be a number")
        if not math.isfinite(float(self.temperature)) or self.temperature <= 0:
            raise ValueError("Distillation temperature must be finite and > 0")
        expected = len(self.example.positive_documents) + len(self.example.negative_documents)
        if len(self.document_probabilities) != expected or len(self.teacher_scores) != expected:
            raise ValueError("Distillation scores must align with all positive and negative documents")
        if any(
            not math.isfinite(float(probability)) or probability < 0
            for probability in self.document_probabilities
        ):
            raise ValueError("Distillation probabilities must be finite and non-negative")
        if any(not math.isfinite(float(score)) for score in self.teacher_scores):
            raise ValueError("Teacher scores must be finite")
        if expected and not math.isclose(sum(self.document_probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Distillation probabilities must sum to one")
        example_snapshot = RetrievalTrainingExample(
            query=self.example.query,
            positive_documents=self.example.positive_documents,
            negative_documents=self.example.negative_documents,
            answer=self.example.answer,
            metadata=self.example.metadata,
        )
        object.__setattr__(self, "example", example_snapshot)
        object.__setattr__(self, "document_probabilities", tuple(float(value) for value in self.document_probabilities))
        object.__setattr__(self, "teacher_scores", tuple(float(value) for value in self.teacher_scores))
        object.__setattr__(self, "temperature", float(self.temperature))


class TeacherScoreDistiller:
    """Convert arbitrary teacher scores into temperature-scaled soft labels."""

    def __init__(
        self,
        scorer: Callable[[str, Sequence[Document]], Sequence[float]],
        *,
        temperature: float = 1.0,
    ):
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be a finite number > 0")
        self.scorer = scorer
        self.temperature = float(temperature)

    def distill(self, example: RetrievalTrainingExample) -> DistilledRetrievalExample:
        documents = (*example.positive_documents, *example.negative_documents)
        scorer_documents = _snapshots(documents)
        raw_scores = tuple(float(score) for score in self.scorer(example.query, scorer_documents))
        if len(raw_scores) != len(documents):
            raise ValueError("Teacher scorer must return one score per document")
        if any(not math.isfinite(score) for score in raw_scores):
            raise ValueError("Teacher scores must be finite")
        probabilities = _softmax(raw_scores, temperature=self.temperature)
        return DistilledRetrievalExample(example, probabilities, raw_scores, self.temperature)


def contrastive_retrieval_loss(
    query_embeddings: Any,
    positive_embeddings: Any,
    negative_embeddings: Any,
    *,
    temperature: float = 1.0,
) -> float:
    """Return mean InfoNCE loss for aligned positives and per-query negatives."""

    np = _numpy()
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be a finite number > 0")
    queries = np.asarray(query_embeddings, dtype=float)
    positives = np.asarray(positive_embeddings, dtype=float)
    negatives = np.asarray(negative_embeddings, dtype=float)
    if queries.ndim != 2 or positives.shape != queries.shape:
        raise ValueError("query and positive embeddings must share shape (batch, dimension)")
    if queries.shape[0] == 0:
        raise ValueError("contrastive loss requires a non-empty batch")
    if queries.shape[1] == 0:
        raise ValueError("contrastive embeddings require a non-empty dimension")
    if negatives.ndim != 3 or negatives.shape[0] != queries.shape[0] or negatives.shape[2] != queries.shape[1]:
        raise ValueError("negative embeddings must have shape (batch, negatives, dimension)")
    if negatives.shape[1] == 0:
        raise ValueError("contrastive loss requires at least one negative per query")
    if not np.isfinite(queries).all() or not np.isfinite(positives).all() or not np.isfinite(negatives).all():
        raise ValueError("contrastive embeddings must contain only finite values")
    positive_scores = np.sum(queries * positives, axis=1, keepdims=True)
    negative_scores = np.einsum("bd,bnd->bn", queries, negatives)
    logits = np.concatenate([positive_scores, negative_scores], axis=1) / float(temperature)
    if not np.isfinite(logits).all():
        raise ValueError("contrastive logits must be finite")
    logits -= np.max(logits, axis=1, keepdims=True)
    log_denominator = np.log(np.sum(np.exp(logits), axis=1))
    return float(np.mean(log_denominator - logits[:, 0]))


@dataclass(frozen=True)
class RAFTTrainingRecord:
    """Open-book fine-tuning record with oracle evidence and distractors."""

    question: str
    answer: str
    documents: tuple[Document, ...]
    oracle_doc_ids: tuple[str, ...]
    oracle_included: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("RAFT question must be a non-empty string")
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("RAFT answer must be a non-empty string")
        documents = _snapshots(self.documents)
        if len({_document_key(document) for document in documents}) != len(documents):
            raise ValueError("RAFT documents must be unique")
        if not isinstance(self.oracle_included, bool):
            raise TypeError("oracle_included must be a boolean")
        oracle_doc_ids = tuple(self.oracle_doc_ids)
        if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in oracle_doc_ids):
            raise ValueError("RAFT oracle document ids must be non-empty strings")
        if len(set(oracle_doc_ids)) != len(oracle_doc_ids):
            raise ValueError("RAFT oracle document ids must be unique")
        document_keys = {_document_key(document) for document in documents}
        oracle_keys = set(oracle_doc_ids)
        if self.oracle_included and not oracle_keys.issubset(document_keys):
            raise ValueError("Included RAFT oracle ids must be present in documents")
        if not self.oracle_included and oracle_keys & document_keys:
            raise ValueError("Excluded RAFT oracle ids must not be present in documents")
        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "oracle_doc_ids", oracle_doc_ids)
        object.__setattr__(self, "metadata", deepcopy(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "documents": [_serialize_document(document) for document in self.documents],
            "oracle_doc_ids": list(self.oracle_doc_ids),
            "oracle_included": self.oracle_included,
            "metadata": deepcopy(self.metadata),
        }

    def render_prompt(self) -> str:
        evidence = "\n\n".join(
            f"[source: {_document_key(document)}]\n{document.content}" for document in self.documents
        )
        return (
            "Réponds à la question à partir des documents utiles et ignore les distracteurs. "
            "Cite les sources utilisées.\n\n"
            f"Documents :\n{evidence}\n\nQuestion : {self.question}\nRéponse :"
        )


class RAFTDatasetBuilder:
    """Create deterministic RAFT-style records from retrieval examples."""

    def __init__(self, *, oracle_probability: float = 1.0, seed: int = 0):
        if not 0.0 <= oracle_probability <= 1.0 or not math.isfinite(oracle_probability):
            raise ValueError("oracle_probability must be between 0 and 1")
        self.oracle_probability = float(oracle_probability)
        self.seed = int(seed)

    def build(self, examples: Iterable[RetrievalTrainingExample]) -> list[RAFTTrainingRecord]:
        rng = random.Random(self.seed)
        records: list[RAFTTrainingRecord] = []
        for example in examples:
            if example.answer is None or not example.answer.strip():
                raise ValueError("RAFT records require a non-empty answer")
            include_oracle = rng.random() < self.oracle_probability
            documents = (
                (*example.positive_documents, *example.negative_documents)
                if include_oracle
                else example.negative_documents
            )
            records.append(
                RAFTTrainingRecord(
                    question=example.query,
                    answer=example.answer,
                    documents=_snapshots(documents),
                    oracle_doc_ids=example.positive_doc_ids,
                    oracle_included=include_oracle,
                    metadata={**example.metadata, "recipe": "raft"},
                )
            )
        return records


@runtime_checkable
class RetrievalTrainerProtocol(Protocol):
    """Adapter boundary for PyTorch, Sentence Transformers or hosted trainers."""

    def fit(
        self,
        examples: Sequence[RetrievalTrainingExample | DistilledRetrievalExample],
        **kwargs: Any,
    ) -> Mapping[str, Any] | None:
        ...


class RetrievalTrainingPipeline:
    """Mine, optionally distill, then hand examples to an injected trainer."""

    def __init__(
        self,
        miner: HardNegativeMiner | None = None,
        distiller: TeacherScoreDistiller | None = None,
    ):
        self.miner = miner
        self.distiller = distiller

    def prepare(
        self,
        examples: Iterable[RetrievalTrainingExample],
    ) -> list[RetrievalTrainingExample | DistilledRetrievalExample]:
        prepared: list[RetrievalTrainingExample | DistilledRetrievalExample] = []
        for source in examples:
            example = source
            if self.miner is not None and not example.negative_documents:
                mined = self.miner.mine(example.query, example.positive_documents)
                example = RetrievalTrainingExample(
                    query=example.query,
                    positive_documents=example.positive_documents,
                    negative_documents=mined.negative_documents,
                    answer=example.answer,
                    metadata={**example.metadata, **mined.metadata},
                )
            prepared.append(self.distiller.distill(example) if self.distiller is not None else example)
        return prepared

    def fit(
        self,
        examples: Iterable[RetrievalTrainingExample],
        trainer: RetrievalTrainerProtocol,
        **kwargs: Any,
    ) -> Mapping[str, Any] | None:
        if not isinstance(trainer, RetrievalTrainerProtocol) or not callable(getattr(trainer, "fit", None)):
            raise TypeError("trainer must implement RetrievalTrainerProtocol.fit")
        prepared = self.prepare(examples)
        if not prepared:
            raise ValueError("Retrieval training requires at least one example")
        return trainer.fit(prepared, **kwargs)


def _softmax(scores: Sequence[float], *, temperature: float) -> tuple[float, ...]:
    if not scores:
        return ()
    scaled = [score / temperature for score in scores]
    maximum = max(scaled)
    exponentials = [math.exp(score - maximum) for score in scaled]
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


def _serialize_document(document: Document) -> dict[str, Any]:
    return {
        "content": document.content,
        "metadata": deepcopy(document.metadata),
        "doc_id": document.doc_id,
        "score": document.score,
    }


__all__ = [
    "DistilledRetrievalExample",
    "HardNegativeMiner",
    "RAFTDatasetBuilder",
    "RAFTTrainingRecord",
    "RetrievalTrainerProtocol",
    "RetrievalTrainingExample",
    "RetrievalTrainingPipeline",
    "TeacherScoreDistiller",
    "contrastive_retrieval_loss",
]
