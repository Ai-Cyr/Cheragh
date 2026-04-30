"""Retrieval evaluation metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping, Sequence

from ..base import BaseRetriever, Document


@dataclass
class RetrievalExample:
    """One retrieval evaluation example.

    ``expected_doc_ids`` are treated as relevant ids. A retrieved chunk also
    matches when its ``metadata.parent_doc_id`` is expected, which makes chunked
    corpora easier to evaluate against document-level labels.

    ``graded_relevance`` optionally assigns gains per doc id, for nDCG.
    """

    query: str
    expected_doc_ids: set[str]
    graded_relevance: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalEvaluationResult:
    metrics: dict[str, float]
    rows: list[dict]


def evaluate_retrieval(
    examples: Iterable[RetrievalExample | dict],
    retriever: BaseRetriever,
    top_k: int = 5,
) -> RetrievalEvaluationResult:
    """Evaluate a retriever.

    Metrics returned in v0.9:

    - ``hit_rate@k``: at least one relevant context retrieved.
    - ``mrr``: mean reciprocal rank of the first relevant context.
    - ``precision@k``: relevant retrieved contexts divided by k.
    - ``recall@k``: relevant labels covered by the retrieved contexts.
    - ``ndcg@k``: normalized discounted cumulative gain, binary or graded.
    - ``context_precision@k``: average precision over the retrieved context list.
    """
    rows: list[dict] = []
    hit_count = 0
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    context_precisions: list[float] = []

    parsed = [_parse_example(example) for example in examples]
    for example in parsed:
        docs = retriever.retrieve(example.query, top_k=top_k)
        retrieved_ids = [doc.doc_id for doc in docs if doc.doc_id is not None]
        expected = set(example.expected_doc_ids)
        relevance_scores = [_relevance(doc, example) for doc in docs]
        binary_hits = [score > 0 for score in relevance_scores]
        hit = any(binary_hits)
        hit_count += int(hit)

        rr = _reciprocal_rank(binary_hits)
        precision = sum(binary_hits) / max(top_k, 1)
        covered_expected = _covered_expected_ids(docs, expected)
        recall = len(covered_expected) / len(expected) if expected else 1.0
        ndcg = _ndcg(relevance_scores, example, top_k=top_k)
        context_precision = _average_precision(binary_hits, relevant_total=min(len(expected), top_k) if expected else sum(binary_hits))

        reciprocal_ranks.append(rr)
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        context_precisions.append(context_precision)
        rows.append(
            {
                "query": example.query,
                "expected_doc_ids": sorted(expected),
                "retrieved_doc_ids": retrieved_ids,
                "covered_expected_doc_ids": sorted(covered_expected),
                "hit": hit,
                "reciprocal_rank": rr,
                f"precision@{top_k}": precision,
                f"recall@{top_k}": recall,
                f"ndcg@{top_k}": ndcg,
                f"context_precision@{top_k}": context_precision,
            }
        )

    n = len(parsed) or 1
    return RetrievalEvaluationResult(
        metrics={
            f"hit_rate@{top_k}": hit_count / n,
            "mrr": sum(reciprocal_ranks) / n,
            f"precision@{top_k}": sum(precisions) / n,
            f"recall@{top_k}": sum(recalls) / n,
            f"ndcg@{top_k}": sum(ndcgs) / n,
            f"context_precision@{top_k}": sum(context_precisions) / n,
        },
        rows=rows,
    )


def recall_at_k(retrieved_ids: Sequence[str], expected_ids: set[str], k: int) -> float:
    if not expected_ids:
        return 1.0
    return len(set(retrieved_ids[:k]) & expected_ids) / len(expected_ids)


def ndcg_at_k(relevance_scores: Sequence[float], ideal_scores: Sequence[float] | None = None, k: int = 5) -> float:
    scores = list(relevance_scores[:k])
    if ideal_scores is None:
        ideal_scores = sorted(scores, reverse=True)
    else:
        ideal_scores = sorted(list(ideal_scores), reverse=True)[:k]
    dcg = _dcg(scores)
    idcg = _dcg(ideal_scores)
    return dcg / idcg if idcg > 0 else 0.0


def context_precision_at_k(relevance_flags: Sequence[bool], k: int | None = None) -> float:
    flags = list(relevance_flags[:k]) if k is not None else list(relevance_flags)
    return _average_precision(flags, relevant_total=sum(flags))


def _parse_example(example: RetrievalExample | dict) -> RetrievalExample:
    if isinstance(example, RetrievalExample):
        if not example.graded_relevance:
            example.graded_relevance.update({doc_id: 1.0 for doc_id in example.expected_doc_ids})
        return example
    expected = example.get("expected_doc_ids", example.get("expected", []))
    if isinstance(expected, str):
        expected = [expected]
    graded = example.get("graded_relevance", example.get("relevance", {})) or {}
    if isinstance(graded, Mapping):
        graded_relevance = {str(key): float(value) for key, value in graded.items()}
    else:
        graded_relevance = {}
    expected_set = set(str(doc_id) for doc_id in expected)
    for doc_id in expected_set:
        graded_relevance.setdefault(doc_id, 1.0)
    return RetrievalExample(query=str(example["query"]), expected_doc_ids=expected_set, graded_relevance=graded_relevance)


def _candidate_ids(doc: Document) -> set[str]:
    ids: set[str] = set()
    if doc.doc_id:
        ids.add(str(doc.doc_id))
        # Common chunk convention: parent#chunk-0 should count for parent labels.
        ids.add(str(doc.doc_id).split("#", 1)[0])
    for key in ("parent_doc_id", "source_doc_id", "source", "filename"):
        value = doc.metadata.get(key)
        if value is not None:
            ids.add(str(value))
    return ids


def _relevance(doc: Document, example: RetrievalExample) -> float:
    candidates = _candidate_ids(doc)
    if example.graded_relevance:
        return max((example.graded_relevance.get(candidate, 0.0) for candidate in candidates), default=0.0)
    return 1.0 if candidates & example.expected_doc_ids else 0.0


def _covered_expected_ids(docs: Sequence[Document], expected: set[str]) -> set[str]:
    covered: set[str] = set()
    for doc in docs:
        covered.update(_candidate_ids(doc) & expected)
    return covered


def _reciprocal_rank(binary_hits: Sequence[bool]) -> float:
    for rank, is_hit in enumerate(binary_hits, start=1):
        if is_hit:
            return 1.0 / rank
    return 0.0


def _dcg(scores: Sequence[float]) -> float:
    return sum(((2.0**score) - 1.0) / math.log2(rank + 1) for rank, score in enumerate(scores, start=1))


def _ndcg(relevance_scores: Sequence[float], example: RetrievalExample, top_k: int) -> float:
    if example.graded_relevance:
        ideal = sorted(example.graded_relevance.values(), reverse=True)[:top_k]
    else:
        ideal = [1.0] * min(len(example.expected_doc_ids), top_k)
    return ndcg_at_k(relevance_scores, ideal, k=top_k)


def _average_precision(binary_hits: Sequence[bool], relevant_total: int) -> float:
    if relevant_total <= 0:
        return 0.0
    precisions_at_hits: list[float] = []
    hit_count = 0
    for rank, is_hit in enumerate(binary_hits, start=1):
        if is_hit:
            hit_count += 1
            precisions_at_hits.append(hit_count / rank)
    return sum(precisions_at_hits) / relevant_total if precisions_at_hits else 0.0
