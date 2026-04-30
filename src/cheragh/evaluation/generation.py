"""Lightweight generation quality helpers."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from ..base import Document, _tokenize
from ..citations import extract_citations


@dataclass
class GenerationEvaluationResult:
    metrics: dict[str, float]
    rows: list[dict]

    def to_markdown(self) -> str:
        lines = ["| metric | value |", "|---|---:|"]
        for key, value in self.metrics.items():
            lines.append(f"| {key} | {value:.4f} |")
        return "\n".join(lines)


def extract_source_citations(answer: str) -> list[str]:
    return extract_citations(answer)


def citation_coverage(answer: str, expected_source_ids: Iterable[str]) -> float:
    """Share of expected sources cited in the answer."""
    expected = set(expected_source_ids)
    if not expected:
        return 1.0
    cited = set(extract_source_citations(answer))
    return len(expected & cited) / len(expected)


def citation_accuracy(answer: str, allowed_source_ids: Iterable[str]) -> float:
    """Share of answer citations that point to allowed source IDs."""
    citations = extract_source_citations(answer)
    if not citations:
        return 1.0
    allowed = set(allowed_source_ids)
    return sum(1 for citation in citations if citation in allowed) / len(citations)


def lexical_groundedness(answer: str, contexts: Sequence[str | Document]) -> float:
    """Approximate groundedness using content-word overlap with retrieved context."""
    answer_tokens = set(_tokenize(_strip_citations(answer)))
    if not answer_tokens:
        return 1.0
    context_text = " ".join(ctx.content if isinstance(ctx, Document) else str(ctx) for ctx in contexts)
    context_tokens = set(_tokenize(context_text))
    if not context_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def answer_relevance(answer: str, query: str) -> float:
    """Approximate answer relevance using lexical overlap with the query."""
    answer_tokens = set(_tokenize(answer))
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 1.0
    return len(answer_tokens & query_tokens) / len(query_tokens)


def evaluate_generation(rows: Iterable[dict]) -> GenerationEvaluationResult:
    """Evaluate generated answers from dictionaries.

    Each row may contain: ``query``, ``answer``, ``contexts`` and either
    ``expected_doc_ids`` or ``source_ids``.
    """
    parsed_rows = []
    accum: dict[str, list[float]] = {
        "citation_accuracy": [],
        "citation_coverage": [],
        "groundedness": [],
        "answer_relevance": [],
    }
    for row in rows:
        answer = str(row.get("answer", ""))
        query = str(row.get("query", ""))
        contexts = row.get("contexts", []) or row.get("documents", []) or []
        source_ids = row.get("source_ids", row.get("expected_doc_ids", [])) or []
        values = {
            "citation_accuracy": citation_accuracy(answer, source_ids),
            "citation_coverage": citation_coverage(answer, source_ids),
            "groundedness": lexical_groundedness(answer, contexts),
            "answer_relevance": answer_relevance(answer, query),
        }
        for key, value in values.items():
            accum[key].append(value)
        parsed_rows.append({**row, **values})
    metrics = {key: (sum(values) / len(values) if values else 0.0) for key, values in accum.items()}
    return GenerationEvaluationResult(metrics=metrics, rows=parsed_rows)


def _strip_citations(answer: str) -> str:
    return re.sub(r"\[source:\s*[^\]]+\]", "", answer, flags=re.I)
