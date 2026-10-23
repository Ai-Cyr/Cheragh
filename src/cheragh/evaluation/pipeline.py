"""Pipeline-level evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..base import _snapshot_document, _validate_top_k
from .retrieval import _evaluate_retrieval_results, _parse_example
from .generation import evaluate_generation


@dataclass
class PipelineEvaluationResult:
    retrieval_metrics: dict[str, float]
    generation_metrics: dict[str, float]
    rows: list[dict]

    def to_markdown(self) -> str:
        lines = ["| metric | value |", "|---|---:|"]
        for group, metrics in [("retrieval", self.retrieval_metrics), ("generation", self.generation_metrics)]:
            for key, value in metrics.items():
                lines.append(f"| {group}.{key} | {value:.4f} |")
        return "\n".join(lines)


def evaluate_pipeline(engine, examples: Iterable[dict], top_k: int = 5) -> PipelineEvaluationResult:
    """Score each response's actual evidence and generation in one engine call.

    Retrieval metrics describe the final passages after query transforms,
    reranking and compression, not a separate first-stage retrieval. Evaluate
    the raw retriever with ``evaluate_retrieval`` when that is the desired metric.
    """
    top_k = _validate_top_k(top_k)
    examples_list = [_parse_example(example) for example in examples]
    rows = []
    evidence_results = []
    for example in examples_list:
        response = engine.ask(example.query, top_k=top_k)
        source_ids = [source.doc_id for source in response.sources if source.doc_id]
        documents = [_snapshot_document(document) for document in response.retrieved_documents]
        evidence_results.append((example, documents))
        rows.append(
            {
                "query": example.query,
                "answer": response.answer,
                "contexts": documents,
                "source_ids": source_ids,
                "expected_doc_ids": sorted(example.expected_doc_ids),
            }
        )
    retrieval = _evaluate_retrieval_results(evidence_results, top_k=top_k)
    generation = evaluate_generation(rows)
    return PipelineEvaluationResult(
        retrieval_metrics=retrieval.metrics,
        generation_metrics=generation.metrics,
        rows=generation.rows,
    )
