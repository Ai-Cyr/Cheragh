"""Pipeline-level evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .retrieval import evaluate_retrieval
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
    """Run a RAGEngine over examples and compute retrieval + generation metrics."""
    examples_list = list(examples)
    retrieval = evaluate_retrieval(examples_list, engine.retriever, top_k=top_k)
    rows = []
    for example in examples_list:
        response = engine.ask(str(example["query"]), top_k=top_k)
        source_ids = [source.doc_id for source in response.sources if source.doc_id]
        rows.append(
            {
                "query": example["query"],
                "answer": response.answer,
                "contexts": response.retrieved_documents,
                "source_ids": source_ids,
                "expected_doc_ids": example.get("expected_doc_ids", []),
            }
        )
    generation = evaluate_generation(rows)
    return PipelineEvaluationResult(
        retrieval_metrics=retrieval.metrics,
        generation_metrics=generation.metrics,
        rows=generation.rows,
    )
