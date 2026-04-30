"""Evaluation helpers."""
from .retrieval import (
    RetrievalExample,
    RetrievalEvaluationResult,
    context_precision_at_k,
    evaluate_retrieval,
    ndcg_at_k,
    recall_at_k,
)
from .generation import (
    GenerationEvaluationResult,
    answer_relevance,
    citation_accuracy,
    citation_coverage,
    evaluate_generation,
    extract_source_citations,
    lexical_groundedness,
)
from .pipeline import PipelineEvaluationResult, evaluate_pipeline

__all__ = [
    "RetrievalExample",
    "RetrievalEvaluationResult",
    "evaluate_retrieval",
    "recall_at_k",
    "ndcg_at_k",
    "context_precision_at_k",
    "GenerationEvaluationResult",
    "answer_relevance",
    "citation_accuracy",
    "citation_coverage",
    "evaluate_generation",
    "extract_source_citations",
    "lexical_groundedness",
    "PipelineEvaluationResult",
    "evaluate_pipeline",
]
