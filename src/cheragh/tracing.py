"""Tracing utilities for RAG pipeline execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from time import perf_counter, time
from typing import Any
from uuid import uuid4


def estimate_tokens(text: str) -> int:
    """Return a deterministic rough token estimate without provider SDKs."""

    if not text:
        return 0
    # Practical approximation used for observability only; not billing-grade.
    return max(1, round(len(text) / 4))


@dataclass
class RAGTraceStep:
    """One timed step in a RAG request."""

    name: str
    started_at: float
    ended_at: float | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, **metadata: Any) -> None:
        self.ended_at = perf_counter()
        self.duration_ms = (self.ended_at - self.started_at) * 1000
        self.metadata.update(metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class RAGTrace:
    """Serializable request trace for production debugging and evaluation."""

    request_id: str = field(default_factory=lambda: uuid4().hex)
    started_at_unix: float = field(default_factory=time)
    ended_at_unix: float | None = None
    duration_ms: float | None = None
    steps: list[RAGTraceStep] = field(default_factory=list)
    retrieval: list[dict[str, Any]] = field(default_factory=list)
    reranking: list[dict[str, Any]] = field(default_factory=list)
    compression: dict[str, Any] = field(default_factory=dict)
    prompt: str | None = None
    query: str | None = None
    query_variants: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def start_step(self, name: str, **metadata: Any) -> RAGTraceStep:
        step = RAGTraceStep(name=name, started_at=perf_counter(), metadata=dict(metadata))
        self.steps.append(step)
        return step

    def add_retrieval(self, query: str, documents: list[Any]) -> None:
        self.retrieval.append(
            {
                "query": query,
                "documents": [
                    {
                        "doc_id": getattr(doc, "doc_id", None),
                        "score": getattr(doc, "score", None),
                        "metadata": dict(getattr(doc, "metadata", {}) or {}),
                    }
                    for doc in documents
                ],
            }
        )

    def record_generation(
        self,
        *,
        prompt: str,
        answer: str,
        model: str | None = None,
        pricing: dict[str, float] | None = None,
    ) -> None:
        """Record approximate generation token/cost details.

        ``pricing`` can contain ``input_per_1k`` and ``output_per_1k`` floats.
        Values are approximate unless callers replace token counts with provider
        telemetry.
        """

        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(answer)
        self.token_usage.update(
            {
                "input_tokens_estimated": input_tokens,
                "output_tokens_estimated": output_tokens,
                "total_tokens_estimated": input_tokens + output_tokens,
            }
        )
        if model:
            self.metadata["model"] = model
        if pricing:
            input_cost = input_tokens / 1000 * float(pricing.get("input_per_1k", 0.0))
            output_cost = output_tokens / 1000 * float(pricing.get("output_per_1k", 0.0))
            self.cost.update(
                {
                    "currency": pricing.get("currency", "USD"),
                    "input_cost_estimated": input_cost,
                    "output_cost_estimated": output_cost,
                    "total_cost_estimated": input_cost + output_cost,
                }
            )

    def finish(self, **metadata: Any) -> None:
        if self.ended_at_unix is None:
            self.ended_at_unix = time()
            self.duration_ms = (self.ended_at_unix - self.started_at_unix) * 1000
        self.metadata.update(metadata)

    def to_dict(self, include_prompt: bool = False) -> dict[str, Any]:
        data = {
            "request_id": self.request_id,
            "started_at_unix": self.started_at_unix,
            "ended_at_unix": self.ended_at_unix,
            "duration_ms": self.duration_ms,
            "steps": [step.to_dict() for step in self.steps],
            "retrieval": self.retrieval,
            "reranking": self.reranking,
            "compression": self.compression,
            "query": self.query,
            "query_variants": self.query_variants,
            "warnings": self.warnings,
            "cost": self.cost,
            "token_usage": self.token_usage,
            "metadata": self.metadata,
        }
        if include_prompt:
            data["prompt"] = self.prompt
        elif self.prompt is not None:
            data["prompt_chars"] = len(self.prompt)
        return data

    def export_jsonl(self, path: str | Path, *, include_prompt: bool = False) -> None:
        append_trace_jsonl(path, self, include_prompt=include_prompt)


def append_trace_jsonl(path: str | Path, trace: RAGTrace, *, include_prompt: bool = False) -> None:
    """Append one trace to a JSONL file."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace.to_dict(include_prompt=include_prompt), ensure_ascii=False) + "\n")
