"""Tracing utilities for RAG pipeline execution."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
from numbers import Real
import os
from pathlib import Path
import threading
from time import perf_counter, time
from typing import Any
from uuid import uuid4


_TRACE_WRITE_LOCK = threading.RLock()
_PRICING_FIELDS = frozenset({"input_per_1k", "output_per_1k", "currency"})


def _validate_pricing(pricing: Mapping[str, Any]) -> dict[str, float | str]:
    if not isinstance(pricing, Mapping):
        raise TypeError("pricing must be a mapping")
    unknown = [key for key in pricing if key not in _PRICING_FIELDS]
    if unknown:
        rendered = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"pricing contains unsupported fields: {rendered}")
    if not pricing:
        return {}

    validated: dict[str, float | str] = {}
    for key in ("input_per_1k", "output_per_1k"):
        value = pricing.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"pricing.{key} must be a finite real number >= 0")
        rate = float(value)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError(f"pricing.{key} must be a finite real number >= 0")
        validated[key] = rate

    currency = pricing.get("currency", "USD")
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("pricing.currency must be a non-empty string")
    validated["currency"] = currency.strip()
    return validated


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
        if self.ended_at is None:
            self.ended_at = perf_counter()
            self.duration_ms = max(0.0, (self.ended_at - self.started_at) * 1000)
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
    _started_at_monotonic: float = field(default_factory=perf_counter, init=False, repr=False)

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
        pricing: Mapping[str, Any] | None = None,
    ) -> None:
        """Record approximate generation token/cost details.

        ``pricing`` can contain ``input_per_1k`` and ``output_per_1k`` floats.
        Values are approximate unless callers replace token counts with provider
        telemetry.
        """

        validated_pricing = _validate_pricing(pricing) if pricing is not None else {}
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
        if validated_pricing:
            input_cost = input_tokens / 1000 * float(validated_pricing["input_per_1k"])
            output_cost = output_tokens / 1000 * float(validated_pricing["output_per_1k"])
            self.cost.update(
                {
                    "currency": validated_pricing["currency"],
                    "input_cost_estimated": input_cost,
                    "output_cost_estimated": output_cost,
                    "total_cost_estimated": input_cost + output_cost,
                }
            )

    def finish(self, **metadata: Any) -> None:
        if self.ended_at_unix is None:
            self.ended_at_unix = time()
            # Wall clocks can jump under NTP/manual adjustment; latency cannot.
            self.duration_ms = max(0.0, (perf_counter() - self._started_at_monotonic) * 1000)
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

    def export_jsonl(
        self,
        path: str | Path,
        *,
        include_prompt: bool = False,
        durable: bool = False,
    ) -> None:
        append_trace_jsonl(path, self, include_prompt=include_prompt, durable=durable)


def append_trace_jsonl(
    path: str | Path,
    trace: RAGTrace,
    *,
    include_prompt: bool = False,
    durable: bool = False,
) -> None:
    """Append exactly one valid JSONL record.

    Writes are protected across threads and, on POSIX, cooperating processes.
    ``durable=True`` fsyncs each record for audit-grade persistence at the cost
    of throughput.
    """

    p = Path(path)
    record = json.dumps(
        trace.to_dict(include_prompt=include_prompt),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    with _TRACE_WRITE_LOCK:
        fd = os.open(p, flags, 0o600)
        try:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - Windows fallback
                fcntl = None  # type: ignore[assignment]
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                view = memoryview(record)
                while view:
                    try:
                        written = os.write(fd, view)
                    except InterruptedError:  # pragma: no cover - signal timing
                        continue
                    if written <= 0:  # pragma: no cover - defensive OS contract
                        raise OSError("trace append made no forward progress")
                    view = view[written:]
                if durable:
                    os.fsync(fd)
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
