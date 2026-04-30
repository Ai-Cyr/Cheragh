"""User feedback collection and export utilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid
from typing import Any, Iterable, Protocol


@dataclass
class FeedbackRecord:
    """Single feedback event for a RAG answer."""

    query: str
    rating: str
    response_id: str | None = None
    answer: str | None = None
    correct_answer: str | None = None
    correct_source_ids: list[str] = field(default_factory=list)
    retrieved_source_ids: list[str] = field(default_factory=list)
    comment: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    collection_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeedbackRecord":
        payload = dict(data)
        payload.setdefault("correct_source_ids", [])
        payload.setdefault("retrieved_source_ids", [])
        payload.setdefault("metadata", {})
        return cls(**payload)


@dataclass
class FeedbackSummary:
    """Aggregated feedback statistics."""

    total: int
    by_rating: dict[str, int]
    positive_rate: float
    negative_rate: float
    top_missing_sources: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_rating": self.by_rating,
            "positive_rate": self.positive_rate,
            "negative_rate": self.negative_rate,
            "top_missing_sources": self.top_missing_sources,
        }


class FeedbackStore(Protocol):
    def append(self, record: FeedbackRecord) -> None: ...
    def list(self, **filters: Any) -> list[FeedbackRecord]: ...


class InMemoryFeedbackStore:
    """Simple in-memory feedback store."""

    def __init__(self):
        self.records: list[FeedbackRecord] = []

    def append(self, record: FeedbackRecord) -> None:
        self.records.append(record)

    def list(self, **filters: Any) -> list[FeedbackRecord]:
        return _filter_records(self.records, filters)


class JSONLFeedbackStore:
    """Append-only JSONL feedback store."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: FeedbackRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def list(self, **filters: Any) -> list[FeedbackRecord]:
        if not self.path.exists():
            return []
        records: list[FeedbackRecord] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(FeedbackRecord.from_dict(json.loads(line)))
        return _filter_records(records, filters)


class FeedbackLoop:
    """Collects feedback and exports datasets for evaluation/tuning."""

    def __init__(self, store: FeedbackStore | None = None):
        self.store = store or InMemoryFeedbackStore()
    @classmethod
    def from_jsonl(cls, path: str | Path) -> "FeedbackLoop":
        """Create a feedback loop backed by an append-only JSONL file."""
        return cls(JSONLFeedbackStore(path))


    def log_feedback(
        self,
        query: str,
        rating: str,
        response: Any | None = None,
        response_id: str | None = None,
        answer: str | None = None,
        correct_answer: str | None = None,
        correct_source_ids: Iterable[str] | None = None,
        comment: str | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        collection_id: str | None = None,
        **metadata: Any,
    ) -> FeedbackRecord:
        if response is not None:
            answer = answer if answer is not None else getattr(response, "answer", None)
            if response_id is None:
                response_id = getattr(response, "response_id", None) or getattr(response, "id", None)
            retrieved = [getattr(src, "doc_id", None) for src in getattr(response, "sources", []) or []]
            retrieved = [str(item) for item in retrieved if item]
            resp_meta = getattr(response, "metadata", {}) or {}
            tenant_id = tenant_id or (resp_meta.get("tenant") or {}).get("tenant_id")
            collection_id = collection_id or (resp_meta.get("tenant") or {}).get("collection_id")
        else:
            retrieved = []
        record = FeedbackRecord(
            query=query,
            rating=_normalize_rating(rating),
            response_id=response_id,
            answer=answer,
            correct_answer=correct_answer,
            correct_source_ids=list(correct_source_ids or []),
            retrieved_source_ids=retrieved,
            comment=comment,
            user_id=user_id,
            tenant_id=tenant_id,
            collection_id=collection_id,
            metadata=metadata,
        )
        self.store.append(record)
        return record

    def list_feedback(self, **filters: Any) -> list[FeedbackRecord]:
        return self.store.list(**filters)

    def summary(self, **filters: Any) -> FeedbackSummary:
        records = self.list_feedback(**filters)
        by_rating: dict[str, int] = {}
        missing: dict[str, int] = {}
        for record in records:
            by_rating[record.rating] = by_rating.get(record.rating, 0) + 1
            retrieved = set(record.retrieved_source_ids)
            for source_id in record.correct_source_ids:
                if source_id not in retrieved:
                    missing[source_id] = missing.get(source_id, 0) + 1
        total = len(records)
        positive = by_rating.get("positive", 0)
        negative = by_rating.get("negative", 0)
        top_missing = sorted(missing.items(), key=lambda item: item[1], reverse=True)[:10]
        return FeedbackSummary(
            total=total,
            by_rating=by_rating,
            positive_rate=(positive / total) if total else 0.0,
            negative_rate=(negative / total) if total else 0.0,
            top_missing_sources=top_missing,
        )

    def export_evalset(self, path: str | Path | None = None, only_negative: bool = False) -> list[dict[str, Any]]:
        records = self.list_feedback()
        if only_negative:
            records = [record for record in records if record.rating == "negative"]
        dataset = [
            {
                "question": record.query,
                "expected_answer": record.correct_answer or record.answer or "",
                "expected_doc_ids": record.correct_source_ids,
                "metadata": {
                    "feedback_id": record.feedback_id,
                    "rating": record.rating,
                    "tenant_id": record.tenant_id,
                    "collection_id": record.collection_id,
                    "comment": record.comment,
                },
            }
            for record in records
        ]
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return dataset


def _normalize_rating(rating: str) -> str:
    value = rating.lower().strip()
    if value in {"good", "up", "thumbs_up", "+", "1", "positive", "ok"}:
        return "positive"
    if value in {"bad", "down", "thumbs_down", "-", "0", "negative", "ko"}:
        return "negative"
    return value or "unknown"


def _filter_records(records: Iterable[FeedbackRecord], filters: dict[str, Any]) -> list[FeedbackRecord]:
    result = list(records)
    for key, expected in filters.items():
        if expected is None:
            continue
        result = [record for record in result if getattr(record, key) == expected]
    return result
