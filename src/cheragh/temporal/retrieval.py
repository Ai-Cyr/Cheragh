"""Dependency-free temporal retrieval primitives.

The module treats ``metadata["timestamp"]`` as the time at which a piece of
evidence became available.  Optional ``valid_from`` / ``valid_to`` values model
the interval in which a version is valid.  Every timestamp is required to carry
a timezone so comparisons cannot silently depend on the host locale.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import math
from numbers import Real
from typing import Any, Mapping

from ..base import BaseRetriever, Document, _snapshot_document, _validate_top_k


TIMESTAMP_KEY = "timestamp"
VALID_FROM_KEY = "valid_from"
VALID_TO_KEY = "valid_to"
VERSION_ID_KEY = "version_id"
VERSION_GROUP_KEY = "version_group"
TEMPORAL_PROVENANCE_KEY = "temporal_provenance"


TemporalValue = datetime | str


class MissingTimestampPolicy(str, Enum):
    """How temporal retrieval handles evidence without a timestamp."""

    FAIL = "fail"
    ALLOW = "allow"


class ConflictResolution(str, Enum):
    """How several retrieved versions of the same logical document are handled."""

    LATEST = "latest"
    HIGHEST_SCORE = "highest_score"
    ALL = "all"
    RAISE = "raise"


def temporal_metadata(
    timestamp: TemporalValue,
    *,
    valid_from: TemporalValue | None = None,
    valid_to: TemporalValue | None = None,
    version_id: str | int | None = None,
    version_group: str | int | None = None,
) -> dict[str, Any]:
    """Build JSON-friendly, validated temporal metadata.

    Datetimes are normalized to UTC ISO-8601 strings.  Naive datetimes and ISO
    strings without an offset are rejected.
    """

    metadata: dict[str, Any] = {TIMESTAMP_KEY: _isoformat(timestamp, name=TIMESTAMP_KEY)}
    metadata.update(
        version_metadata(
            version_id=version_id,
            version_group=version_group,
            valid_from=valid_from,
            valid_to=valid_to,
        )
    )
    return metadata


def version_metadata(
    version_id: str | int | None = None,
    version_group: str | int | None = None,
    *,
    valid_from: TemporalValue | None = None,
    valid_to: TemporalValue | None = None,
) -> dict[str, Any]:
    """Build validated version metadata that can be merged into a document.

    ``version_group`` identifies versions of the same logical fact or source.
    ``valid_to`` is exclusive when an ``as_of`` snapshot is requested.
    """

    metadata: dict[str, Any] = {}
    normalized_from = _optional_datetime(valid_from, name=VALID_FROM_KEY)
    normalized_to = _optional_datetime(valid_to, name=VALID_TO_KEY)
    if normalized_from is not None and normalized_to is not None and normalized_from >= normalized_to:
        raise ValueError("valid_from must be earlier than valid_to")
    if normalized_from is not None:
        metadata[VALID_FROM_KEY] = normalized_from.isoformat()
    if normalized_to is not None:
        metadata[VALID_TO_KEY] = normalized_to.isoformat()
    if version_id is not None:
        metadata[VERSION_ID_KEY] = _identifier(version_id, name=VERSION_ID_KEY)
    if version_group is not None:
        metadata[VERSION_GROUP_KEY] = _identifier(version_group, name=VERSION_GROUP_KEY)
    return metadata


class TemporalDocument(Document):
    """A :class:`Document` with validated, persistence-safe temporal metadata."""

    def __init__(
        self,
        content: str,
        timestamp: TemporalValue,
        *,
        valid_from: TemporalValue | None = None,
        valid_to: TemporalValue | None = None,
        version_id: str | int | None = None,
        version_group: str | int | None = None,
        metadata: Mapping[str, Any] | None = None,
        doc_id: str | None = None,
        score: float | None = None,
    ) -> None:
        values = deepcopy(dict(metadata or {}))
        reserved = {
            TIMESTAMP_KEY,
            VALID_FROM_KEY,
            VALID_TO_KEY,
            VERSION_ID_KEY,
            VERSION_GROUP_KEY,
        }
        conflicts = sorted(reserved & set(values))
        if conflicts:
            raise ValueError(
                "TemporalDocument metadata contains reserved keys; use the explicit arguments instead: "
                + ", ".join(conflicts)
            )
        values.update(
            temporal_metadata(
                timestamp,
                valid_from=valid_from,
                valid_to=valid_to,
                version_id=version_id,
                version_group=version_group,
            )
        )
        super().__init__(content=content, metadata=values, doc_id=doc_id, score=score)

    @classmethod
    def from_document(
        cls,
        document: Document,
        *,
        timestamp: TemporalValue | None = None,
        valid_from: TemporalValue | None = None,
        valid_to: TemporalValue | None = None,
        version_id: str | int | None = None,
        version_group: str | int | None = None,
    ) -> "TemporalDocument":
        """Create an independent temporal snapshot of an existing document."""

        metadata = deepcopy(document.metadata or {})
        stored_timestamp = metadata.pop(TIMESTAMP_KEY, None)
        stored_valid_from = metadata.pop(VALID_FROM_KEY, None)
        stored_valid_to = metadata.pop(VALID_TO_KEY, None)
        stored_version_id = metadata.pop(VERSION_ID_KEY, None)
        stored_version_group = metadata.pop(VERSION_GROUP_KEY, None)
        resolved_timestamp = timestamp if timestamp is not None else stored_timestamp
        if resolved_timestamp is None:
            raise ValueError(f"document metadata is missing {TIMESTAMP_KEY!r}")
        return cls(
            document.content,
            resolved_timestamp,
            valid_from=valid_from if valid_from is not None else stored_valid_from,
            valid_to=valid_to if valid_to is not None else stored_valid_to,
            version_id=version_id if version_id is not None else stored_version_id,
            version_group=version_group if version_group is not None else stored_version_group,
            metadata=metadata,
            doc_id=document.doc_id,
            score=document.score,
        )

    @property
    def timestamp(self) -> datetime:
        return _require_datetime(self.metadata.get(TIMESTAMP_KEY), name=TIMESTAMP_KEY)

    @property
    def valid_from(self) -> datetime | None:
        return _optional_datetime(self.metadata.get(VALID_FROM_KEY), name=VALID_FROM_KEY)

    @property
    def valid_to(self) -> datetime | None:
        return _optional_datetime(self.metadata.get(VALID_TO_KEY), name=VALID_TO_KEY)

    @property
    def version_id(self) -> str | None:
        value = self.metadata.get(VERSION_ID_KEY)
        return None if value is None else str(value)

    @property
    def version_group(self) -> str | None:
        value = self.metadata.get(VERSION_GROUP_KEY)
        return None if value is None else str(value)


@dataclass
class _TemporalCandidate:
    document: Document
    timestamp: datetime | None
    valid_from: datetime | None
    valid_to: datetime | None
    version_id: str | None
    version_group: str | None
    original_rank: int


class TemporalRetriever(BaseRetriever):
    """Add point-in-time filtering, freshness scoring and version selection.

    Parameters
    ----------
    base_retriever:
        Retriever providing the relevance-ranked candidate pool.
    as_of / start / end:
        Optional default temporal window.  ``as_of`` is mutually exclusive with
        ``start`` and ``end``.  Per-call values on :meth:`retrieve` override the
        whole default window.
    missing_timestamp:
        ``"fail"`` rejects unverifiable evidence.  ``"allow"`` keeps it and
        marks the result as temporally unverified in provenance.
    freshness_half_life:
        Optional positive ``timedelta`` or number of seconds.  With it, evidence
        receives ``exp(-ln(2) * age / half_life)`` freshness.
    freshness_weight:
        Weight in ``[0, 1]`` used to combine base relevance and freshness.
    conflict_resolution:
        Selection policy for documents sharing ``version_group``.
    candidate_top_k:
        Number of base candidates requested before temporal filtering.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        *,
        as_of: TemporalValue | None = None,
        start: TemporalValue | None = None,
        end: TemporalValue | None = None,
        timestamp_key: str = TIMESTAMP_KEY,
        valid_from_key: str = VALID_FROM_KEY,
        valid_to_key: str = VALID_TO_KEY,
        version_id_key: str = VERSION_ID_KEY,
        version_group_key: str = VERSION_GROUP_KEY,
        missing_timestamp: MissingTimestampPolicy | str = MissingTimestampPolicy.FAIL,
        freshness_half_life: timedelta | Real | None = None,
        freshness_weight: float = 0.2,
        conflict_resolution: ConflictResolution | str = ConflictResolution.LATEST,
        candidate_top_k: int = 50,
        reference_time: TemporalValue | None = None,
    ) -> None:
        if not hasattr(base_retriever, "retrieve"):
            raise TypeError("base_retriever must expose retrieve(query, top_k=...)")
        self.base_retriever = base_retriever
        self.timestamp_key = _metadata_key(timestamp_key, name="timestamp_key")
        self.valid_from_key = _metadata_key(valid_from_key, name="valid_from_key")
        self.valid_to_key = _metadata_key(valid_to_key, name="valid_to_key")
        self.version_id_key = _metadata_key(version_id_key, name="version_id_key")
        self.version_group_key = _metadata_key(version_group_key, name="version_group_key")
        self.missing_timestamp = _enum_value(
            MissingTimestampPolicy, missing_timestamp, name="missing_timestamp"
        )
        self.conflict_resolution = _enum_value(
            ConflictResolution, conflict_resolution, name="conflict_resolution"
        )
        self.freshness_half_life_seconds = _half_life_seconds(freshness_half_life)
        self.freshness_weight = _weight(freshness_weight)
        self.candidate_top_k = _validate_top_k(candidate_top_k, name="candidate_top_k")
        self.reference_time = _optional_datetime(reference_time, name="reference_time")
        self._default_window = _validate_window(as_of=as_of, start=start, end=end)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        as_of: TemporalValue | None = None,
        start: TemporalValue | None = None,
        end: TemporalValue | None = None,
    ) -> list[Document]:
        """Retrieve an independent snapshot for a point in time or time range."""

        top_k = _validate_top_k(top_k)
        if any(value is not None for value in (as_of, start, end)):
            point, range_start, range_end = _validate_window(as_of=as_of, start=start, end=end)
        else:
            point, range_start, range_end = self._default_window

        requested_candidates = max(top_k, self.candidate_top_k)
        raw_documents = list(self.base_retriever.retrieve(query, top_k=requested_candidates))
        candidates = [self._candidate(document, rank) for rank, document in enumerate(raw_documents)]
        candidates = [
            candidate
            for candidate in candidates
            if _matches_window(candidate, as_of=point, start=range_start, end=range_end)
        ]

        reference = point or range_end or self.reference_time or datetime.now(timezone.utc)
        for candidate in candidates:
            self._add_provenance(
                candidate,
                as_of=point,
                start=range_start,
                end=range_end,
                reference=reference,
            )
        selected = self._resolve_versions(candidates)
        selected.sort(key=_ranking_key, reverse=True)
        return [candidate.document for candidate in selected[:top_k]]

    def snapshot(self, query: str, *, as_of: TemporalValue, top_k: int = 5) -> list[Document]:
        """Convenience alias for a point-in-time retrieval."""

        return self.retrieve(query, top_k=top_k, as_of=as_of)

    def between(
        self,
        query: str,
        *,
        start: TemporalValue | None = None,
        end: TemporalValue | None = None,
        top_k: int = 5,
    ) -> list[Document]:
        """Convenience alias for an inclusive timestamp range retrieval."""

        if start is None and end is None:
            raise ValueError("between requires start and/or end")
        return self.retrieve(query, top_k=top_k, start=start, end=end)

    def _candidate(self, source: Document, rank: int) -> _TemporalCandidate:
        if not isinstance(source, Document):
            raise TypeError("base_retriever must return Document instances")
        document = _snapshot_document(source)
        timestamp_value = document.metadata.get(self.timestamp_key)
        timestamp = _optional_datetime(timestamp_value, name=f"metadata[{self.timestamp_key!r}]")
        if timestamp is None and self.missing_timestamp is MissingTimestampPolicy.FAIL:
            identifier = document.doc_id or f"rank {rank + 1}"
            raise ValueError(
                f"document {identifier!r} is missing temporal metadata {self.timestamp_key!r}"
            )
        valid_from = _optional_datetime(
            document.metadata.get(self.valid_from_key), name=f"metadata[{self.valid_from_key!r}]"
        )
        valid_to = _optional_datetime(
            document.metadata.get(self.valid_to_key), name=f"metadata[{self.valid_to_key!r}]"
        )
        if valid_from is not None and valid_to is not None and valid_from >= valid_to:
            raise ValueError(
                f"document {document.doc_id or rank!r} has valid_from >= valid_to"
            )
        version_id = _optional_identifier(document.metadata.get(self.version_id_key), name=self.version_id_key)
        version_group = _optional_identifier(
            document.metadata.get(self.version_group_key), name=self.version_group_key
        )
        return _TemporalCandidate(
            document=document,
            timestamp=timestamp,
            valid_from=valid_from,
            valid_to=valid_to,
            version_id=version_id,
            version_group=version_group,
            original_rank=rank,
        )

    def _add_provenance(
        self,
        candidate: _TemporalCandidate,
        *,
        as_of: datetime | None,
        start: datetime | None,
        end: datetime | None,
        reference: datetime,
    ) -> None:
        document = candidate.document
        base_score = document.score
        freshness_score: float | None = None
        if self.freshness_half_life_seconds is not None and candidate.timestamp is not None:
            age_seconds = max(0.0, (reference - candidate.timestamp).total_seconds())
            freshness_score = math.exp(
                -math.log(2.0) * age_seconds / self.freshness_half_life_seconds
            )
            if base_score is None:
                document.score = freshness_score
            else:
                document.score = (
                    (1.0 - self.freshness_weight) * float(base_score)
                    + self.freshness_weight * freshness_score
                )

        base_method = document.metadata.get("retrieval_method")
        document.metadata["retrieval_method"] = "temporal"
        document.metadata[TEMPORAL_PROVENANCE_KEY] = {
            "timestamp": _optional_iso(candidate.timestamp),
            "timestamp_status": "verified" if candidate.timestamp is not None else "missing_allowed",
            "valid_from": _optional_iso(candidate.valid_from),
            "valid_to": _optional_iso(candidate.valid_to),
            "as_of": _optional_iso(as_of),
            "start": _optional_iso(start),
            "end": _optional_iso(end),
            "base_retrieval_method": base_method,
            "base_score": base_score,
            "freshness_score": freshness_score,
            "freshness_reference": _optional_iso(reference) if freshness_score is not None else None,
            "freshness_half_life_seconds": self.freshness_half_life_seconds,
            "freshness_weight": self.freshness_weight if freshness_score is not None else None,
            "final_score": document.score,
            "missing_timestamp_policy": self.missing_timestamp.value,
            "version_id": candidate.version_id,
            "version_group": candidate.version_group,
            "conflict_resolution": self.conflict_resolution.value,
        }

    def _resolve_versions(self, candidates: list[_TemporalCandidate]) -> list[_TemporalCandidate]:
        groups: dict[str, list[_TemporalCandidate]] = {}
        ungrouped: list[_TemporalCandidate] = []
        for candidate in candidates:
            if candidate.version_group is None:
                ungrouped.append(candidate)
            else:
                groups.setdefault(candidate.version_group, []).append(candidate)

        selected = list(ungrouped)
        for candidate in ungrouped:
            _annotate_version_selection(candidate, candidates=[candidate], selected=True)

        for group_name, versions in groups.items():
            if len(versions) > 1 and self.conflict_resolution is ConflictResolution.RAISE:
                version_ids = [candidate.version_id or candidate.document.doc_id for candidate in versions]
                raise ValueError(
                    f"multiple versions retrieved for group {group_name!r}: {version_ids!r}"
                )
            if self.conflict_resolution is ConflictResolution.ALL:
                selected.extend(versions)
                for candidate in versions:
                    _annotate_version_selection(candidate, candidates=versions, selected=True)
                continue
            if self.conflict_resolution is ConflictResolution.HIGHEST_SCORE:
                winner = max(versions, key=_score_then_time_key)
            else:
                winner = max(versions, key=_latest_then_score_key)
            _annotate_version_selection(winner, candidates=versions, selected=True)
            selected.append(winner)
        return selected


def _annotate_version_selection(
    candidate: _TemporalCandidate,
    *,
    candidates: list[_TemporalCandidate],
    selected: bool,
) -> None:
    provenance = candidate.document.metadata[TEMPORAL_PROVENANCE_KEY]
    provenance["versions_considered"] = len(candidates)
    provenance["selected"] = selected
    provenance["discarded_version_ids"] = [
        other.version_id or other.document.doc_id
        for other in candidates
        if other is not candidate
    ]


def _matches_window(
    candidate: _TemporalCandidate,
    *,
    as_of: datetime | None,
    start: datetime | None,
    end: datetime | None,
) -> bool:
    timestamp = candidate.timestamp
    if as_of is not None:
        if timestamp is not None and timestamp > as_of:
            return False
        if candidate.valid_from is not None and candidate.valid_from > as_of:
            return False
        if candidate.valid_to is not None and as_of >= candidate.valid_to:
            return False
        return True
    if start is not None and timestamp is not None and timestamp < start:
        return False
    if end is not None and timestamp is not None and timestamp > end:
        return False
    if start is not None and candidate.valid_to is not None and candidate.valid_to <= start:
        return False
    if end is not None and candidate.valid_from is not None and candidate.valid_from > end:
        return False
    return True


def _ranking_key(candidate: _TemporalCandidate) -> tuple[bool, float, int]:
    score = candidate.document.score
    return score is not None, float(score) if score is not None else float("-inf"), -candidate.original_rank


def _latest_then_score_key(candidate: _TemporalCandidate) -> tuple[datetime, bool, float, int]:
    timestamp = candidate.timestamp or datetime.min.replace(tzinfo=timezone.utc)
    score = candidate.document.score
    return timestamp, score is not None, float(score) if score is not None else float("-inf"), -candidate.original_rank


def _score_then_time_key(candidate: _TemporalCandidate) -> tuple[bool, float, datetime, int]:
    score = candidate.document.score
    timestamp = candidate.timestamp or datetime.min.replace(tzinfo=timezone.utc)
    return score is not None, float(score) if score is not None else float("-inf"), timestamp, -candidate.original_rank


def _validate_window(
    *,
    as_of: TemporalValue | None,
    start: TemporalValue | None,
    end: TemporalValue | None,
) -> tuple[datetime | None, datetime | None, datetime | None]:
    point = _optional_datetime(as_of, name="as_of")
    range_start = _optional_datetime(start, name="start")
    range_end = _optional_datetime(end, name="end")
    if point is not None and (range_start is not None or range_end is not None):
        raise ValueError("as_of is mutually exclusive with start/end")
    if range_start is not None and range_end is not None and range_start > range_end:
        raise ValueError("start must be earlier than or equal to end")
    return point, range_start, range_end


def _half_life_seconds(value: timedelta | Real | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, timedelta):
        seconds = value.total_seconds()
    elif isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("freshness_half_life must be a timedelta or number of seconds")
    else:
        seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("freshness_half_life must be finite and > 0")
    return seconds


def _weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("freshness_weight must be a number")
    weight = float(value)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("freshness_weight must be between 0 and 1")
    return weight


def _enum_value(enum_type: type[Enum], value: Enum | str, *, name: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise ValueError(f"{name} must be one of: {choices}") from exc


def _metadata_key(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _identifier(value: str | int, *, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{name} must be a string or integer")
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must not be empty")
    return identifier


def _optional_identifier(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, name=name)


def _isoformat(value: TemporalValue, *, name: str) -> str:
    return _require_datetime(value, name=name).isoformat()


def _optional_iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _optional_datetime(value: Any, *, name: str) -> datetime | None:
    if value is None:
        return None
    return _require_datetime(value, name=name)


def _require_datetime(value: Any, *, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{name} must not be empty")
        if normalized.endswith(("Z", "z")):
            normalized = f"{normalized[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO-8601 datetime") from exc
    else:
        raise TypeError(f"{name} must be a datetime or ISO-8601 string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)
