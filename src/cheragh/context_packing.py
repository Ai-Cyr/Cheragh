"""Deterministic, budget-aware context packing for long-context RAG.

The packer in this module sits between retrieval and generation.  It selects
retrieved :class:`~cheragh.base.Document` objects, enforces an exact budget
*according to the injected token estimator*, and places the most relevant
evidence near the two context boundaries to reduce lost-in-the-middle risk.

This is context engineering, not a trained long-context reader and not an
implementation of any particular LongRAG model.  The default estimator is a
small dependency-free approximation; applications should inject the target
model's tokenizer when model-exact accounting is required.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
import numbers
import re
from typing import Literal

from .base import Document, _snapshot_document


TokenEstimator = Callable[[str], int]
DocumentFormatter = Callable[[Document], str]
SourceResolver = Callable[[Document], str]
DeduplicationKey = Callable[[Document], Hashable]
PackingOrder = Literal["lost_in_the_middle", "relevance", "input"]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def approximate_token_count(text: str) -> int:
    """Return a deterministic word-and-punctuation token approximation.

    This function deliberately has no model dependency.  It is suitable for
    tests and conservative application-level budgets only when its approximation
    is acceptable.  Pass a model tokenizer (for example ``len(encode(text))``)
    to :class:`LongContextPacker` for model-exact limits.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a str")
    return len(_TOKEN_RE.findall(text))


def format_document_with_citation(document: Document) -> str:
    """Render a document while retaining Cheragh's citation identifier."""

    if document.doc_id:
        return f"[source: {document.doc_id}]\n{document.content}"
    return document.content


def lost_in_the_middle_order(documents: Sequence[Document]) -> list[Document]:
    """Place relevance-ranked documents alternately near both context edges.

    The input is expected to be ordered by decreasing relevance.  For
    ``[A, B, C, D, E]`` the returned order is ``[A, C, E, D, B]``: the two
    strongest items occupy the beginning and the end, respectively.
    Returned documents are snapshots, so neither side owns the other's metadata.
    """

    _validate_documents(documents)
    snapshots = [_snapshot_document(document) for document in documents]
    beginning = snapshots[::2]
    end = list(reversed(snapshots[1::2]))
    return beginning + end


@dataclass(frozen=True)
class DroppedDocument:
    """Diagnostic record for a candidate excluded from the packed context."""

    input_index: int
    doc_id: str | None
    source_id: str
    reason: Literal["duplicate", "empty_content", "token_budget", "source_budget"]
    estimated_tokens: int
    duplicate_of: str | None = None

    def __post_init__(self) -> None:
        _non_negative_int(self.input_index, name="input_index")
        if self.doc_id is not None and not isinstance(self.doc_id, str):
            raise TypeError("doc_id must be a string or None")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if self.reason not in {"duplicate", "empty_content", "token_budget", "source_budget"}:
            raise ValueError("reason must be a supported drop reason")
        _non_negative_int(self.estimated_tokens, name="estimated_tokens")
        if self.duplicate_of is not None and not isinstance(self.duplicate_of, str):
            raise TypeError("duplicate_of must be a string or None")
        if self.reason == "duplicate" and not self.duplicate_of:
            raise ValueError("duplicate drops must identify duplicate_of")
        if self.reason != "duplicate" and self.duplicate_of is not None:
            raise ValueError("duplicate_of is only valid for duplicate drops")
        object.__setattr__(self, "source_id", self.source_id.strip())


@dataclass(frozen=True)
class SourceTokenUsage:
    """Effective usage and optional cap for one resolved source."""

    source_id: str
    tokens: int
    token_budget: int | None
    document_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        tokens = _non_negative_int(self.tokens, name="tokens")
        budget = (
            None
            if self.token_budget is None
            else _non_negative_int(self.token_budget, name="token_budget")
        )
        document_count = _positive_int(self.document_count, name="document_count")
        if budget is not None and tokens > budget:
            raise ValueError("source tokens must not exceed token_budget")
        object.__setattr__(self, "source_id", self.source_id.strip())
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "token_budget", budget)
        object.__setattr__(self, "document_count", document_count)


@dataclass(frozen=True)
class PackingDiagnostics:
    """Immutable selection and budget diagnostics for one packing operation."""

    input_documents: int
    unique_documents: int
    selected_documents: int
    token_budget: int
    tokens_used: int
    remaining_tokens: int
    ordering: PackingOrder
    dropped: tuple[DroppedDocument, ...]
    source_usage: tuple[SourceTokenUsage, ...]
    truncated_document_ids: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        input_documents = _non_negative_int(self.input_documents, name="input_documents")
        unique_documents = _non_negative_int(self.unique_documents, name="unique_documents")
        selected_documents = _non_negative_int(self.selected_documents, name="selected_documents")
        token_budget = _positive_int(self.token_budget, name="token_budget")
        tokens_used = _non_negative_int(self.tokens_used, name="tokens_used")
        remaining_tokens = _non_negative_int(self.remaining_tokens, name="remaining_tokens")
        if not selected_documents <= unique_documents <= input_documents:
            raise ValueError("document counts must satisfy selected <= unique <= input")
        if tokens_used > token_budget:
            raise ValueError("tokens_used must not exceed token_budget")
        if remaining_tokens != token_budget - tokens_used:
            raise ValueError("remaining_tokens must equal token_budget - tokens_used")
        if self.ordering not in {"lost_in_the_middle", "relevance", "input"}:
            raise ValueError("ordering must be 'lost_in_the_middle', 'relevance', or 'input'")
        if not isinstance(self.dropped, tuple) or any(
            not isinstance(item, DroppedDocument) for item in self.dropped
        ):
            raise TypeError("dropped must be a tuple of DroppedDocument values")
        if not isinstance(self.source_usage, tuple) or any(
            not isinstance(item, SourceTokenUsage) for item in self.source_usage
        ):
            raise TypeError("source_usage must be a tuple of SourceTokenUsage values")
        if not isinstance(self.truncated_document_ids, tuple) or any(
            item is not None and not isinstance(item, str)
            for item in self.truncated_document_ids
        ):
            raise TypeError("truncated_document_ids must be a tuple of strings or None")
        if len(self.dropped) != input_documents - selected_documents:
            raise ValueError("dropped count must equal input_documents - selected_documents")
        if self.duplicate_count != input_documents - unique_documents:
            raise ValueError("duplicate drops must account for input_documents - unique_documents")
        if sum(item.document_count for item in self.source_usage) != selected_documents:
            raise ValueError("source_usage document counts must equal selected_documents")
        if any(item.input_index >= input_documents for item in self.dropped):
            raise ValueError("dropped input_index must reference an input document")
        if len(self.truncated_document_ids) > selected_documents:
            raise ValueError("truncated_document_ids cannot exceed selected_documents")

    @property
    def duplicate_count(self) -> int:
        return sum(item.reason == "duplicate" for item in self.dropped)

    @property
    def budget_drop_count(self) -> int:
        return sum(item.reason in {"token_budget", "source_budget"} for item in self.dropped)


@dataclass(frozen=True)
class PackedContext:
    """A rendered context, its source snapshots and complete diagnostics."""

    text: str
    documents: tuple[Document, ...]
    diagnostics: PackingDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.documents, tuple):
            raise TypeError("documents must be a tuple of Document values")
        _validate_documents(self.documents)
        if not isinstance(self.diagnostics, PackingDiagnostics):
            raise TypeError("diagnostics must be PackingDiagnostics")
        if len(self.documents) != self.diagnostics.selected_documents:
            raise ValueError("documents count must match diagnostics.selected_documents")
        if bool(self.text.strip()) != bool(self.documents):
            raise ValueError("text and documents must either both be empty or both be populated")
        object.__setattr__(
            self,
            "documents",
            tuple(_snapshot_document(document) for document in self.documents),
        )
        object.__setattr__(self, "diagnostics", deepcopy(self.diagnostics))

    @property
    def token_count(self) -> int:
        return self.diagnostics.tokens_used

    @property
    def citation_ids(self) -> tuple[str, ...]:
        return tuple(document.doc_id for document in self.documents if document.doc_id is not None)

    def snapshot(self) -> "PackedContext":
        """Return an independent copy of the mutable document payloads."""

        return PackedContext(
            text=self.text,
            documents=tuple(_snapshot_document(document) for document in self.documents),
            diagnostics=deepcopy(self.diagnostics),
        )


@dataclass(frozen=True)
class _Candidate:
    document: Document
    input_index: int
    source_id: str
    fragment: str
    estimated_tokens: int
    truncated: bool = False


class LongContextPacker:
    """Pack retrieved documents into a strict, source-balanced token budget.

    Parameters
    ----------
    token_budget:
        Maximum tokens in the final rendered ``PackedContext.text``.  The
        separator and citation headers are included in this limit.
    token_estimator:
        Deterministic callable returning a non-negative integer.  Strictness is
        relative to this estimator; inject the generation model's tokenizer for
        model-exact accounting.
    per_source_token_budget:
        Either one cap applied to every source, or a mapping of source ids to
        caps.  Missing mapping entries are uncapped except by ``token_budget``.
    source_resolver:
        Optional source-id resolver.  The default uses ``metadata['source']``,
        then ``metadata['parent_doc_id']``, then ``doc_id``.  Anonymous documents
        are kept in separate synthetic sources.
    formatter:
        Converts a document into the exact fragment sent to the model.  The
        default emits Cheragh's ``[source: id]`` marker when ``doc_id`` exists.
    ordering:
        ``lost_in_the_middle`` alternates relevance-ranked evidence between the
        beginning and end; ``relevance`` retains score order; ``input`` restores
        retrieval input order after score-aware selection.
    truncate_oversized:
        If true, retain the longest fitting prefix of a high-ranked candidate.
        Character provenance is adjusted when standard offset metadata exists.
        This requires a deterministic estimator whose prefix counts are
        non-decreasing, as normal tokenizers are.

    Notes
    -----
    Selection is greedy by descending score (unscored documents come last),
    stable for ties, and continues after an over-budget candidate so smaller
    lower-ranked evidence can still fit.  Packing does not train or replace a
    long-context reader.
    """

    def __init__(
        self,
        token_budget: int,
        *,
        token_estimator: TokenEstimator = approximate_token_count,
        per_source_token_budget: int | Mapping[str, int] | None = None,
        source_resolver: SourceResolver | None = None,
        formatter: DocumentFormatter = format_document_with_citation,
        separator: str = "\n\n",
        deduplicate: bool = True,
        deduplication_key: DeduplicationKey | None = None,
        ordering: PackingOrder = "lost_in_the_middle",
        truncate_oversized: bool = False,
    ) -> None:
        self.token_budget = _positive_int(token_budget, name="token_budget")
        if not callable(token_estimator):
            raise TypeError("token_estimator must be callable")
        if source_resolver is not None and not callable(source_resolver):
            raise TypeError("source_resolver must be callable or None")
        if not callable(formatter):
            raise TypeError("formatter must be callable")
        if not isinstance(separator, str):
            raise TypeError("separator must be a str")
        if not isinstance(deduplicate, bool):
            raise TypeError("deduplicate must be a bool")
        if deduplication_key is not None and not callable(deduplication_key):
            raise TypeError("deduplication_key must be callable or None")
        if ordering not in {"lost_in_the_middle", "relevance", "input"}:
            raise ValueError("ordering must be 'lost_in_the_middle', 'relevance', or 'input'")
        if not isinstance(truncate_oversized, bool):
            raise TypeError("truncate_oversized must be a bool")

        self.token_estimator = token_estimator
        self.per_source_token_budget = _validate_source_budgets(per_source_token_budget)
        self.source_resolver = source_resolver
        self.formatter = formatter
        self.separator = separator
        self.deduplicate = deduplicate
        self.deduplication_key = deduplication_key or _default_deduplication_key
        self.ordering = ordering
        self.truncate_oversized = truncate_oversized

        # Validate the estimator immediately rather than failing halfway through
        # a caller's first request.
        self._estimate("")

    def __call__(self, documents: Sequence[Document]) -> PackedContext:
        return self.pack(documents)

    def pack(self, documents: Sequence[Document]) -> PackedContext:
        """Select, order and render independent snapshots of ``documents``."""

        _validate_documents(documents)
        ranked = self._prepare_candidates(documents)
        unique, duplicate_drops = self._deduplicate(ranked)
        selected: list[_Candidate] = []
        dropped = list(duplicate_drops)

        for candidate in unique:
            if not candidate.document.content.strip() or not candidate.fragment.strip():
                dropped.append(self._drop(candidate, "empty_content"))
                continue

            fits, failure_reason = self._fits(selected + [candidate])
            if fits:
                selected.append(candidate)
                continue

            truncated = self._truncate_to_fit(candidate, selected) if self.truncate_oversized else None
            if truncated is not None:
                selected.append(truncated)
                continue

            dropped.append(self._drop(candidate, failure_reason))

        ordered = self._order(selected)
        text = self.separator.join(candidate.fragment for candidate in ordered)
        tokens_used = self._estimate(text)
        # This assertion guards future changes to selection/order logic.  It is
        # deliberately a runtime error rather than silently returning overflow.
        if tokens_used > self.token_budget:  # pragma: no cover - defensive invariant
            raise RuntimeError("internal error: packed context exceeds token_budget")

        usage = self._source_usage(ordered)
        diagnostics = PackingDiagnostics(
            input_documents=len(documents),
            unique_documents=len(unique),
            selected_documents=len(ordered),
            token_budget=self.token_budget,
            tokens_used=tokens_used,
            remaining_tokens=self.token_budget - tokens_used,
            ordering=self.ordering,
            dropped=tuple(dropped),
            source_usage=tuple(
                SourceTokenUsage(
                    source_id=source_id,
                    tokens=source_tokens,
                    token_budget=self._source_limit(source_id),
                    document_count=sum(item.source_id == source_id for item in ordered),
                )
                for source_id, source_tokens in usage.items()
            ),
            truncated_document_ids=tuple(item.document.doc_id for item in ordered if item.truncated),
        )
        return PackedContext(
            text=text,
            documents=tuple(_snapshot_document(item.document) for item in ordered),
            diagnostics=diagnostics,
        )

    def _prepare_candidates(self, documents: Sequence[Document]) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for index, source_document in enumerate(documents):
            document = _snapshot_document(source_document)
            _score(document.score, index=index)
            source_id = self._resolve_source(document, index)
            fragment = self._format(document)
            candidates.append(
                _Candidate(
                    document=document,
                    input_index=index,
                    source_id=source_id,
                    fragment=fragment,
                    estimated_tokens=self._estimate(fragment),
                )
            )
        return sorted(candidates, key=_relevance_sort_key)

    def _deduplicate(
        self, candidates: Sequence[_Candidate]
    ) -> tuple[list[_Candidate], list[DroppedDocument]]:
        if not self.deduplicate:
            return list(candidates), []

        unique: list[_Candidate] = []
        dropped: list[DroppedDocument] = []
        seen: dict[Hashable, _Candidate] = {}
        for candidate in candidates:
            key = self.deduplication_key(_snapshot_document(candidate.document))
            try:
                hash(key)
            except TypeError as exc:
                raise TypeError("deduplication_key must return a hashable value") from exc
            retained = seen.get(key)
            if retained is None:
                seen[key] = candidate
                unique.append(candidate)
                continue
            dropped.append(
                DroppedDocument(
                    input_index=candidate.input_index,
                    doc_id=candidate.document.doc_id,
                    source_id=candidate.source_id,
                    reason="duplicate",
                    estimated_tokens=candidate.estimated_tokens,
                    duplicate_of=_document_label(retained),
                )
            )
        return unique, dropped

    def _fits(
        self, candidates: Sequence[_Candidate]
    ) -> tuple[bool, Literal["token_budget", "source_budget"]]:
        ordered = self._order(candidates)
        for source_id, tokens in self._source_usage(ordered).items():
            limit = self._source_limit(source_id)
            if limit is not None and tokens > limit:
                return False, "source_budget"
        rendered = self.separator.join(candidate.fragment for candidate in ordered)
        if self._estimate(rendered) > self.token_budget:
            return False, "token_budget"
        return True, "token_budget"

    def _truncate_to_fit(
        self, candidate: _Candidate, selected: Sequence[_Candidate]
    ) -> _Candidate | None:
        content = candidate.document.content
        if not content:
            return None

        low = 1
        high = len(content)
        best: _Candidate | None = None
        while low <= high:
            midpoint = (low + high) // 2
            truncated = self._truncated_candidate(candidate, content[:midpoint].rstrip())
            fits = truncated is not None and self._fits([*selected, truncated])[0]
            if fits:
                best = truncated
                low = midpoint + 1
            else:
                high = midpoint - 1
        return best

    def _truncated_candidate(self, candidate: _Candidate, content: str) -> _Candidate | None:
        if not content:
            return None
        document = _snapshot_document(candidate.document)
        original_length = len(document.content)
        document.content = content
        metadata = deepcopy(document.metadata or {})
        existing = metadata.get("context_packing")
        packing_metadata = deepcopy(existing) if isinstance(existing, dict) else {}
        packing_metadata.update(
            {
                "truncated": True,
                "original_characters": original_length,
                "retained_characters": len(content),
            }
        )
        metadata["context_packing"] = packing_metadata
        _adjust_end_offset(metadata, "source_char_start", "source_char_end", len(content))
        _adjust_end_offset(metadata, "char_start", "char_end", len(content))
        document.metadata = metadata
        fragment = self._format(document)
        return _Candidate(
            document=document,
            input_index=candidate.input_index,
            source_id=candidate.source_id,
            fragment=fragment,
            estimated_tokens=self._estimate(fragment),
            truncated=True,
        )

    def _order(self, candidates: Sequence[_Candidate]) -> list[_Candidate]:
        if self.ordering == "relevance":
            return list(candidates)
        if self.ordering == "input":
            return sorted(candidates, key=lambda candidate: candidate.input_index)
        return list(candidates[::2]) + list(reversed(candidates[1::2]))

    def _source_usage(self, ordered: Sequence[_Candidate]) -> dict[str, int]:
        fragments: dict[str, list[str]] = {}
        for candidate in ordered:
            fragments.setdefault(candidate.source_id, []).append(candidate.fragment)
        return {
            source_id: self._estimate(self.separator.join(source_fragments))
            for source_id, source_fragments in fragments.items()
        }

    def _source_limit(self, source_id: str) -> int | None:
        configured = self.per_source_token_budget
        if isinstance(configured, int):
            return configured
        if configured is not None:
            return configured.get(source_id)
        return None

    def _resolve_source(self, document: Document, index: int) -> str:
        if self.source_resolver is not None:
            source_id = self.source_resolver(_snapshot_document(document))
            if not isinstance(source_id, str):
                raise TypeError("source_resolver must return a str")
            if not source_id.strip():
                raise ValueError("source_resolver must return a non-empty str")
            return source_id.strip()

        metadata = document.metadata or {}
        for value in (metadata.get("source"), metadata.get("parent_doc_id"), document.doc_id):
            if value is not None and str(value).strip():
                return str(value).strip()
        return f"__anonymous__:{index}"

    def _format(self, document: Document) -> str:
        fragment = self.formatter(_snapshot_document(document))
        if not isinstance(fragment, str):
            raise TypeError("formatter must return a str")
        return fragment

    def _estimate(self, text: str) -> int:
        value = self.token_estimator(text)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("token_estimator must return an int")
        if value < 0:
            raise ValueError("token_estimator must return a non-negative value")
        return value

    @staticmethod
    def _drop(
        candidate: _Candidate,
        reason: Literal["empty_content", "token_budget", "source_budget"],
    ) -> DroppedDocument:
        return DroppedDocument(
            input_index=candidate.input_index,
            doc_id=candidate.document.doc_id,
            source_id=candidate.source_id,
            reason=reason,
            estimated_tokens=candidate.estimated_tokens,
        )


# A shorter name for applications that treat packing independently of LongRAG.
ContextPacker = LongContextPacker


def pack_context(
    documents: Sequence[Document],
    token_budget: int,
    **kwargs: object,
) -> PackedContext:
    """Convenience wrapper around :class:`LongContextPacker`.

    Advanced callers should instantiate the packer directly for static typing of
    custom strategies; this helper is useful for one-shot packing.
    """

    return LongContextPacker(token_budget, **kwargs).pack(documents)  # type: ignore[arg-type]


def _default_deduplication_key(document: Document) -> Hashable:
    return " ".join(document.content.split()).casefold()


def _relevance_sort_key(candidate: _Candidate) -> tuple[int, float, int]:
    if candidate.document.score is None:
        return (1, 0.0, candidate.input_index)
    return (0, -float(candidate.document.score), candidate.input_index)


def _score(value: float | None, *, index: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"documents[{index}].score must be a real number or None")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"documents[{index}].score must be finite")
    return numeric


def _validate_documents(documents: Sequence[Document]) -> None:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise TypeError("documents must be a Sequence[Document]")
    for index, document in enumerate(documents):
        if not isinstance(document, Document):
            raise TypeError(f"documents[{index}] must be a Document")
        if not isinstance(document.content, str):
            raise TypeError(f"documents[{index}].content must be a str")
        if not isinstance(document.metadata, dict):
            raise TypeError(f"documents[{index}].metadata must be a dict")
        if document.doc_id is not None and not isinstance(document.doc_id, str):
            raise TypeError(f"documents[{index}].doc_id must be a str or None")
        _score(document.score, index=index)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _validate_source_budgets(
    value: int | Mapping[str, int] | None,
) -> int | dict[str, int] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("per_source_token_budget must be an int, mapping, or None")
    if isinstance(value, int):
        return _non_negative_int(value, name="per_source_token_budget")
    if not isinstance(value, Mapping):
        raise TypeError("per_source_token_budget must be an int, mapping, or None")

    budgets: dict[str, int] = {}
    for source_id, budget in value.items():
        if not isinstance(source_id, str):
            raise TypeError("per_source_token_budget keys must be str")
        if not source_id.strip():
            raise ValueError("per_source_token_budget keys must be non-empty")
        normalized_source_id = source_id.strip()
        if normalized_source_id in budgets:
            raise ValueError("per_source_token_budget contains duplicate normalized source ids")
        budgets[normalized_source_id] = _non_negative_int(
            budget,
            name=f"per_source_token_budget[{normalized_source_id!r}]",
        )
    return budgets


def _document_label(candidate: _Candidate) -> str:
    return candidate.document.doc_id or f"input:{candidate.input_index}"


def _adjust_end_offset(metadata: dict[str, object], start_key: str, end_key: str, length: int) -> None:
    start = metadata.get(start_key)
    end = metadata.get(end_key)
    if (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
    ):
        metadata[end_key] = min(end, start + length)


__all__ = [
    "ContextPacker",
    "DeduplicationKey",
    "DocumentFormatter",
    "DroppedDocument",
    "LongContextPacker",
    "PackedContext",
    "PackingDiagnostics",
    "PackingOrder",
    "SourceResolver",
    "SourceTokenUsage",
    "TokenEstimator",
    "approximate_token_count",
    "format_document_with_citation",
    "lost_in_the_middle_order",
    "pack_context",
]
