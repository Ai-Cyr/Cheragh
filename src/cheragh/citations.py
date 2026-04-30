"""Citation extraction and lightweight grounding checks."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Sequence

from .base import Document

CITATION_PATTERN = re.compile(r"\[source:\s*([^\]]+)\]", flags=re.I)


@dataclass
class CitationValidationResult:
    citations: list[str]
    known_doc_ids: list[str]
    unknown_citations: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    unsourced_claims: list[str] = field(default_factory=list)
    grounded_score: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unknown_citations and not self.missing_citations and not self.unsourced_claims


def extract_citations(answer: str) -> list[str]:
    """Extract citation ids in ``[source: doc_id]`` format."""
    return [citation.strip() for citation in CITATION_PATTERN.findall(answer)]


def validate_citations(
    answer: str,
    documents: Sequence[Document],
    require_citations: bool = False,
    flag_unsourced_sentences: bool = False,
) -> CitationValidationResult:
    """Validate answer citations against retrieved documents.

    This is intentionally lightweight and deterministic. It does not claim full
    factual verification; it catches missing/unknown citations and optionally
    sentences that do not contain any citation marker.
    """
    citations = extract_citations(answer)
    known_ids = [doc.doc_id for doc in documents if doc.doc_id]
    known_set = set(known_ids)
    unknown = [citation for citation in citations if citation not in known_set]
    missing: list[str] = []
    warnings: list[str] = []
    if require_citations and documents and not citations:
        missing = known_ids[:]
        warnings.append("missing_citations")
    if unknown:
        warnings.append("unknown_citations")
    unsourced: list[str] = []
    if flag_unsourced_sentences:
        unsourced = _unsourced_sentences(answer)
        if unsourced:
            warnings.append("unsourced_claims")
    grounded_score = citation_coverage(citations, known_set)
    return CitationValidationResult(
        citations=citations,
        known_doc_ids=known_ids,
        unknown_citations=unknown,
        missing_citations=missing,
        unsourced_claims=unsourced,
        grounded_score=grounded_score,
        warnings=warnings,
    )


def citation_coverage(citations: Iterable[str], known_doc_ids: Iterable[str]) -> float:
    known = set(known_doc_ids)
    if not known:
        return 1.0
    cited = set(citations)
    return len(cited & known) / len(known)


def _unsourced_sentences(answer: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    unsourced: list[str] = []
    for sentence in sentences:
        normalized = sentence.lower()
        if normalized.startswith(("je ne sais pas", "i don't know", "i do not know")):
            continue
        if not CITATION_PATTERN.search(sentence):
            unsourced.append(sentence)
    return unsourced


def citation_location(document: Document) -> str:
    """Return a compact human-readable citation location for a document chunk.

    The function reads common ingestion metadata and is safe to call with any
    ``Document``. Examples: ``page=3; chars=120-260`` or ``lines=10-20``.
    """

    metadata = document.metadata or {}
    parts: list[str] = []
    if metadata.get("source"):
        parts.append(f"source={metadata['source']}")
    if metadata.get("page") is not None:
        parts.append(f"page={metadata['page']}")
    if metadata.get("line_start") is not None and metadata.get("line_end") is not None:
        parts.append(f"lines={metadata['line_start']}-{metadata['line_end']}")
    if metadata.get("source_char_start") is not None and metadata.get("source_char_end") is not None:
        parts.append(f"chars={metadata['source_char_start']}-{metadata['source_char_end']}")
    elif metadata.get("char_start") is not None and metadata.get("char_end") is not None:
        parts.append(f"chars={metadata['char_start']}-{metadata['char_end']}")
    return "; ".join(parts)
