"""Recursive text chunking utilities with citation offsets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

from ...base import Document


@dataclass(frozen=True)
class TextChunk:
    """Chunk text plus character offsets in the original document content."""

    text: str
    source_char_start: int
    source_char_end: int
    normalized_char_start: int
    normalized_char_end: int


@dataclass
class RecursiveTextChunker:
    """Split documents into overlapping chunks.

    The splitter prefers semantic separators first (paragraphs, line breaks,
    sentences, then spaces) and falls back to fixed-size character windows.

    v0.9 adds citation offsets. Every chunk produced by ``split_documents`` has
    ``source_char_start`` and ``source_char_end`` metadata. These offsets are
    exclusive-end character spans in the original ``Document.content``.
    """

    chunk_size: int = 800
    chunk_overlap: int = 120
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")
    keep_separator: bool = True
    min_chunk_size: int = 20
    metadata_key: str = "chunk_index"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split_text(self, text: str) -> list[str]:
        return [chunk.text for chunk in self.split_text_with_offsets(text)]

    def split_text_with_offsets(self, text: str) -> list[TextChunk]:
        clean, mapping = _normalize_whitespace_with_mapping(text)
        if not clean:
            return []
        pieces = self._split_recursive(clean, self.separators)
        chunks = [chunk.strip() for chunk in self._merge_pieces(pieces) if chunk.strip()]
        kept = [chunk for chunk in chunks if len(chunk) >= self.min_chunk_size]
        final_chunks = kept or chunks[:1]
        return _locate_chunks(clean, mapping, final_chunks, overlap_hint=self.chunk_overlap)

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in documents:
            base_id = doc.doc_id or f"doc-{len(chunks)}"
            for index, chunk in enumerate(self.split_text_with_offsets(doc.content)):
                metadata = dict(doc.metadata)
                metadata[self.metadata_key] = index
                metadata["parent_doc_id"] = base_id
                metadata["source_char_start"] = chunk.source_char_start
                metadata["source_char_end"] = chunk.source_char_end
                metadata["normalized_char_start"] = chunk.normalized_char_start
                metadata["normalized_char_end"] = chunk.normalized_char_end
                metadata["chunker"] = "recursive"
                chunks.append(
                    Document(
                        content=chunk.text,
                        metadata=metadata,
                        doc_id=f"{base_id}#chunk-{index}",
                    )
                )
        return chunks

    def _split_recursive(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        sep = separators[0]
        parts = text.split(sep)
        if len(parts) == 1:
            return self._split_recursive(text, separators[1:])

        pieces: list[str] = []
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            piece = part + sep if self.keep_separator and i < len(parts) - 1 else part
            if len(piece) > self.chunk_size:
                pieces.extend(self._split_recursive(piece, separators[1:]))
            else:
                pieces.append(piece)
        return pieces

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece)
            if current and current_len + piece_len > self.chunk_size:
                chunks.append("".join(current).strip())
                overlap_text = _tail_text(chunks[-1], self.chunk_overlap)
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)
            current.append(piece)
            current_len += piece_len

        if current:
            chunks.append("".join(current).strip())
        return chunks


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    **kwargs,
) -> list[Document]:
    """Convenience wrapper around :class:`RecursiveTextChunker`."""
    return RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs).split_documents(documents)


def _normalize_whitespace(text: str) -> str:
    return _normalize_whitespace_with_mapping(text)[0]


def _normalize_whitespace_with_mapping(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    mapping: list[int] = []
    i = 0
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    while i < len(text):
        ch = text[i]
        if ch in " \t":
            start = i
            while i < len(text) and text[i] in " \t":
                i += 1
            normalized_chars.append(" ")
            mapping.append(start)
            continue
        normalized_chars.append(ch)
        mapping.append(i)
        i += 1

    raw = "".join(normalized_chars)
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw.rstrip())
    clean = raw[leading:trailing]
    return clean, mapping[leading:trailing]


def _locate_chunks(clean: str, mapping: list[int], chunks: list[str], overlap_hint: int = 0) -> list[TextChunk]:
    located: list[TextChunk] = []
    search_from = 0
    for chunk in chunks:
        needle = chunk.strip()
        if not needle:
            continue
        pos = clean.find(needle, max(0, search_from - overlap_hint - 8))
        if pos < 0:
            pos = clean.find(needle)
        if pos < 0:
            # Last-resort span preserves content even if whitespace normalization
            # made exact matching impossible.
            pos = min(search_from, max(len(clean) - len(needle), 0))
        end = min(pos + len(needle), len(mapping))
        if end <= pos:
            continue
        source_start = mapping[pos]
        source_end = mapping[end - 1] + 1
        located.append(
            TextChunk(
                text=needle,
                source_char_start=source_start,
                source_char_end=source_end,
                normalized_char_start=pos,
                normalized_char_end=end,
            )
        )
        search_from = max(pos + 1, end)
    return located


def _tail_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or not text:
        return ""
    tail = text[-max_chars:]
    first_space = tail.find(" ")
    if first_space > 0:
        tail = tail[first_space + 1 :]
    return tail
