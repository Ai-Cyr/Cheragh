"""Sentence and paragraph packing with exact source spans and bounded chunks."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from copy import deepcopy
from dataclasses import dataclass
import re
from typing import ClassVar, Iterable

from ...base import Document, _validate_non_negative_int, _validate_top_k
from .recursive import TextChunk


_SENTENCE_END = re.compile(r"(?P<punct>[.!?…。！？]+)[\"'”’»\)\]\}」』】》]*")
_LAST_WORD = re.compile(r"([^\W\d_]+(?:\.[^\W\d_]+)*)\.$", flags=re.UNICODE)
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
    "mme", "mlle", "mm", "mes", "pr", "cf", "etc", "e.g", "i.e",
    "fig", "vol", "no", "pp",
})
_LINE = re.compile(r"[^\r\n]*(?:\r\n|\r(?!\n)|\n|$)")


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentence_spans(text: str, *, max_spans: int | None = None) -> list[tuple[int, int]]:
    """A deterministic punctuation heuristic, not a linguistic tokenizer."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        punctuation = match.group("punct")
        # CJK full stops/question marks do not require intervening spaces.
        # For Western punctuation, avoid decimal numbers and embedded dots in
        # words, domains and abbreviations by requiring whitespace or EOF.
        if end < len(text) and not text[end].isspace() and not any(ch in punctuation for ch in "。！？"):
            continue
        if punctuation == "." and end < len(text):
            period_end = match.start() + 1
            # Inspect a bounded prefix rather than copying the entire prefix
            # for each sentence in a long document.
            word_match = _LAST_WORD.search(text[max(0, period_end - 64):period_end])
            if word_match is not None:
                word = word_match.group(1)
                if (
                    word.casefold() in _ABBREVIATIONS
                    or (len(word) == 1 and word.isupper())
                    or re.fullmatch(r"(?:[^\W\d_]\.)+[^\W\d_]", word, flags=re.UNICODE)
                ):
                    continue
        left, right = _trim_span(text, start, end)
        if left < right:
            if max_spans is not None and len(spans) >= max_spans:
                raise ValueError(f"Semantic chunking exceeds max_sentences={max_spans}")
            spans.append((left, right))
        start = end
    left, right = _trim_span(text, start, len(text))
    if left < right:
        if max_spans is not None and len(spans) >= max_spans:
            raise ValueError(f"Semantic chunking exceeds max_sentences={max_spans}")
        spans.append((left, right))
    return spans


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Separate paragraphs at blank lines, preserving internal line endings."""
    spans: list[tuple[int, int]] = []
    start = 0
    for line in _LINE.finditer(text):
        if line.start() == line.end():
            continue
        if not line.group().strip():
            left, right = _trim_span(text, start, line.start())
            if left < right:
                spans.append((left, right))
            start = line.end()
    left, right = _trim_span(text, start, len(text))
    if left < right:
        spans.append((left, right))
    return spans


def _pack_spans(
    text: str, spans: list[tuple[int, int]], chunk_size: int, chunk_overlap: int,
    *, max_chunks: int | None = None,
) -> list[TextChunk]:
    """Pack whole units; overlap only units fitting the requested character budget.

    Oversized units use character windows. Binary search chooses each window's
    end and retained context without repeatedly scanning overlapping units.
    Every next packed window must contain a previously unseen unit.
    """
    chunks: list[TextChunk] = []
    starts = [start for start, _ in spans]
    ends = [end for _, end in spans]

    def append(start: int, end: int) -> None:
        start, end = _trim_span(text, start, end)
        if start >= end:
            return
        if chunks:
            # Trimming a window that advances through whitespace can leave
            # an already-covered suffix, or grow the same source prefix.
            # Discard the former and replace the latter instead of emitting
            # redundant chunks with identical start offsets.
            if end <= chunks[-1].source_char_end:
                return
            if start == chunks[-1].source_char_start:
                chunks.pop()
        if max_chunks is not None and len(chunks) >= max_chunks:
            raise ValueError(f"Chunking exceeds max_chunks_per_document={max_chunks}")
        # No whitespace normalization is performed: both coordinate systems
        # point into the original content and the chunk is its exact slice.
        chunks.append(TextChunk(text[start:end], start, end, start, end))

    index = 0
    while index < len(spans):
        start, end = spans[index]
        if end - start > chunk_size:
            cursor = start
            while cursor < end:
                stop = min(cursor + chunk_size, end)
                append(cursor, stop)
                if stop == end:
                    break
                cursor = stop - chunk_overlap
            index += 1
            continue

        last = bisect_right(ends, start + chunk_size, lo=index) - 1
        append(start, ends[last])
        if last == len(spans) - 1:
            break
        # Whole-unit overlap is optional context: reduce it when retaining it
        # would prevent the next unseen unit from fitting in a bounded chunk.
        lower_bound = max(ends[last] - chunk_overlap, ends[last + 1] - chunk_size)
        index = bisect_left(starts, lower_bound, lo=index + 1, hi=last + 1)
    return chunks


@dataclass
class _BoundaryChunker:
    chunk_size: int = 800
    chunk_overlap: int = 120
    _kind: ClassVar[str]

    def __post_init__(self) -> None:
        _validate_top_k(self.chunk_size, name="chunk_size")
        _validate_non_negative_int(self.chunk_overlap, name="chunk_overlap")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def _spans(self, text: str) -> list[tuple[int, int]]:
        raise NotImplementedError

    def split_text(self, text: str) -> list[str]:
        return [chunk.text for chunk in self.split_text_with_offsets(text)]

    def split_text_with_offsets(self, text: str) -> list[TextChunk]:
        return _pack_spans(text, self._spans(text), self.chunk_size, self.chunk_overlap)

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        output: list[Document] = []
        for document_number, document in enumerate(documents):
            base_id = document.doc_id if document.doc_id is not None else f"doc-{document_number}"
            for index, chunk in enumerate(self.split_text_with_offsets(document.content)):
                metadata = deepcopy(document.metadata)
                metadata.update({
                    "chunk_index": index,
                    "parent_doc_id": base_id,
                    "chunker": self._kind,
                    "source_char_start": chunk.source_char_start,
                    "source_char_end": chunk.source_char_end,
                    "normalized_char_start": chunk.normalized_char_start,
                    "normalized_char_end": chunk.normalized_char_end,
                })
                output.append(Document(chunk.text, metadata=metadata, doc_id=f"{base_id}#{self._kind}-{index}"))
        return output


@dataclass
class SentenceTextChunker(_BoundaryChunker):
    """Pack complete sentences within a hard character limit.

    Boundary detection is a dependency-free punctuation heuristic with common
    French/English abbreviations, decimal numbers, closing quotes and CJK
    punctuation. It is not a trained linguistic sentence segmenter. Oversized
    sentences fall back to exact character windows. ``chunk_overlap`` is a
    maximum character budget for whole-sentence overlap; less context is kept
    when a complete sentence does not fit or would prevent forward progress.
    Every non-whitespace source character is retained, including short tails.
    """

    _kind: ClassVar[str] = "sentence"

    def _spans(self, text: str) -> list[tuple[int, int]]:
        return _sentence_spans(text)


@dataclass
class ParagraphChunker(_BoundaryChunker):
    """Pack blank-line-separated paragraphs within a hard character limit.

    Single line breaks and CRLF sequences inside paragraphs are preserved.
    ``chunk_overlap`` is a maximum character budget for complete-paragraph
    overlap. Oversized paragraphs use character windows with exact overlap;
    no non-whitespace content or short final paragraph is discarded.
    """

    _kind: ClassVar[str] = "paragraph"

    def _spans(self, text: str) -> list[tuple[int, int]]:
        return _paragraph_spans(text)
