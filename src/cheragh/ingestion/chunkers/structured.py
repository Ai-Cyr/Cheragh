"""Structure-aware chunkers for markdown, HTML and sentence windows."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import re
from typing import Iterable

from ...base import Document
from ..loaders.text import html_to_text
from .recursive import RecursiveTextChunker


@dataclass
class MarkdownHeaderChunker:
    """Split Markdown documents by headings, then optionally sub-chunk large sections."""

    chunk_size: int = 1000
    chunk_overlap: int = 120
    min_chunk_size: int = 20

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        output: list[Document] = []
        fallback = RecursiveTextChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.min_chunk_size,
        )
        for doc in documents:
            sections = self.split_text(doc.content)
            base_id = doc.doc_id or f"doc-{len(output)}"
            for section_index, section in enumerate(sections):
                section_doc = Document(
                    content=section["content"],
                    metadata={**deepcopy(doc.metadata), "section": section["title"], "heading_level": section["level"]},
                    doc_id=f"{base_id}#section-{section_index}",
                )
                if len(section_doc.content) > self.chunk_size:
                    output.extend(fallback.split_documents([section_doc]))
                elif section_doc.content.strip():
                    section_doc.metadata["chunk_index"] = 0
                    section_doc.metadata["parent_doc_id"] = base_id
                    output.append(section_doc)
        return output

    def split_text(self, text: str) -> list[dict[str, str | int]]:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        sections: list[dict[str, str | int]] = []
        current_title = "document"
        current_level = 0
        current_lines: list[str] = []
        for line in lines:
            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if match:
                if "\n".join(current_lines).strip():
                    sections.append({"title": current_title, "level": current_level, "content": "\n".join(current_lines).strip()})
                current_title = match.group(2).strip()
                current_level = len(match.group(1))
                current_lines = [line]
            else:
                current_lines.append(line)
        if "\n".join(current_lines).strip():
            sections.append({"title": current_title, "level": current_level, "content": "\n".join(current_lines).strip()})
        return sections


@dataclass
class HTMLSectionChunker:
    """Split HTML content by heading tags and strip markup from each section."""

    chunk_size: int = 1000
    chunk_overlap: int = 120
    min_chunk_size: int = 20

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        output: list[Document] = []
        fallback = RecursiveTextChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.min_chunk_size,
        )
        for doc in documents:
            base_id = doc.doc_id or f"doc-{len(output)}"
            sections = self.split_html(doc.content)
            for idx, section in enumerate(sections):
                section_doc = Document(
                    content=section["content"],
                    metadata={**deepcopy(doc.metadata), "section": section["title"], "heading_level": section["level"]},
                    doc_id=f"{base_id}#section-{idx}",
                )
                if len(section_doc.content) > self.chunk_size:
                    output.extend(fallback.split_documents([section_doc]))
                elif section_doc.content.strip():
                    section_doc.metadata["chunk_index"] = 0
                    section_doc.metadata["parent_doc_id"] = base_id
                    output.append(section_doc)
        return output

    def split_html(self, raw_html: str) -> list[dict[str, str | int]]:
        heading_re = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1>", flags=re.I | re.S)
        matches = list(heading_re.finditer(raw_html))
        if not matches:
            text = html_to_text(raw_html)
            return [{"title": "document", "level": 0, "content": text}] if text else []
        sections: list[dict[str, str | int]] = []
        preamble = html_to_text(raw_html[:matches[0].start()]).strip()
        if preamble:
            sections.append({"title": "document", "level": 0, "content": preamble})
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_html)
            title = html_to_text(match.group(2)) or "section"
            content = html_to_text(raw_html[start:end])
            if content.strip():
                sections.append({"title": title, "level": int(match.group(1)), "content": content.strip()})
        return sections


@dataclass
class SentenceWindowChunker:
    """Create chunks from sliding windows of sentences."""

    window_size: int = 5
    window_overlap: int = 1
    min_chunk_size: int = 20

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.window_overlap < 0 or self.window_overlap >= self.window_size:
            raise ValueError("window_overlap must be >= 0 and < window_size")

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        output: list[Document] = []
        step = self.window_size - self.window_overlap
        for doc in documents:
            base_id = doc.doc_id or f"doc-{len(output)}"
            sentences = _split_sentences(doc.content)
            for idx, start in enumerate(range(0, len(sentences), step)):
                chunk = " ".join(sentences[start : start + self.window_size]).strip()
                if len(chunk) >= self.min_chunk_size:
                    output.append(
                        Document(
                            content=chunk,
                            metadata={**doc.metadata, "chunk_index": idx, "parent_doc_id": base_id, "sentence_start": start},
                            doc_id=f"{base_id}#sentwin-{idx}",
                        )
                    )
                if start + self.window_size >= len(sentences):
                    # The window already covered the last sentence; further
                    # windows would only duplicate a suffix of this one.
                    break
        return output


def _split_sentences(text: str) -> list[str]:
    return [text[start:end] for start, end in _sentence_spans(text)]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans for the dependency-free sentence boundary baseline."""
    boundaries = [0, *(match.end() for match in re.finditer(r"(?<=[.!?。！？])\s+", text)), len(text)]
    spans = []
    for start, end in zip(boundaries, boundaries[1:]):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans
