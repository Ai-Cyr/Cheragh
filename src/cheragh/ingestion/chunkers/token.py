"""Token-like chunker based on regex words, dependency-free."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

from ...base import Document


@dataclass
class TokenTextChunker:
    """Chunk text by whitespace-like tokens and preserve citation offsets.

    This is a lightweight approximation. For exact model-token chunking, provide
    your own tokenizer and create chunks before calling the retriever.
    """

    chunk_size: int = 250
    chunk_overlap: int = 40

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    def split_text(self, text: str) -> list[str]:
        return [chunk for chunk, _, _ in self._split_text_with_offsets(text)]

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in documents:
            base_id = doc.doc_id or f"doc-{len(chunks)}"
            for index, (chunk, start, end) in enumerate(self._split_text_with_offsets(doc.content)):
                chunks.append(
                    Document(
                        content=chunk,
                        doc_id=f"{base_id}#tok-{index}",
                        metadata={
                            **doc.metadata,
                            "chunk_index": index,
                            "parent_doc_id": base_id,
                            "chunker": "token",
                            "source_char_start": start,
                            "source_char_end": end,
                        },
                    )
                )
        return chunks

    def _split_text_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        tokens = list(re.finditer(r"\S+", text))
        if not tokens:
            return []
        step = self.chunk_size - self.chunk_overlap
        chunks: list[tuple[str, int, int]] = []
        for i in range(0, len(tokens), step):
            window = tokens[i : i + self.chunk_size]
            start = window[0].start()
            end = window[-1].end()
            chunks.append((text[start:end], start, end))
        return chunks
