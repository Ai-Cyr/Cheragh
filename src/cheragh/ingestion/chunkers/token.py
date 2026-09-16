"""Token-like chunker based on regex words, dependency-free."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
from copy import deepcopy
import re
from numbers import Integral

from ...base import Document, _validate_top_k, _validate_non_negative_int


@dataclass
class TokenTextChunker:
    """Chunk text by whitespace-like tokens and preserve citation offsets.

    With ``tokenizer=None`` this counts whitespace-delimited words. Supply a
    fast model tokenizer to use exact subword offsets and verify every rendered
    chunk fits ``chunk_size`` tokens (excluding the model's special tokens).
    """

    chunk_size: int = 250
    chunk_overlap: int = 40
    tokenizer: Any | None = None

    def __post_init__(self) -> None:
        _validate_top_k(self.chunk_size, name="chunk_size")
        _validate_non_negative_int(self.chunk_overlap, name="chunk_overlap")
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
                            **deepcopy(doc.metadata),
                            "chunk_index": index,
                            "parent_doc_id": base_id,
                            "chunker": "model-token" if self.tokenizer is not None else "token",
                            "source_char_start": start,
                            "source_char_end": end,
                        },
                    )
                )
        return chunks

    def _split_text_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        if self.tokenizer is not None:
            return self._model_token_spans(text)
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
            if i + self.chunk_size >= len(tokens):
                # The window already reached the last token; a further step
                # would only re-emit a suffix of this chunk.
                break
        return chunks

    def _model_token_spans(self, text: str) -> list[tuple[str, int, int]]:
        assert self.tokenizer is not None
        encoded = self.tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise ValueError("Exact token chunking requires a tokenizer with character offset mappings")
        if any(isinstance(value, bool) or not isinstance(value, Integral) for pair in offsets for value in pair):
            raise ValueError("Tokenizer offsets must be integer character indices")
        if any(start < 0 or end < start or end > len(text) for start, end in offsets):
            raise ValueError("Tokenizer returned offsets outside the source text")
        offsets = [(int(start), int(end)) for start, end in offsets if end > start]
        if any(start < previous_start or end < previous_end
               for (previous_start, previous_end), (start, end) in zip(offsets, offsets[1:])):
            raise ValueError("Tokenizer offsets must follow source order")
        chunks: list[tuple[str, int, int]] = []
        index = 0
        while index < len(offsets):
            stop = min(index + self.chunk_size, len(offsets))
            start, end = offsets[index][0], offsets[stop - 1][1]
            # Retokenizing a character span can differ at its first boundary,
            # especially for byte-level BPE. Shrink until the actual count fits.
            while len(self.tokenizer(text[start:end], add_special_tokens=False, truncation=False)["input_ids"]) > self.chunk_size:
                stop -= 1
                if stop <= index or offsets[stop - 1][1] <= start:
                    raise ValueError("A source character needs more tokens than chunk_size")
                end = offsets[stop - 1][1]
            # Several byte-level tokens can share a Unicode character offset.
            # Overlap must not produce duplicate source chunks for those bytes.
            if not chunks or chunks[-1][1:] != (start, end):
                chunks.append((text[start:end], start, end))
            if stop == len(offsets):
                break
            index = max(index + 1, stop - self.chunk_overlap)
        return chunks
