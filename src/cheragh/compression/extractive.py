"""Dependency-free context compression utilities."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from ..base import Document, _tokenize


class ContextCompressor(ABC):
    """Interface for components that shrink retrieved context before generation."""

    @abstractmethod
    def compress(self, query: str, documents: Sequence[Document]) -> list[Document]:
        """Return compressed documents."""


@dataclass
class ExtractiveContextCompressor(ContextCompressor):
    """Keep only the most query-relevant sentences from each document.

    The compressor is intentionally deterministic and dependency-free. It is not
    a semantic summarizer; for production compression, it can be replaced by a
    custom compressor implementing the same interface.
    """

    max_sentences_per_doc: int = 3
    max_chars_per_doc: int = 900
    min_sentence_chars: int = 20

    def compress(self, query: str, documents: Sequence[Document]) -> list[Document]:
        q_tokens = set(_tokenize(query))
        compressed: list[Document] = []
        for doc in documents:
            sentences = _split_sentences(doc.content)
            ranked: list[tuple[float, int, str]] = []
            for idx, sentence in enumerate(sentences):
                if len(sentence) < self.min_sentence_chars:
                    continue
                tokens = set(_tokenize(sentence))
                if q_tokens and tokens:
                    score = len(q_tokens & tokens) / max(len(q_tokens), 1)
                else:
                    score = 0.0
                ranked.append((score, idx, sentence))
            if not ranked:
                text = doc.content[: self.max_chars_per_doc]
            else:
                ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                selected = sorted(ranked[: self.max_sentences_per_doc], key=lambda item: item[1])
                text = " ".join(sentence for _, _, sentence in selected)[: self.max_chars_per_doc]
            metadata = dict(doc.metadata)
            metadata["compressed"] = True
            metadata["original_chars"] = len(doc.content)
            metadata["compressed_chars"] = len(text)
            compressed.append(Document(content=text, metadata=metadata, doc_id=doc.doc_id, score=doc.score))
        return compressed


@dataclass
class RedundancyFilter(ContextCompressor):
    """Remove near-duplicate chunks based on token Jaccard overlap."""

    threshold: float = 0.86
    min_tokens: int = 8

    def compress(self, query: str, documents: Sequence[Document]) -> list[Document]:
        kept: list[Document] = []
        kept_tokens: list[set[str]] = []
        for doc in documents:
            tokens = set(_tokenize(doc.content))
            if len(tokens) < self.min_tokens:
                kept.append(doc)
                kept_tokens.append(tokens)
                continue
            duplicate = False
            for other in kept_tokens:
                if not other:
                    continue
                overlap = len(tokens & other) / max(len(tokens | other), 1)
                if overlap >= self.threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(doc)
                kept_tokens.append(tokens)
        return kept


@dataclass
class CompressionPipeline(ContextCompressor):
    """Apply several compressors in sequence."""

    compressors: Iterable[ContextCompressor]

    def compress(self, query: str, documents: Sequence[Document]) -> list[Document]:
        current = list(documents)
        for compressor in self.compressors:
            current = compressor.compress(query, current)
        return current


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?。！？])\s+", text) if sentence.strip()]
