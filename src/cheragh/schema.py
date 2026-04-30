"""Stable public schema and protocol types for cheragh v1.0.

The classes in this module are the compatibility boundary for applications that
build on top of the package. Internal modules may evolve, but these lightweight
objects should remain backward-compatible across 1.x releases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Protocol, runtime_checkable

from .base import Document
from .citations import CitationValidationResult
from .tracing import RAGTrace


@dataclass
class Chunk(Document):
    """A retrievable chunk with explicit parent and character-span metadata."""

    parent_doc_id: str | None = None
    source_char_start: int | None = None
    source_char_end: int | None = None

    @classmethod
    def from_document(cls, document: Document) -> "Chunk":
        metadata = dict(document.metadata or {})
        return cls(
            content=document.content,
            metadata=metadata,
            doc_id=document.doc_id,
            score=document.score,
            parent_doc_id=metadata.get("parent_doc_id"),
            source_char_start=metadata.get("source_char_start"),
            source_char_end=metadata.get("source_char_end"),
        )

    def to_document(self) -> Document:
        metadata = dict(self.metadata or {})
        if self.parent_doc_id is not None:
            metadata.setdefault("parent_doc_id", self.parent_doc_id)
        if self.source_char_start is not None:
            metadata.setdefault("source_char_start", self.source_char_start)
        if self.source_char_end is not None:
            metadata.setdefault("source_char_end", self.source_char_end)
        return Document(content=self.content, metadata=metadata, doc_id=self.doc_id, score=self.score)


@dataclass
class Source:
    """Source returned with a RAG answer."""

    doc_id: str | None
    score: float | None
    preview: str
    metadata: dict[str, Any] = field(default_factory=dict)
    location: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": self.score,
            "preview": self.preview,
            "metadata": self.metadata,
            "location": self.location,
        }


@dataclass
class RAGResponse:
    """Structured response returned by :class:`cheragh.RAGEngine`."""

    query: str
    answer: str
    sources: list[Source]
    retrieved_documents: list[Document]
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    grounded_score: float = 0.0
    unsourced_claims: list[str] = field(default_factory=list)
    citation_validation: CitationValidationResult | None = None
    trace: RAGTrace | None = None

    def to_dict(self, *, include_prompt: bool = False) -> dict[str, Any]:
        data = {
            "query": self.query,
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "citations": self.citations,
            "warnings": self.warnings,
            "grounded_score": self.grounded_score,
            "unsourced_claims": self.unsourced_claims,
            "metadata": self.metadata,
        }
        if self.citation_validation is not None:
            data["citation_validation"] = self.citation_validation.__dict__
        if self.trace is not None:
            data["trace"] = self.trace.to_dict(include_prompt=include_prompt)
        if include_prompt:
            data["prompt"] = self.prompt
        return data


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol implemented by retrievers."""

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        ...


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Protocol implemented by embedding providers."""

    def embed_documents(self, texts: list[str]) -> Any:
        ...

    def embed_query(self, text: str) -> Any:
        ...

    def get_fingerprint(self) -> str:
        ...


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol implemented by generation clients."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Protocol implemented by rerankers."""

    def rerank(self, query: str, documents: Iterable[Document], top_k: int | None = None) -> list[Document]:
        ...
