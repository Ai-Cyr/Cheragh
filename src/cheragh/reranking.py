"""
Reranking utilities
===================

Two-stage retrieval: retrieve a larger candidate set, then rerank query/document
pairs with a stronger scoring model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from typing import Iterable, List, Sequence

from .base import BaseRetriever, Document, _tokenize, _validate_top_k


class BaseReranker(ABC):
    """Common interface for rerankers."""

    @abstractmethod
    def rerank(self, query: str, documents: Sequence[Document], top_k: int = 5) -> list[Document]:
        """Return reranked documents."""


class CrossEncoderReranker(BaseReranker):
    """Reranker backed by ``sentence-transformers`` CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", model=None, **model_kwargs):
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "CrossEncoderReranker requires sentence-transformers. "
                    "Install with: pip install cheragh[rerank]"
                ) from exc
            model = CrossEncoder(model_name, **model_kwargs)
        self.model_name = model_name
        self.model = model

    def rerank(self, query: str, documents: Sequence[Document], top_k: int = 5) -> list[Document]:
        top_k = _validate_top_k(top_k)
        if not documents:
            return []
        pairs = [(query, doc.content) for doc in documents]
        scores = self.model.predict(pairs)
        scored = sorted(zip(documents, scores), key=lambda item: float(item[1]), reverse=True)
        return [_copy_with_rerank_score(doc, float(score)) for doc, score in scored[:top_k]]


class KeywordOverlapReranker(BaseReranker):
    """Dependency-free fallback reranker based on token overlap.

    This is not a replacement for a cross-encoder, but it is deterministic and
    useful for tests, examples and offline environments.
    """

    def rerank(self, query: str, documents: Sequence[Document], top_k: int = 5) -> list[Document]:
        top_k = _validate_top_k(top_k)
        query_tokens = set(_tokenize(query))
        scored: list[tuple[Document, float]] = []
        for doc in documents:
            doc_tokens = set(_tokenize(doc.content))
            if not query_tokens or not doc_tokens:
                score = 0.0
            else:
                score = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
            scored.append((doc, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [_copy_with_rerank_score(doc, score) for doc, score in scored[:top_k]]


class CohereReranker(BaseReranker):
    """Reranker backed by Cohere Rerank."""

    def __init__(self, model: str = "rerank-multilingual-v3.0", api_key: str | None = None, client=None):
        if client is None:
            try:
                import cohere
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("CohereReranker requires: pip install cheragh[cohere]") from exc
            client = cohere.Client(api_key)
        self.client = client
        self.model = model

    def rerank(self, query: str, documents: Sequence[Document], top_k: int = 5) -> list[Document]:
        top_k = _validate_top_k(top_k)
        if not documents:
            return []
        response = self.client.rerank(query=query, documents=[doc.content for doc in documents], top_n=top_k, model=self.model)
        output: list[Document] = []
        for item in response.results:
            doc = documents[int(item.index)]
            output.append(_copy_with_rerank_score(doc, float(item.relevance_score)))
        return output


class ReciprocalRankFusionReranker(BaseReranker):
    """Fuse several already-ranked document lists with RRF.

    ``rerank`` applies the single-list RRF prior and ``fuse`` combines multiple
    independently ranked lists.  For retrieval from several sources, prefer
    :class:`ReciprocalRankFusionRetriever`.
    """

    def __init__(self, k: int = 60):
        self.k = _validate_top_k(k, name="k")

    def rerank(self, query: str, documents: Sequence[Document], top_k: int = 5) -> list[Document]:
        top_k = _validate_top_k(top_k)
        return self.fuse([documents], top_k=top_k)

    def fuse(self, ranked_lists: Iterable[Sequence[Document]], top_k: int = 5) -> list[Document]:
        top_k = _validate_top_k(top_k)
        scores: dict[str, float] = {}
        docs_by_key: dict[str, Document] = {}
        source_ranks: dict[str, list[tuple[int, int]]] = {}
        for source_index, ranked in enumerate(ranked_lists):
            seen_in_source: set[str] = set()
            for rank, doc in enumerate(ranked, start=1):
                key = _rrf_document_key(doc)
                if key in seen_in_source:
                    continue
                seen_in_source.add(key)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.k + rank)
                docs_by_key.setdefault(key, doc)
                source_ranks.setdefault(key, []).append((source_index, rank))
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            _copy_with_rerank_score(
                docs_by_key[key],
                score,
                extra_metadata={
                    "rrf_sources": len(source_ranks[key]),
                    "rrf_source_ranks": [
                        {"source": source, "rank": rank}
                        for source, rank in source_ranks[key]
                    ],
                },
            )
            for key, score in ordered
        ]


class ReciprocalRankFusionRetriever(BaseRetriever):
    """Retrieve from several sources and fuse their ranks with canonical RRF."""

    def __init__(
        self,
        retrievers: Sequence[BaseRetriever],
        *,
        candidate_top_k: int = 20,
        k: int = 60,
    ):
        if not retrievers:
            raise ValueError("ReciprocalRankFusionRetriever requires at least one retriever")
        self.retrievers = tuple(retrievers)
        self.candidate_top_k = _validate_top_k(candidate_top_k, name="candidate_top_k")
        self.fuser = ReciprocalRankFusionReranker(k=k)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        top_k = _validate_top_k(top_k)
        depth = max(top_k, self.candidate_top_k)
        ranked_lists = [retriever.retrieve(query, top_k=depth) for retriever in self.retrievers]
        results = self.fuser.fuse(ranked_lists, top_k=top_k)
        for document in results:
            document.metadata["retrieval_method"] = "reciprocal-rank-fusion"
        return results


class RerankingRetriever(BaseRetriever):
    """Two-stage retriever: base retriever + reranker.

    The constructor remains backward-compatible with v0.1/v0.2: passing
    ``cross_encoder_model`` without ``reranker`` creates a ``CrossEncoderReranker``.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        first_stage_top_k: int = 30,
        reranker: BaseReranker | None = None,
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker or CrossEncoderReranker(cross_encoder_model)
        self.first_stage_top_k = _validate_top_k(first_stage_top_k, name="first_stage_top_k")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        top_k = _validate_top_k(top_k)
        candidates = self.base_retriever.retrieve(query, top_k=max(self.first_stage_top_k, top_k))
        return self.reranker.rerank(query, candidates, top_k=top_k)


@dataclass
class RerankingConfig:
    enabled: bool = False
    provider: str = "cross-encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    first_stage_top_k: int = 30


def build_reranker(provider: str = "cross-encoder", model: str | None = None, **kwargs) -> BaseReranker:
    """Build a reranker from a small provider string."""
    p = provider.lower().replace("_", "-")
    if p in {"cross-encoder", "crossencoder", "sentence-transformers"}:
        return CrossEncoderReranker(model_name=model or "cross-encoder/ms-marco-MiniLM-L-6-v2", **kwargs)
    if p in {"keyword", "keyword-overlap", "local"}:
        return KeywordOverlapReranker()
    if p == "cohere":
        return CohereReranker(model=model or "rerank-multilingual-v3.0", **kwargs)
    raise ValueError(f"Unsupported reranker provider: {provider}")


def _rrf_document_key(doc: Document) -> str:
    if doc.doc_id is not None and not isinstance(doc.doc_id, str):
        raise TypeError("RRF document ids must be strings or None")
    if isinstance(doc.doc_id, str) and doc.doc_id.strip():
        return f"id::{doc.doc_id}"
    digest = hashlib.sha256(doc.content.encode("utf-8")).hexdigest()
    return f"content::{digest}"


def _copy_with_rerank_score(
    doc: Document,
    score: float,
    *,
    extra_metadata: dict[str, object] | None = None,
) -> Document:
    return Document(
        content=doc.content,
        metadata={
            **deepcopy(doc.metadata),
            "first_stage_score": doc.score,
            "rerank_score": score,
            **deepcopy(extra_metadata or {}),
        },
        doc_id=doc.doc_id,
        score=score,
    )
