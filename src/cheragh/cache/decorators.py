"""Cache decorators and wrappers for RAG components."""
from __future__ import annotations

import functools
from typing import Any, Callable, Iterable, Sequence

from ..base import BaseRetriever, Document, EmbeddingModel, LLMClient
from ..reranking import BaseReranker
from .base import CacheBackend, make_cache_key


def cached_call(
    cache: CacheBackend,
    namespace: str,
    key_builder: Callable[..., str] | None = None,
    ttl: int | float | None = None,
):
    """Decorate a function with a cache backend."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = key_builder(*args, **kwargs) if key_builder else make_cache_key(fn.__module__, fn.__qualname__, args, kwargs)
            return cache.get_or_set(key, lambda: fn(*args, **kwargs), ttl=ttl, namespace=namespace)

        return wrapper

    return decorator


class CachedEmbeddingModel(EmbeddingModel):
    """EmbeddingModel wrapper with document/query embedding cache."""

    def __init__(self, model: EmbeddingModel, cache: CacheBackend, ttl: int | float | None = None, namespace: str = "embeddings"):
        self.model = model
        self.cache = cache
        self.ttl = ttl
        self.namespace = namespace

    def embed_query(self, text: str):
        key = make_cache_key("query", self.model.get_fingerprint(), text)
        return self.cache.get_or_set(key, lambda: self.model.embed_query(text), ttl=self.ttl, namespace=self.namespace)

    def embed_documents(self, texts: list[str]):
        # Cache per text to maximize reuse across incremental ingestion and retrieval.
        import numpy as np

        if not texts:
            return self.model.embed_documents(texts)
        outputs: list[Any] = []
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        keys: list[str] = []
        sentinel = object()
        for i, text in enumerate(texts):
            key = make_cache_key("document", self.model.get_fingerprint(), text)
            keys.append(key)
            value = self.cache.get(key, default=sentinel, namespace=self.namespace)
            if value is sentinel:
                outputs.append(None)
                missing_indices.append(i)
                missing_texts.append(text)
            else:
                outputs.append(value)
        if missing_texts:
            embedded = self.model.embed_documents(missing_texts)
            for idx, vector in zip(missing_indices, embedded):
                outputs[idx] = vector
                self.cache.set(keys[idx], vector, ttl=self.ttl, namespace=self.namespace)
        return np.vstack(outputs)

    def get_fingerprint(self) -> str:
        return f"Cached::{self.model.get_fingerprint()}"


class CachedRetriever(BaseRetriever):
    """Retriever wrapper with query/top_k cache."""

    def __init__(self, retriever: BaseRetriever, cache: CacheBackend, ttl: int | float | None = None, namespace: str = "retrieval"):
        self.retriever = retriever
        self.cache = cache
        self.ttl = ttl
        self.namespace = namespace

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        key = make_cache_key(self.retriever.__class__.__name__, query, top_k, getattr(self.retriever, "filters", None))
        return self.cache.get_or_set(key, lambda: self.retriever.retrieve(query, top_k=top_k), ttl=self.ttl, namespace=self.namespace)


class CachedReranker(BaseReranker):
    """Reranker wrapper with cache over query, document IDs/content and top_k."""

    def __init__(self, reranker: BaseReranker, cache: CacheBackend, ttl: int | float | None = None, namespace: str = "reranking"):
        self.reranker = reranker
        self.cache = cache
        self.ttl = ttl
        self.namespace = namespace

    def rerank(self, query: str, documents: Sequence[Document], top_k: int = 5) -> list[Document]:
        doc_fingerprint = [(doc.doc_id, doc.content, doc.score) for doc in documents]
        key = make_cache_key(self.reranker.__class__.__name__, query, top_k, doc_fingerprint)
        return self.cache.get_or_set(
            key,
            lambda: self.reranker.rerank(query, documents, top_k=top_k),
            ttl=self.ttl,
            namespace=self.namespace,
        )


class CachedLLMClient(LLMClient):
    """LLMClient wrapper caching non-streaming generate() responses."""

    def __init__(self, client: LLMClient, cache: CacheBackend, ttl: int | float | None = None, namespace: str = "llm"):
        self.client = client
        self.cache = cache
        self.ttl = ttl
        self.namespace = namespace

    def generate(self, prompt: str, **kwargs: Any) -> str:
        key = make_cache_key(self.client.__class__.__name__, prompt, kwargs)
        return self.cache.get_or_set(key, lambda: self.client.generate(prompt, **kwargs), ttl=self.ttl, namespace=self.namespace)

    def stream(self, prompt: str, **kwargs: Any):
        # Use the cached generate path for deterministic cache behavior.
        yield self.generate(prompt, **kwargs)


def cache_embedding_model(model: EmbeddingModel, cache: CacheBackend, ttl: int | float | None = None) -> CachedEmbeddingModel:
    return CachedEmbeddingModel(model, cache, ttl=ttl)


def cache_retriever(retriever: BaseRetriever, cache: CacheBackend, ttl: int | float | None = None) -> CachedRetriever:
    return CachedRetriever(retriever, cache, ttl=ttl)


def cache_reranker(reranker: BaseReranker, cache: CacheBackend, ttl: int | float | None = None) -> CachedReranker:
    return CachedReranker(reranker, cache, ttl=ttl)


def cache_llm_client(client: LLMClient, cache: CacheBackend, ttl: int | float | None = None) -> CachedLLMClient:
    return CachedLLMClient(client, cache, ttl=ttl)
