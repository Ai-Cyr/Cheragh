"""Cache decorators and wrappers for RAG components."""
from __future__ import annotations

from collections.abc import Mapping, Sequence as SequenceABC
import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from ..base import BaseRetriever, Document, EmbeddingModel, LLMClient, _snapshot_documents
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

    def __init__(
        self,
        model: EmbeddingModel,
        cache: CacheBackend,
        ttl: int | float | None = None,
        namespace: str = "embeddings",
        fingerprint: str | None = None,
    ):
        self.model = model
        self.cache = cache
        self.ttl = ttl
        self.namespace = namespace
        self.cache_fingerprint = (
            str(fingerprint) if fingerprint is not None else _component_fingerprint(model, purpose="embedder")
        )

    def embed_query(self, text: str):
        key = make_cache_key("query", self.cache_fingerprint, text)
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
            key = make_cache_key("document", self.cache_fingerprint, text)
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

    def __init__(
        self,
        retriever: BaseRetriever,
        cache: CacheBackend,
        ttl: int | float | None = None,
        namespace: str = "retrieval",
        fingerprint: str | None = None,
    ):
        self.retriever = retriever
        self.cache = cache
        self.ttl = ttl
        self.namespace = namespace
        self.fingerprint = str(fingerprint) if fingerprint is not None else _retriever_fingerprint(retriever)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        key = make_cache_key(
            self.retriever.__class__.__name__,
            self.fingerprint,
            query,
            top_k,
            getattr(self.retriever, "filters", None),
        )
        documents = self.cache.get_or_set(
            key,
            lambda: _snapshot_documents(self.retriever.retrieve(query, top_k=top_k)),
            ttl=self.ttl,
            namespace=self.namespace,
        )
        return _snapshot_documents(documents)


class CachedReranker(BaseReranker):
    """Reranker wrapper with cache over query, document IDs/content and top_k."""

    def __init__(
        self,
        reranker: BaseReranker,
        cache: CacheBackend,
        ttl: int | float | None = None,
        namespace: str = "reranking",
        fingerprint: str | None = None,
    ):
        self.reranker = reranker
        self.cache = cache
        self.ttl = ttl
        self.namespace = namespace
        self.fingerprint = (
            str(fingerprint) if fingerprint is not None else _component_fingerprint(reranker, purpose="reranker")
        )

    def rerank(self, query: str, documents: Sequence[Document], top_k: int = 5) -> list[Document]:
        doc_fingerprint, complete = _documents_fingerprint(documents)
        if not complete:
            return _snapshot_documents(self.reranker.rerank(query, documents, top_k=top_k))
        key = make_cache_key(self.reranker.__class__.__name__, self.fingerprint, query, top_k, doc_fingerprint)
        reranked = self.cache.get_or_set(
            key,
            lambda: _snapshot_documents(self.reranker.rerank(query, documents, top_k=top_k)),
            ttl=self.ttl,
            namespace=self.namespace,
        )
        return _snapshot_documents(reranked)


class CachedLLMClient(LLMClient):
    """LLMClient wrapper caching non-streaming generate() responses."""

    def __init__(
        self,
        client: LLMClient,
        cache: CacheBackend,
        ttl: int | float | None = None,
        namespace: str = "llm",
        fingerprint: str | None = None,
    ):
        self.client = client
        self.cache = cache
        self.ttl = ttl
        self.namespace = namespace
        self.fingerprint = (
            str(fingerprint) if fingerprint is not None else _component_fingerprint(client, purpose="llm")
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        key = make_cache_key(self.client.__class__.__name__, self.fingerprint, prompt, kwargs)
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


def _retriever_fingerprint(retriever: BaseRetriever) -> str:
    """Return a stable cache identity for built-in retrievers.

    Corpus content, store/embedder identity and retrieval configuration are
    hashed rather than placed directly in cache keys. Custom retrievers can
    expose ``get_fingerprint()`` or callers can pass ``fingerprint=...`` to
    :class:`CachedRetriever`. When no stable data-source identity is available,
    the object identity is included to prefer a cache miss over cross-index data
    reuse.
    """

    return _retriever_snapshot(retriever, seen=set())


def _retriever_snapshot(retriever: Any, *, seen: set[int]) -> str:
    object_id = id(retriever)
    if object_id in seen:
        return make_cache_key("retriever-cycle", _class_name(retriever), object_id)
    seen.add(object_id)

    custom = getattr(retriever, "get_fingerprint", None)
    if callable(custom):
        try:
            value = custom()
            fingerprint = str(value) if value is not None else ""
        except Exception:
            fingerprint = ""
        if fingerprint:
            return make_cache_key("retriever", _class_name(retriever), fingerprint)

    snapshot: dict[str, Any] = {"class": _class_name(retriever)}
    complete = True
    stable_source = False

    documents = getattr(retriever, "documents", None)
    store = getattr(retriever, "store", None)
    if documents is None and store is not None:
        documents = getattr(store, "documents", None)
    if isinstance(documents, SequenceABC) and not isinstance(documents, (str, bytes, bytearray)):
        corpus_fingerprint, corpus_complete = _documents_fingerprint(documents)
        snapshot["corpus"] = corpus_fingerprint
        complete = complete and corpus_complete
        stable_source = True

    if store is not None:
        store_snapshot, store_complete = _selected_attributes(
            store,
            ("collection_name", "distance", "normalize", "path", "url"),
        )
        snapshot["store"] = {"class": _class_name(store), **store_snapshot}
        complete = complete and store_complete

    embedder = getattr(retriever, "embedding_model", None)
    if embedder is None and store is not None:
        embedder = getattr(store, "embedding_model", None)
    if embedder is not None:
        snapshot["embedder"] = _component_fingerprint(embedder, purpose="embedder")

    config_snapshot, config_complete = _selected_attributes(
        retriever,
        ("alpha", "filters", "first_stage_top_k", "normalize", "distance", "collection_name"),
    )
    if config_snapshot:
        snapshot["config"] = config_snapshot
    complete = complete and config_complete

    tokenizer = getattr(retriever, "tokenizer", None)
    if tokenizer is not None:
        tokenizer_snapshot, tokenizer_complete = _selected_attributes(
            tokenizer,
            ("lowercase", "strip_accents", "keep_hyphenated", "stopwords", "ngram_range", "min_token_length"),
        )
        snapshot["tokenizer"] = {"class": _class_name(tokenizer), **tokenizer_snapshot}
        complete = complete and tokenizer_complete

    for name in ("base_retriever", "retriever"):
        nested = getattr(retriever, name, None)
        if nested is not None and nested is not retriever:
            snapshot[name] = _retriever_snapshot(nested, seen=seen)
            stable_source = True

    reranker = getattr(retriever, "reranker", None)
    if reranker is not None:
        snapshot["reranker"] = _component_fingerprint(reranker, purpose="reranker")

    if not stable_source or not complete or not _is_package_component(retriever):
        snapshot["instance"] = object_id
    return make_cache_key("retriever", snapshot)


def _component_fingerprint(component: Any, *, purpose: str) -> str:
    custom = getattr(component, "get_fingerprint", None)
    if callable(custom):
        try:
            value = custom()
            fingerprint = str(value) if value is not None else ""
        except Exception:
            fingerprint = ""
        if fingerprint:
            declared: dict[str, Any] = {
                "class": _class_name(component),
                "declared": fingerprint,
            }
            cache_fingerprint = getattr(component, "cache_fingerprint", None)
            if cache_fingerprint not in {None, ""}:
                declared["cache_fingerprint"] = str(cache_fingerprint)
            # Provider SDK clients can point at different endpoints, projects,
            # deployments or test transports while exposing the same model
            # name. Without an explicit wrapper fingerprint, isolate opaque
            # client instances rather than risking cross-provider cache hits.
            if getattr(component, "client", None) is not None:
                declared["instance"] = id(component)
            return make_cache_key(purpose, declared)

    snapshot: dict[str, Any] = {"class": _class_name(component)}
    attributes, complete = _selected_attributes(
        component,
        (
            "model",
            "model_name",
            "deployment_name",
            "api_version",
            "base_url",
            "distance",
            "normalize",
            "collection_name",
            "default_kwargs",
            "response",
            "k",
        ),
    )
    snapshot.update(attributes)
    if (
        not attributes
        or not complete
        or not _is_package_component(component)
        or getattr(component, "client", None) is not None
    ):
        snapshot["instance"] = id(component)
    return make_cache_key(purpose, snapshot)


def _documents_fingerprint(documents: SequenceABC[Any]) -> tuple[str, bool]:
    digest = hashlib.sha256()
    complete = True
    for document in documents:
        if not isinstance(document, Document):
            complete = False
            digest.update(_class_name(document).encode("utf-8"))
            continue
        metadata, metadata_complete = _safe_snapshot(document.metadata)
        complete = complete and metadata_complete
        payload = {
            "doc_id": document.doc_id,
            "content": document.content,
            "metadata": metadata,
            "score": document.score,
        }
        digest.update(_stable_json(payload).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest(), complete


def _selected_attributes(component: Any, names: Sequence[str]) -> tuple[dict[str, Any], bool]:
    output: dict[str, Any] = {}
    complete = True
    for name in names:
        if not hasattr(component, name):
            continue
        value = getattr(component, name)
        snapshot, value_complete = _safe_snapshot(value)
        output[name] = snapshot
        complete = complete and value_complete
    return output, complete


def _safe_snapshot(value: Any) -> tuple[Any, bool]:
    if value is None or isinstance(value, (str, int, bool)):
        return value, True
    if isinstance(value, float):
        return repr(value), True
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}, True
    if isinstance(value, Path):
        return str(value), True
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        complete = True
        for key in sorted(value, key=lambda item: str(item)):
            item, item_complete = _safe_snapshot(value[key])
            output[str(key)] = item
            complete = complete and item_complete
        return output, complete
    if isinstance(value, (list, tuple)):
        output = []
        complete = True
        for raw_item in value:
            item, item_complete = _safe_snapshot(raw_item)
            output.append(item)
            complete = complete and item_complete
        return output, complete
    if isinstance(value, (set, frozenset)):
        items = [_safe_snapshot(item) for item in value]
        return sorted((item for item, _ in items), key=_stable_json), all(complete for _, complete in items)
    return {"type": _class_name(value)}, False


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _class_name(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_package_component(value: Any) -> bool:
    module = value.__class__.__module__
    return module == "cheragh" or module.startswith("cheragh.")
