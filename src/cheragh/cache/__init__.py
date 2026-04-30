"""Advanced cache layer for embeddings, retrieval, reranking and LLM calls."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import CacheBackend, CacheEntry, CacheSerializerError, CacheStats, make_cache_key
from .decorators import (
    CachedEmbeddingModel,
    CachedLLMClient,
    CachedReranker,
    CachedRetriever,
    cache_embedding_model,
    cache_llm_client,
    cache_reranker,
    cache_retriever,
    cached_call,
)
from .legacy import embedder_fingerprint, hash_documents, load_cache, save_cache
from .memory import MemoryCache
from .redis import RedisCache
from .sqlite import SQLiteCache

# cache_method is kept as a public alias for users expecting a decorator name.
cache_method = cached_call


def build_cache_backend(config: dict[str, Any] | None = None, **overrides: Any) -> CacheBackend | None:
    """Build a cache backend from config.

    Accepted config shape::

        cache:
          enabled: true
          backend: sqlite   # memory | sqlite | redis
          path: .cheragh/cache.sqlite
          ttl: 3600
          namespace: default
          serializer: json # json | signed-pickle | pickle
          secret_key: my-cache-hmac-secret # required for signed-pickle
          allow_pickle: false
          allow_unsigned_pickle: false
          redis_url: redis://localhost:6379/0
    """

    cfg = {**(config or {}), **{k: v for k, v in overrides.items() if v is not None}}
    if not cfg:
        return None
    enabled = cfg.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return None
    backend = str(cfg.get("backend", cfg.get("type", "memory"))).lower().replace("_", "-")
    ttl = cfg.get("ttl", cfg.get("default_ttl"))
    ttl = float(ttl) if ttl not in {None, ""} else None
    namespace = str(cfg.get("namespace", "default"))
    serializer = str(cfg.get("serializer", "json")).lower().replace("_", "-")
    secret_key = cfg.get("secret_key") or cfg.get("hmac_key")
    allow_pickle = _as_bool(cfg.get("allow_pickle", serializer in {"pickle", "signed-pickle"}))
    allow_unsigned_pickle = _as_bool(cfg.get("allow_unsigned_pickle", False))
    if serializer == "signed-pickle":
        if not secret_key:
            raise ValueError("cache.serializer='signed-pickle' requires cache.secret_key")
        serializer = "pickle"
    unsafe_persistent_pickle = (
        backend in {"sqlite", "sqlite3", "redis"}
        and serializer == "pickle"
        and allow_pickle
        and not secret_key
        and not allow_unsigned_pickle
    )
    if unsafe_persistent_pickle:
        raise ValueError("persistent pickle cache requires secret_key or allow_unsigned_pickle=True")
    if backend in {"memory", "in-memory", "mem"}:
        return MemoryCache(default_ttl=ttl, namespace=namespace)
    if backend in {"sqlite", "sqlite3"}:
        path = cfg.get("path") or cfg.get("cache_path") or ".cheragh/cache.sqlite"
        return SQLiteCache(
            Path(path),
            default_ttl=ttl,
            namespace=namespace,
            serializer=serializer,
            secret_key=secret_key,
            allow_pickle=allow_pickle,
            allow_unsigned_pickle=allow_unsigned_pickle,
        )
    if backend == "redis":
        from .redis import RedisCache

        return RedisCache(
            url=str(cfg.get("url", cfg.get("redis_url", "redis://localhost:6379/0"))),
            default_ttl=ttl,
            namespace=namespace,
            key_prefix=str(cfg.get("key_prefix", "cheragh")),
            serializer=serializer,
            secret_key=secret_key,
            allow_pickle=allow_pickle,
            allow_unsigned_pickle=allow_unsigned_pickle,
        )
    raise ValueError(f"Unsupported cache backend: {backend}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no", "off"}
    return bool(value)


__all__ = [
    "CacheBackend",
    "CacheEntry",
    "CacheStats",
    "CacheSerializerError",
    "MemoryCache",
    "SQLiteCache",
    "RedisCache",
    "build_cache_backend",
    "make_cache_key",
    "CachedEmbeddingModel",
    "CachedRetriever",
    "CachedReranker",
    "CachedLLMClient",
    "cached_call",
    "cache_method",
    "cache_embedding_model",
    "cache_retriever",
    "cache_reranker",
    "cache_llm_client",
    "hash_documents",
    "embedder_fingerprint",
    "save_cache",
    "load_cache",
]
