"""Redis cache backend."""
from __future__ import annotations

from typing import Any

from .base import CacheBackend, CacheEntry, dumps_entry, loads_entry


class RedisCache(CacheBackend):
    """Redis-backed cache.

    The redis dependency is optional. Install it separately or through the
    package extra: ``pip install cheragh[redis]``.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        default_ttl: int | float | None = None,
        namespace: str = "default",
        client: Any | None = None,
        key_prefix: str = "cheragh",
        serializer: str = "json",
        secret_key: str | bytes | None = None,
        allow_pickle: bool = False,
        allow_unsigned_pickle: bool = False,
        **client_kwargs: Any,
    ):
        super().__init__(default_ttl=default_ttl, namespace=namespace)
        if client is None:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "RedisCache requires the optional dependency 'redis'. Install with: pip install redis"
                ) from exc
            client = redis.Redis.from_url(url, **client_kwargs)
        self.client = client
        self.url = url
        self.key_prefix = key_prefix
        self.serializer = serializer
        self.secret_key = secret_key
        self.allow_pickle = allow_pickle
        serializer_normalized = serializer.lower().replace("_", "-")
        if serializer_normalized in {"pickle", "signed-pickle"} and not allow_pickle:
            raise ValueError("pickle serializer requires allow_pickle=True")
        if serializer_normalized == "signed-pickle" and not secret_key:
            raise ValueError("signed-pickle serializer requires secret_key")
        if serializer_normalized == "pickle" and allow_pickle and not secret_key and not allow_unsigned_pickle:
            raise ValueError(
                "unsigned pickle is disabled for persistent caches; provide secret_key "
                "or set allow_unsigned_pickle=True for trusted local caches"
            )

    def _redis_key(self, namespace: str, key: str) -> str:
        return f"{self.key_prefix}:{namespace}:{key}"

    def _pattern(self, namespace: str | None = None) -> str:
        ns = namespace if namespace is not None else "*"
        return f"{self.key_prefix}:{ns}:*"

    def _get_entry(self, namespace: str, key: str) -> CacheEntry | None:
        raw = self.client.get(self._redis_key(namespace, key))
        if raw is None:
            return None
        return loads_entry(
            raw,
            serializer=self.serializer,
            secret_key=self.secret_key,
            allow_pickle=self.allow_pickle,
        )

    def _set_entry(self, entry: CacheEntry) -> None:
        raw = dumps_entry(
            entry,
            serializer=self.serializer,
            secret_key=self.secret_key,
            allow_pickle=self.allow_pickle,
        )
        ttl = None
        if entry.expires_at is not None:
            import time

            ttl = max(1, int(entry.expires_at - time.time()))
        redis_key = self._redis_key(entry.namespace, entry.key)
        if ttl is None:
            self.client.set(redis_key, raw)
        else:
            self.client.setex(redis_key, ttl, raw)

    def _delete_entry(self, namespace: str, key: str) -> None:
        self.client.delete(self._redis_key(namespace, key))

    def _clear_namespace(self, namespace: str) -> int:
        keys = list(self.client.scan_iter(match=self._pattern(namespace)))
        if keys:
            self.client.delete(*keys)
        return len(keys)

    def _clear_all(self) -> int:
        keys = list(self.client.scan_iter(match=self._pattern(None)))
        if keys:
            self.client.delete(*keys)
        return len(keys)

    def entry_count(self) -> int:
        return sum(1 for _ in self.client.scan_iter(match=self._pattern(None)))

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # pragma: no cover - client-dependent
            pass
