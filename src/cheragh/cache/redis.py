"""Redis cache backend."""
from __future__ import annotations

import math
import time
from typing import Any

from .base import CacheBackend, CacheEntry, CacheSerializerError, dumps_entry, loads_entry


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
            # A cache outage must not pin an application worker indefinitely.
            # Callers can override either timeout explicitly.
            client_kwargs.setdefault("socket_connect_timeout", 5.0)
            client_kwargs.setdefault("socket_timeout", 5.0)
            client_kwargs.setdefault("health_check_interval", 30)
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
        prefix = _escape_redis_glob(self.key_prefix)
        ns = _escape_redis_glob(namespace) if namespace is not None else "*"
        return f"{prefix}:{ns}:*"

    def _get_entry(self, namespace: str, key: str) -> CacheEntry | None:
        raw = self.client.get(self._redis_key(namespace, key))
        if raw is None:
            return None
        try:
            entry = loads_entry(
                raw,
                serializer=self.serializer,
                secret_key=self.secret_key,
                allow_pickle=self.allow_pickle,
            )
            if entry.namespace != namespace or entry.key != key:
                raise CacheSerializerError("cache entry identity does not match its Redis key")
            return entry
        except Exception:
            # Remove poisoned data after surfacing the error to the base layer.
            self.client.delete(self._redis_key(namespace, key))
            raise

    def _set_entry(self, entry: CacheEntry) -> None:
        raw = dumps_entry(
            entry,
            serializer=self.serializer,
            secret_key=self.secret_key,
            allow_pickle=self.allow_pickle,
        )
        ttl = None
        if entry.expires_at is not None:
            # Round up: truncating a fractional TTL expires healthy data early.
            ttl = max(1, math.ceil(entry.expires_at - time.time()))
        redis_key = self._redis_key(entry.namespace, entry.key)
        if ttl is None:
            self.client.set(redis_key, raw)
        else:
            self.client.setex(redis_key, ttl, raw)

    def _delete_entry(self, namespace: str, key: str) -> None:
        self.client.delete(self._redis_key(namespace, key))

    def _delete_expired_entry(self, namespace: str, key: str, entry: CacheEntry) -> None:
        # Compare-and-delete prevents a slow reader from deleting a newer value
        # installed under the same key. Redis entries also have a server TTL, so
        # inability to run the optional Lua cleanup is safe.
        expected = dumps_entry(
            entry,
            serializer=self.serializer,
            secret_key=self.secret_key,
            allow_pickle=self.allow_pickle,
        )
        script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] "
            "then return redis.call('del', KEYS[1]) else return 0 end"
        )
        try:
            self.client.eval(script, 1, self._redis_key(namespace, key), expected)
        except Exception:  # pragma: no cover - optional/fake client behavior
            return

    def _clear_namespace(self, namespace: str) -> int:
        return self._delete_pattern(self._pattern(namespace))

    def _clear_all(self) -> int:
        return self._delete_pattern(self._pattern(None))

    def entry_count(self) -> int:
        return sum(1 for _ in self.client.scan_iter(match=self._pattern(None)))

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # pragma: no cover - client-dependent
            pass

    def _delete_pattern(self, pattern: str, *, batch_size: int = 500) -> int:
        batch: list[Any] = []
        removed = 0
        for key in self.client.scan_iter(match=pattern, count=batch_size):
            batch.append(key)
            if len(batch) >= batch_size:
                removed += int(self.client.delete(*batch) or 0)
                batch.clear()
        if batch:
            removed += int(self.client.delete(*batch) or 0)
        return removed


def _escape_redis_glob(value: str) -> str:
    """Escape Redis glob metacharacters used by SCAN MATCH.

    Namespace text remains unchanged in actual keys; only invalidation patterns
    are escaped, preserving compatibility with already persisted entries.
    """

    escaped: list[str] = []
    for character in str(value):
        if character in {"*", "?", "[", "]", "\\"}:
            escaped.append("\\")
        escaped.append(character)
    return "".join(escaped)
