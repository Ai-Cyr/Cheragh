"""Configurable cache backends for cheragh.

The cache layer is intentionally small and dependency-light. Backends store
values in a versioned envelope with TTL, namespaces, invalidation and basic
hit/miss statistics.

Security note
-------------
Persistent backends default to a JSON serializer that supports the built-in
``Document`` type and NumPy arrays without executing code while loading entries.
Pickle remains available only for trusted single-user caches. If pickle is used
with SQLite or Redis, provide ``secret_key`` so entries are authenticated with
HMAC before unpickling, or explicitly opt in to unsigned pickle in trusted
local-only environments.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass, field, is_dataclass
import hashlib
import hmac
import json
import pickle
import time
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1
_HMAC_PREFIX = b"ARAGC1-HMAC\n"
_JSON_PREFIX = b"ARAGC1-JSON\n"


class CacheSerializerError(ValueError):
    """Raised when a cache entry cannot be safely serialized/deserialized."""


@dataclass
class CacheStats:
    """Runtime statistics collected by cache backends."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    expired: int = 0
    errors: int = 0
    clears: int = 0
    backend: str = "unknown"
    entries: int | None = None

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "expired": self.expired,
            "errors": self.errors,
            "clears": self.clears,
            "entries": self.entries,
            "requests": self.requests,
            "hit_rate": self.hit_rate,
        }


@dataclass
class CacheEntry:
    """Serialized cache entry envelope."""

    key: str
    namespace: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() >= self.expires_at


def make_cache_key(*parts: Any, prefix: str | None = None) -> str:
    """Build a stable SHA256 cache key from arbitrary Python objects."""

    digest = hashlib.sha256()
    if prefix:
        digest.update(str(prefix).encode("utf-8"))
        digest.update(b"\x00")
    for part in parts:
        digest.update(_stable_bytes(part))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _stable_bytes(value: Any) -> bytes:
    if value is None or isinstance(value, (str, int, float, bool)):
        return repr(value).encode("utf-8")
    if isinstance(value, bytes):
        return value
    if isinstance(value, Mapping):
        items = sorted((str(k), _stable_bytes(v)) for k, v in value.items())
        return b"{" + b",".join(k.encode("utf-8") + b":" + v for k, v in items) + b"}"
    if isinstance(value, (list, tuple, set, frozenset)):
        iterable: Iterable[Any] = value if not isinstance(value, (set, frozenset)) else sorted(value, key=repr)
        return b"[" + b",".join(_stable_bytes(v) for v in iterable) + b"]"
    # NumPy arrays and dataclasses are supported through pickle hashing.
    try:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return repr(value).encode("utf-8")


def dumps_entry(
    entry: CacheEntry,
    *,
    serializer: str = "json",
    secret_key: str | bytes | None = None,
    allow_pickle: bool = False,
) -> bytes:
    """Serialize a cache entry.

    ``serializer="json"`` is safe for common cache payloads including
    primitives, lists/dicts, NumPy arrays and :class:`cheragh.base.Document`.
    ``serializer="pickle"`` preserves full Python-object compatibility but must
    only be used with trusted storage; ``secret_key`` adds an HMAC envelope so
    tampered entries are rejected before unpickling.
    """

    payload = _entry_payload(entry)
    serializer = serializer.lower().replace("_", "-")
    if serializer == "json":
        encoded = json.dumps(
            _json_safe_encode(payload),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = _JSON_PREFIX + encoded
    elif serializer in {"pickle", "signed-pickle"}:
        if not allow_pickle:
            raise CacheSerializerError("pickle serialization is disabled for this cache backend")
        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise CacheSerializerError(f"Unsupported cache serializer: {serializer}")
    return _sign(raw, secret_key) if secret_key else raw


def loads_entry(
    data: bytes,
    *,
    serializer: str = "json",
    secret_key: str | bytes | None = None,
    allow_pickle: bool = False,
) -> CacheEntry:
    """Deserialize a cache entry and validate its schema."""

    raw = _verify(data, secret_key) if secret_key else data
    serializer = serializer.lower().replace("_", "-")
    if raw.startswith(_JSON_PREFIX):
        payload = _json_safe_decode(json.loads(raw[len(_JSON_PREFIX) :].decode("utf-8")))
    elif serializer == "json":
        payload = _json_safe_decode(json.loads(raw.decode("utf-8")))
    elif serializer in {"pickle", "signed-pickle"}:
        if not allow_pickle:
            raise CacheSerializerError("pickle deserialization is disabled for this cache backend")
        payload = pickle.loads(raw)
    else:
        raise CacheSerializerError(f"Unsupported cache serializer: {serializer}")
    return _entry_from_payload(payload)


def _json_safe_encode(value: Any) -> Any:
    """Encode common RAG cache values as JSON without pickle."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"__cheragh_type__": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(k): _json_safe_encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_encode(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe_encode(v) for v in sorted(value, key=repr)]
    if is_dataclass(value) and value.__class__.__name__ == "Document":
        return {
            "__cheragh_type__": "Document",
            "content": _json_safe_encode(getattr(value, "content")),
            "metadata": _json_safe_encode(getattr(value, "metadata", {})),
            "doc_id": getattr(value, "doc_id", None),
            "score": getattr(value, "score", None),
        }
    # NumPy import is deliberately last so normal Document/list/string caches do
    # not pay heavy native-library startup cost.
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return {
                "__cheragh_type__": "ndarray",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist(),
            }
        if isinstance(value, np.generic):
            return value.item()
    except Exception:  # pragma: no cover - numpy optional/import-specific
        pass
    raise CacheSerializerError(f"Value of type {type(value).__name__!r} is not JSON-cache serializable")


def _json_safe_decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_safe_decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    marker = value.get("__cheragh_type__")
    if marker == "bytes":
        return base64.b64decode(value["data"].encode("ascii"))
    if marker == "ndarray":
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise CacheSerializerError("NumPy is required to decode cached ndarray values") from exc
        return np.asarray(value["data"], dtype=value.get("dtype")).reshape(value.get("shape"))
    if marker == "Document":
        from ..base import Document

        return Document(
            content=str(value.get("content", "")),
            metadata=dict(_json_safe_decode(value.get("metadata") or {})),
            doc_id=value.get("doc_id"),
            score=value.get("score"),
        )
    return {k: _json_safe_decode(v) for k, v in value.items()}


def _entry_payload(entry: CacheEntry) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "key": entry.key,
        "namespace": entry.namespace,
        "created_at": entry.created_at,
        "expires_at": entry.expires_at,
        "metadata": entry.metadata,
        "value": entry.value,
    }


def _entry_from_payload(payload: Any) -> CacheEntry:
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise CacheSerializerError("Unsupported cache entry schema")
    return CacheEntry(
        key=str(payload["key"]),
        namespace=str(payload["namespace"]),
        value=payload.get("value"),
        created_at=float(payload.get("created_at", time.time())),
        expires_at=payload.get("expires_at"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _sign(raw: bytes, secret_key: str | bytes | None) -> bytes:
    if secret_key is None:
        return raw
    key = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
    digest = hmac.new(key, raw, hashlib.sha256).hexdigest().encode("ascii")
    return _HMAC_PREFIX + digest + b"\n" + base64.b64encode(raw)


def _verify(data: bytes, secret_key: str | bytes | None) -> bytes:
    if secret_key is None:
        return data
    if not data.startswith(_HMAC_PREFIX):
        raise CacheSerializerError("signed cache entry expected but entry is unsigned")
    try:
        digest, encoded = data[len(_HMAC_PREFIX) :].split(b"\n", 1)
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise CacheSerializerError("invalid signed cache entry envelope") from exc
    key = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
    expected = hmac.new(key, raw, hashlib.sha256).hexdigest().encode("ascii")
    if not hmac.compare_digest(digest, expected):
        raise CacheSerializerError("cache entry signature mismatch")
    return raw


class CacheBackend(ABC):
    """Abstract cache backend with TTL and namespace support."""

    def __init__(self, default_ttl: int | float | None = None, namespace: str = "default"):
        self.default_ttl = default_ttl
        self.namespace = namespace
        self._stats = CacheStats(backend=self.__class__.__name__)

    def get(self, key: str, default: Any = None, namespace: str | None = None) -> Any:
        ns = namespace or self.namespace
        try:
            entry = self._get_entry(ns, key)
            if entry is None:
                self._stats.misses += 1
                return default
            if entry.is_expired:
                self._stats.expired += 1
                self._stats.misses += 1
                self.delete(key, namespace=ns)
                return default
            self._stats.hits += 1
            return entry.value
        except Exception:
            self._stats.errors += 1
            self._stats.misses += 1
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | float | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ns = namespace or self.namespace
        effective_ttl = self.default_ttl if ttl is None else ttl
        expires_at = time.time() + float(effective_ttl) if effective_ttl else None
        entry = CacheEntry(key=key, namespace=ns, value=value, expires_at=expires_at, metadata=metadata or {})
        try:
            self._set_entry(entry)
            self._stats.sets += 1
        except Exception:
            self._stats.errors += 1
            raise

    def get_or_set(
        self,
        key: str,
        factory,
        ttl: int | float | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        sentinel = object()
        value = self.get(key, default=sentinel, namespace=namespace)
        if value is not sentinel:
            return value
        value = factory()
        self.set(key, value, ttl=ttl, namespace=namespace, metadata=metadata)
        return value

    def delete(self, key: str, namespace: str | None = None) -> None:
        try:
            self._delete_entry(namespace or self.namespace, key)
            self._stats.deletes += 1
        except Exception:
            self._stats.errors += 1
            raise

    def invalidate_namespace(self, namespace: str | None = None) -> int:
        try:
            removed = self._clear_namespace(namespace or self.namespace)
            self._stats.clears += 1
            return removed
        except Exception:
            self._stats.errors += 1
            raise

    def clear(self) -> int:
        try:
            removed = self._clear_all()
            self._stats.clears += 1
            return removed
        except Exception:
            self._stats.errors += 1
            raise

    def cleanup_expired(self) -> int:
        return self._cleanup_expired()

    def stats(self) -> CacheStats:
        stats = CacheStats(**self._stats.__dict__)
        stats.entries = self.entry_count()
        return stats

    def entry_count(self) -> int | None:
        return None

    def close(self) -> None:
        pass

    @abstractmethod
    def _get_entry(self, namespace: str, key: str) -> CacheEntry | None:
        raise NotImplementedError

    @abstractmethod
    def _set_entry(self, entry: CacheEntry) -> None:
        raise NotImplementedError

    @abstractmethod
    def _delete_entry(self, namespace: str, key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def _clear_namespace(self, namespace: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def _clear_all(self) -> int:
        raise NotImplementedError

    def _cleanup_expired(self) -> int:
        return 0
