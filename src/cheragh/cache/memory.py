"""In-memory cache backend."""
from __future__ import annotations

from .base import CacheBackend, CacheEntry


class MemoryCache(CacheBackend):
    """Thread-light in-process cache backend for tests and short-lived apps."""

    def __init__(self, default_ttl: int | float | None = None, namespace: str = "default"):
        super().__init__(default_ttl=default_ttl, namespace=namespace)
        self._data: dict[tuple[str, str], CacheEntry] = {}

    def _get_entry(self, namespace: str, key: str) -> CacheEntry | None:
        return self._data.get((namespace, key))

    def _set_entry(self, entry: CacheEntry) -> None:
        self._data[(entry.namespace, entry.key)] = entry

    def _delete_entry(self, namespace: str, key: str) -> None:
        self._data.pop((namespace, key), None)

    def _clear_namespace(self, namespace: str) -> int:
        keys = [key for key in self._data if key[0] == namespace]
        for key in keys:
            self._data.pop(key, None)
        return len(keys)

    def _clear_all(self) -> int:
        count = len(self._data)
        self._data.clear()
        return count

    def _cleanup_expired(self) -> int:
        keys = [key for key, entry in self._data.items() if entry.is_expired]
        for key in keys:
            self._data.pop(key, None)
        self._stats.expired += len(keys)
        return len(keys)

    def entry_count(self) -> int:
        return len(self._data)
