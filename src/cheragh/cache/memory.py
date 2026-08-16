"""In-memory cache backend."""
from __future__ import annotations

from collections import OrderedDict

from .base import CacheBackend, CacheEntry


class MemoryCache(CacheBackend):
    """Thread-safe, bounded in-process cache.

    ``max_entries`` enables least-recently-used eviction and defaults to a
    production-safe finite bound. Pass ``None`` explicitly only for a trusted,
    short-lived workload where unbounded growth is intentional.
    """

    def __init__(
        self,
        default_ttl: int | float | None = None,
        namespace: str = "default",
        *,
        max_entries: int | None = 10_000,
    ):
        super().__init__(default_ttl=default_ttl, namespace=namespace)
        if max_entries is not None and (
            isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0
        ):
            raise ValueError("max_entries must be a positive integer or None")
        self.max_entries = max_entries
        self._data: OrderedDict[tuple[str, str], CacheEntry] = OrderedDict()

    def _get_entry(self, namespace: str, key: str) -> CacheEntry | None:
        identity = (namespace, key)
        with self._state_lock:
            entry = self._data.get(identity)
            if entry is not None:
                self._data.move_to_end(identity)
            return entry

    def _set_entry(self, entry: CacheEntry) -> None:
        identity = (entry.namespace, entry.key)
        with self._state_lock:
            self._data[identity] = entry
            self._data.move_to_end(identity)
            self._remove_expired_locked()
            if self.max_entries is not None:
                while len(self._data) > self.max_entries:
                    self._data.popitem(last=False)
                    self._stats.evictions += 1

    def _delete_entry(self, namespace: str, key: str) -> None:
        with self._state_lock:
            self._data.pop((namespace, key), None)

    def _delete_expired_entry(self, namespace: str, key: str, entry: CacheEntry) -> None:
        with self._state_lock:
            identity = (namespace, key)
            # Identity comparison prevents an expired reader from deleting a
            # fresh entry installed under the same key by another thread.
            if self._data.get(identity) is entry:
                self._data.pop(identity, None)

    def _clear_namespace(self, namespace: str) -> int:
        with self._state_lock:
            keys = [key for key in self._data if key[0] == namespace]
            for key in keys:
                self._data.pop(key, None)
            return len(keys)

    def _clear_all(self) -> int:
        with self._state_lock:
            count = len(self._data)
            self._data.clear()
            return count

    def _cleanup_expired(self) -> int:
        with self._state_lock:
            return self._remove_expired_locked(count_stats=True)

    def _remove_expired_locked(self, *, count_stats: bool = False) -> int:
        keys = [key for key, entry in self._data.items() if entry.is_expired]
        for key in keys:
            self._data.pop(key, None)
        if count_stats:
            self._stats.expired += len(keys)
        return len(keys)

    def entry_count(self) -> int:
        with self._state_lock:
            return len(self._data)
