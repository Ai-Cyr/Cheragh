"""SQLite cache backend."""
from __future__ import annotations

import math
from pathlib import Path
import sqlite3
import threading
import time

from .base import CacheBackend, CacheEntry, CacheSerializerError, dumps_entry, loads_entry


class SQLiteCache(CacheBackend):
    """Persistent local cache backed by SQLite.

    Parameters
    ----------
    serializer:
        ``"pickle"`` for full Python-object compatibility or ``"json"`` for
        simple JSON-compatible values.
    secret_key:
        Optional HMAC key. When set, entries are verified before deserialization.
    allow_pickle:
        Set to ``False`` to prevent unsafe pickle serialization/deserialization.
    max_entries:
        Global on-disk LRU bound. ``None`` explicitly disables eviction; the
        production default keeps accidental cache growth finite.
    """

    def __init__(
        self,
        path: str | Path,
        default_ttl: int | float | None = None,
        namespace: str = "default",
        *,
        max_entries: int | None = 10_000,
        serializer: str = "json",
        secret_key: str | bytes | None = None,
        allow_pickle: bool = False,
        allow_unsigned_pickle: bool = False,
        timeout: float = 5.0,
    ):
        super().__init__(default_ttl=default_ttl, namespace=namespace)
        self.path = Path(path)
        self.serializer = serializer
        self.secret_key = secret_key
        self.allow_pickle = allow_pickle
        if max_entries is not None and (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries <= 0
        ):
            raise ValueError("max_entries must be a positive integer or None")
        self.max_entries = max_entries
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive number")
        self.timeout = float(timeout)
        self._db_lock = threading.RLock()
        self._closed = False
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=self.timeout, check_same_thread=False)
        initial_expired = 0
        initial_evictions = 0
        with self._db_lock:
            # WAL allows concurrent readers while writes are committed. The
            # busy timeout prevents transient write contention from surfacing as
            # an immediate "database is locked" cache failure.
            self._conn.execute(f"PRAGMA busy_timeout={max(1, int(self.timeout * 1000))}")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    expires_at REAL,
                    payload BLOB NOT NULL,
                    PRIMARY KEY(namespace, key)
                )
                """
            )
            columns = {
                str(row[1])
                for row in self._conn.execute("PRAGMA table_info(cache_entries)")
            }
            if "accessed_at" not in columns:
                # Backward-compatible migration for caches created before the
                # persistent LRU bound. A nullable add is required by SQLite for
                # non-empty legacy tables, then every row is backfilled.
                self._conn.execute("ALTER TABLE cache_entries ADD COLUMN accessed_at REAL")
            self._conn.execute(
                "UPDATE cache_entries SET accessed_at=created_at WHERE accessed_at IS NULL"
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)")
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_lru
                ON cache_entries(accessed_at, created_at, namespace, key)
                """
            )
            initial_expired = self._purge_expired_locked(time.time())
            initial_evictions = self._enforce_bound_locked()
            self._conn.commit()
        self._increment_stat("expired", initial_expired)
        self._increment_stat("evictions", initial_evictions)

    def _get_entry(self, namespace: str, key: str) -> CacheEntry | None:
        with self._db_lock:
            self._ensure_open()
            row = self._conn.execute(
                "SELECT payload FROM cache_entries WHERE namespace=? AND key=?",
                (namespace, key),
            ).fetchone()
            if row is None:
                return None
            try:
                entry = loads_entry(
                    row[0],
                    serializer=self.serializer,
                    secret_key=self.secret_key,
                    allow_pickle=self.allow_pickle,
                )
                if entry.namespace != namespace or entry.key != key:
                    raise CacheSerializerError("cache entry identity does not match its SQLite key")
                if not entry.is_expired:
                    self._conn.execute(
                        "UPDATE cache_entries SET accessed_at=? WHERE namespace=? AND key=?",
                        (time.time(), namespace, key),
                    )
                    self._conn.commit()
                return entry
            except Exception:
                # Quarantine a corrupt/tampered value so every future request
                # does not pay the same decode failure. The exception is still
                # propagated to CacheBackend.get() and counted as an error.
                self._conn.execute(
                    "DELETE FROM cache_entries WHERE namespace=? AND key=?",
                    (namespace, key),
                )
                self._conn.commit()
                raise

    def _set_entry(self, entry: CacheEntry) -> None:
        payload = dumps_entry(
            entry,
            serializer=self.serializer,
            secret_key=self.secret_key,
            allow_pickle=self.allow_pickle,
        )
        with self._db_lock:
            self._ensure_open()
            expired_removed = 0
            evicted = 0
            with self._conn:
                expired_removed = self._purge_expired_locked(time.time())
                self._conn.execute(
                    """
                    INSERT INTO cache_entries(
                        namespace, key, created_at, accessed_at, expires_at, payload
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, key) DO UPDATE SET
                        created_at=excluded.created_at,
                        accessed_at=excluded.accessed_at,
                        expires_at=excluded.expires_at,
                        payload=excluded.payload
                    """,
                    (
                        entry.namespace,
                        entry.key,
                        entry.created_at,
                        entry.created_at,
                        entry.expires_at,
                        sqlite3.Binary(payload),
                    ),
                )
                evicted = self._enforce_bound_locked()
            self._increment_stat("expired", expired_removed)
            self._increment_stat("evictions", evicted)

    def _delete_entry(self, namespace: str, key: str) -> None:
        with self._db_lock:
            self._ensure_open()
            self._conn.execute("DELETE FROM cache_entries WHERE namespace=? AND key=?", (namespace, key))
            self._conn.commit()

    def _delete_expired_entry(self, namespace: str, key: str, entry: CacheEntry) -> None:
        with self._db_lock:
            self._ensure_open()
            self._conn.execute(
                """
                DELETE FROM cache_entries
                WHERE namespace=? AND key=? AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (namespace, key, time.time()),
            )
            self._conn.commit()

    def _clear_namespace(self, namespace: str) -> int:
        with self._db_lock:
            self._ensure_open()
            cur = self._conn.execute("DELETE FROM cache_entries WHERE namespace=?", (namespace,))
            self._conn.commit()
            return int(cur.rowcount or 0)

    def _clear_all(self) -> int:
        with self._db_lock:
            self._ensure_open()
            cur = self._conn.execute("DELETE FROM cache_entries")
            self._conn.commit()
            return int(cur.rowcount or 0)

    def _cleanup_expired(self) -> int:
        with self._db_lock:
            self._ensure_open()
            with self._conn:
                removed = self._purge_expired_locked(time.time())
            self._increment_stat("expired", removed)
            return removed

    def entry_count(self) -> int:
        with self._db_lock:
            self._ensure_open()
            row = self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
            return int(row[0] if row else 0)

    def close(self) -> None:
        with self._db_lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SQLiteCache is closed")

    def _purge_expired_locked(self, now: float) -> int:
        cursor = self._conn.execute(
            "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        return int(cursor.rowcount or 0)

    def _enforce_bound_locked(self) -> int:
        if self.max_entries is None:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
        excess = max(0, int(row[0] if row else 0) - self.max_entries)
        if excess == 0:
            return 0
        cursor = self._conn.execute(
            """
            DELETE FROM cache_entries
            WHERE rowid IN (
                SELECT rowid
                FROM cache_entries
                ORDER BY accessed_at ASC, created_at ASC, namespace ASC, key ASC
                LIMIT ?
            )
            """,
            (excess,),
        )
        return int(cursor.rowcount or 0)
