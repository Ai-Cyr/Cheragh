"""SQLite cache backend."""
from __future__ import annotations

from pathlib import Path
import sqlite3
import time

from .base import CacheBackend, CacheEntry, dumps_entry, loads_entry


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
    """

    def __init__(
        self,
        path: str | Path,
        default_ttl: int | float | None = None,
        namespace: str = "default",
        *,
        serializer: str = "json",
        secret_key: str | bytes | None = None,
        allow_pickle: bool = False,
        allow_unsigned_pickle: bool = False,
    ):
        super().__init__(default_ttl=default_ttl, namespace=namespace)
        self.path = Path(path)
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL,
                payload BLOB NOT NULL,
                PRIMARY KEY(namespace, key)
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)")
        self._conn.commit()

    def _get_entry(self, namespace: str, key: str) -> CacheEntry | None:
        row = self._conn.execute(
            "SELECT payload FROM cache_entries WHERE namespace=? AND key=?",
            (namespace, key),
        ).fetchone()
        if row is None:
            return None
        return loads_entry(
            row[0],
            serializer=self.serializer,
            secret_key=self.secret_key,
            allow_pickle=self.allow_pickle,
        )

    def _set_entry(self, entry: CacheEntry) -> None:
        payload = dumps_entry(
            entry,
            serializer=self.serializer,
            secret_key=self.secret_key,
            allow_pickle=self.allow_pickle,
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cache_entries(namespace, key, created_at, expires_at, payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            (entry.namespace, entry.key, entry.created_at, entry.expires_at, sqlite3.Binary(payload)),
        )
        self._conn.commit()

    def _delete_entry(self, namespace: str, key: str) -> None:
        self._conn.execute("DELETE FROM cache_entries WHERE namespace=? AND key=?", (namespace, key))
        self._conn.commit()

    def _clear_namespace(self, namespace: str) -> int:
        cur = self._conn.execute("DELETE FROM cache_entries WHERE namespace=?", (namespace,))
        self._conn.commit()
        return int(cur.rowcount or 0)

    def _clear_all(self) -> int:
        cur = self._conn.execute("DELETE FROM cache_entries")
        self._conn.commit()
        return int(cur.rowcount or 0)

    def _cleanup_expired(self) -> int:
        now = time.time()
        cur = self._conn.execute("DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
        self._conn.commit()
        removed = int(cur.rowcount or 0)
        self._stats.expired += removed
        return removed

    def entry_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
        return int(row[0] if row else 0)

    def close(self) -> None:
        self._conn.close()
