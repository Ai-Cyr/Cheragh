import json
import math
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cheragh import Document, EmbeddingModel, HashingEmbedding, MemoryVectorStore, index_from_config, index_path
from cheragh.cache import MemoryCache, SQLiteCache, build_cache_backend
from cheragh.ingestion import load_documents
from cheragh.indexing import (
    IndexManifest,
    _index_lock,
    _remove_abandoned_lock,
    load_manifest,
    save_manifest,
    scan_indexable_files,
)
from cheragh.tracing import RAGTrace, append_trace_jsonl
from cheragh.vectorstores.memory import _fsync_directory as _real_store_fsync_directory


class _ControlledEmbedding(EmbeddingModel):
    dimension = 2

    def __init__(self, *, document_vectors=None, query_vector=None):
        self.document_vectors = document_vectors
        self.query_vector = query_vector
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        if self.document_vectors is not None:
            return np.asarray(self.document_vectors)
        return np.ones((len(texts), self.dimension), dtype=float)

    def embed_query(self, text):
        if self.query_vector is not None:
            return np.asarray(self.query_vector)
        return np.ones(self.dimension, dtype=float)

    def get_fingerprint(self):
        return "tests.ControlledEmbedding::2"


class _SimulatedProcessDeath(BaseException):
    pass


class VectorStoreReliabilityTests(unittest.TestCase):
    def test_malformed_documents_are_rejected_before_embedding(self):
        invalid_documents = [
            [object()],
            [Document(3)],  # type: ignore[arg-type]
            [Document("")],
            [Document("content", metadata=None)],  # type: ignore[arg-type]
            [Document("content", doc_id=3)],  # type: ignore[arg-type]
            [Document("content", score=True)],
            [Document("content", score=math.nan)],
        ]
        for documents in invalid_documents:
            with self.subTest(documents=documents):
                embedding = _ControlledEmbedding()
                store = MemoryVectorStore(embedding)
                with self.assertRaises((TypeError, ValueError)):
                    store.add_documents(documents)  # type: ignore[arg-type]
                self.assertEqual(embedding.document_calls, 0)
                self.assertEqual(store.documents, [])
                self.assertIsNone(store.embeddings)

    def test_non_finite_and_wrong_dimension_embeddings_fail_closed(self):
        store = MemoryVectorStore(
            _ControlledEmbedding(document_vectors=[[math.nan, 0.0]])
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            store.add_documents([Document("alpha")])

        store = MemoryVectorStore(_ControlledEmbedding(query_vector=[1.0, 2.0, 3.0]))
        store.add_documents([Document("alpha")])
        with self.assertRaisesRegex(ValueError, "query has 3"):
            store.similarity_search("query")

        store.embedding_model.query_vector = [math.inf, 0.0]  # type: ignore[attr-defined]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            store.similarity_search("query")

        for invalid in (
            np.asarray([[True, False]]),
            np.asarray([[1 + 2j, 0j]]),
        ):
            with self.subTest(dtype=invalid.dtype):
                store = MemoryVectorStore(_ControlledEmbedding(document_vectors=invalid))
                with self.assertRaisesRegex(ValueError, "real numeric"):
                    store.add_documents([Document("alpha")])

    def test_search_results_deep_copy_nested_metadata(self):
        store = MemoryVectorStore(HashingEmbedding(8))
        store.add_documents(
            [Document("alpha", metadata={"nested": {"value": 1}}, doc_id="a")]
        )
        result = store.similarity_search("alpha", top_k=1)
        result[0].metadata["nested"]["value"] = 2
        self.assertEqual(store.documents[0].metadata["nested"]["value"], 1)

    def test_persisted_same_shape_tampering_is_detected_by_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            store = MemoryVectorStore(HashingEmbedding(8))
            store.add_documents([Document("alpha", doc_id="a")])
            store.save(path)
            vectors = np.load(path / "embeddings.npy", allow_pickle=False)
            vectors[0, 0] += 0.125
            with (path / "embeddings.npy").open("wb") as file:
                np.save(file, vectors, allow_pickle=False)

            with self.assertRaisesRegex(ValueError, "embeddings checksum mismatch"):
                with patch("numpy.load") as load:
                    MemoryVectorStore.load(path)
            load.assert_not_called()

    def test_non_empty_zero_dimension_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            embedding = _ControlledEmbedding()
            store = MemoryVectorStore(embedding)
            store.documents = [Document("alpha", doc_id="a")]
            store.embeddings = np.zeros((1, 0))
            store.save(path)

            with self.assertRaisesRegex(ValueError, "zero-dimensional"):
                MemoryVectorStore.load(path, embedding)

    def test_interrupted_save_rolls_back_the_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            store = MemoryVectorStore(HashingEmbedding(8))
            store.add_documents([Document("alpha", doc_id="a")])
            store.save(path)
            store.add_documents([Document("beta", doc_id="b")])
            real_replace = os.replace
            calls = 0

            def fail_second_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated interrupted commit")
                return real_replace(source, destination)

            with patch("cheragh.vectorstores.memory.os.replace", side_effect=fail_second_replace):
                with self.assertRaisesRegex(OSError, "interrupted commit"):
                    store.save(path)

            loaded = MemoryVectorStore.load(path)
            self.assertEqual([document.doc_id for document in loaded.documents], ["a"])
            self.assertEqual(list(path.glob(".*.tmp")), [])

    def test_post_manifest_fsync_failure_restores_complete_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            store = MemoryVectorStore(HashingEmbedding(8))
            store.add_documents([Document("alpha", doc_id="a")])
            store.save(path)
            persisted_names = ("documents.jsonl", "embeddings.npy", "manifest.json")
            before = {name: (path / name).read_bytes() for name in persisted_names}
            store.add_documents([Document("beta", doc_id="b")])

            fsync_calls = 0

            def fail_once_after_manifest_replace(directory):
                nonlocal fsync_calls
                fsync_calls += 1
                # Existing snapshot save: backup entries, new data entries,
                # then the newly replaced manifest entry.
                if fsync_calls == 3:
                    raise OSError("post-manifest directory fsync failed")
                return _real_store_fsync_directory(directory)

            with patch(
                "cheragh.vectorstores.memory._fsync_directory",
                side_effect=fail_once_after_manifest_replace,
            ), patch(
                "cheragh.vectorstores.memory.os.replace",
                wraps=os.replace,
            ) as replace:
                with self.assertRaisesRegex(OSError, "post-manifest"):
                    store.save(path)

            after = {name: (path / name).read_bytes() for name in persisted_names}
            self.assertEqual(after, before)
            self.assertEqual(fsync_calls, 4)
            restored_destinations = [
                Path(call.args[1]).name for call in replace.call_args_list[-3:]
            ]
            self.assertEqual(
                restored_destinations,
                ["documents.jsonl", "embeddings.npy", "manifest.json"],
            )
            loaded = MemoryVectorStore.load(path)
            self.assertEqual([document.doc_id for document in loaded.documents], ["a"])

    def test_abrupt_process_death_remains_readable_via_previous_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            store = MemoryVectorStore(HashingEmbedding(8))
            store.add_documents([Document("alpha", doc_id="a")])
            store.save(path)
            store.add_documents([Document("beta", doc_id="b")])
            real_replace = os.replace
            calls = 0

            def crash_second_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise _SimulatedProcessDeath()
                return real_replace(source, destination)

            with patch("cheragh.vectorstores.memory.os.replace", side_effect=crash_second_replace):
                with self.assertRaises(_SimulatedProcessDeath):
                    store.save(path)

            loaded = MemoryVectorStore.load(path)
            self.assertEqual([document.doc_id for document in loaded.documents], ["a"])
            self.assertTrue(list(path.glob(".documents.*.snapshot")))

            store.save(path)
            repaired = MemoryVectorStore.load(path)
            self.assertEqual([document.doc_id for document in repaired.documents], ["a", "b"])
            self.assertEqual(list(path.glob(".*.snapshot")), [])

    def test_legacy_snapshot_is_upgraded_before_risky_replacement(self):
        for legacy_schema in (1, 2):
            with self.subTest(schema=legacy_schema), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                store = MemoryVectorStore(HashingEmbedding(8))
                store.add_documents([Document("alpha", doc_id="a")])
                store.save(path)
                manifest_path = path / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["schema_version"] = legacy_schema
                manifest.pop("documents")
                manifest.pop("embeddings")
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                store.add_documents([Document("beta", doc_id="b")])
                real_replace = os.replace
                calls = 0

                def crash_after_documents(source, destination):
                    nonlocal calls
                    calls += 1
                    # 1: atomic legacy manifest upgrade, 2: documents, 3: embeddings.
                    if calls == 3:
                        raise _SimulatedProcessDeath()
                    return real_replace(source, destination)

                with patch(
                    "cheragh.vectorstores.memory.os.replace",
                    side_effect=crash_after_documents,
                ):
                    with self.assertRaises(_SimulatedProcessDeath):
                        store.save(path)

                upgraded = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(upgraded["schema_version"], 3)
                loaded = MemoryVectorStore.load(path)
                self.assertEqual([document.doc_id for document in loaded.documents], ["a"])


class IndexingReliabilityTests(unittest.TestCase):
    def test_scan_and_ingestion_reject_nested_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            nested = source / "nested"
            outside = base / "outside"
            nested.mkdir(parents=True)
            outside.mkdir()
            secret = outside / "secret.txt"
            secret.write_text("must not be indexed", encoding="utf-8")
            escape = nested / "escape.txt"
            try:
                escape.symlink_to(secret)
            except (NotImplementedError, OSError) as exc:  # pragma: no cover - platform
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaisesRegex(ValueError, "outside source root"):
                scan_indexable_files(source)
            with self.assertRaisesRegex(ValueError, "outside source root"):
                load_documents(source)

    def test_indexing_rechecks_containment_before_loading_changed_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            nested = source / "nested"
            outside = base / "outside"
            nested.mkdir(parents=True)
            outside.mkdir()
            (nested / "document.txt").write_text("inside", encoding="utf-8")
            (outside / "document.txt").write_text("outside", encoding="utf-8")
            output = base / "index"

            from cheragh.indexing import plan_incremental_update as real_plan

            def swap_parent_after_scan(previous, current, *, force=False):
                plan = real_plan(previous, current, force=force)
                nested.rename(source / "nested-original")
                try:
                    nested.symlink_to(outside, target_is_directory=True)
                except (NotImplementedError, OSError) as exc:  # pragma: no cover - platform
                    self.skipTest(f"symlinks unavailable: {exc}")
                return plan

            with patch(
                "cheragh.indexing.plan_incremental_update",
                side_effect=swap_parent_after_scan,
            ), self.assertRaisesRegex(ValueError, "outside source root"):
                index_path(source, output, embedding_model=HashingEmbedding(8))

            self.assertFalse((output / "manifest.json").exists())

    def test_index_from_config_resolves_environment_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "rag.json"
            config.write_text(
                json.dumps(
                    {
                        "ingestion": {"path": "${CHERAGH_TEST_CORPUS}"},
                        "vectorstore": {
                            "type": "memory",
                            "path": "${CHERAGH_TEST_INDEX}",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "CHERAGH_TEST_CORPUS": "corpus-from-env",
                    "CHERAGH_TEST_INDEX": "index-from-env",
                },
            ), patch("cheragh.indexing.index_path", return_value={}) as build:
                index_from_config(config, embedding_model=HashingEmbedding(8))

            self.assertEqual(build.call_args.args[0], (root / "corpus-from-env").resolve())
            self.assertEqual(build.call_args.args[1], (root / "index-from-env").resolve())

    def test_stale_lock_file_does_not_block_a_new_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / ".index.lock").write_text("dead-owner", encoding="utf-8")
            with _index_lock(directory, enabled=True, timeout=0):
                self.assertTrue((directory / ".index.lock").exists())

    def test_live_lock_times_out_without_deleting_the_owner_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with _index_lock(directory, enabled=True, timeout=0):
                with self.assertRaises(TimeoutError):
                    with _index_lock(directory, enabled=True, timeout=0):
                        pass
                self.assertTrue((directory / ".index.lock").exists())

    def test_portable_lock_fallback_removes_a_dead_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / ".index.lock"
            lock.write_text(
                json.dumps({"pid": 999_999_999, "acquired_at": time.time()}),
                encoding="ascii",
            )
            self.assertTrue(_remove_abandoned_lock(lock))
            self.assertFalse(lock.exists())

    def test_manifest_replacement_failure_preserves_previous_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            original = IndexManifest(metadata={"generation": 1})
            save_manifest(directory, original)
            before = (directory / "index_manifest.json").read_bytes()
            with patch("cheragh.indexing.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    save_manifest(directory, IndexManifest(metadata={"generation": 2}))
            self.assertEqual((directory / "index_manifest.json").read_bytes(), before)
            self.assertEqual(list(directory.glob(".index_manifest.*.tmp")), [])

    def test_corrupt_manifest_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "index_manifest.json").write_text('{"files": [}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid index manifest"):
                load_manifest(directory)

    def test_missing_store_files_force_a_complete_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "docs"
            output = Path(tmp) / "index"
            source.mkdir()
            (source / "a.txt").write_text("alpha", encoding="utf-8")
            index_path(source, output, embedding_model=HashingEmbedding(8))
            (output / "embeddings.npy").unlink()

            result = index_path(source, output, embedding_model=HashingEmbedding(8))

            self.assertTrue(result["store_snapshot_changed"])
            self.assertEqual(result["changed_files"], 1)
            self.assertEqual(len(MemoryVectorStore.load(output).documents), 1)

    def test_source_change_aborts_before_replacing_a_valid_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "docs"
            output = Path(tmp) / "index"
            source.mkdir()
            file_path = source / "a.txt"
            file_path.write_text("alpha", encoding="utf-8")
            index_path(source, output, embedding_model=HashingEmbedding(8))
            before = {
                name: (output / name).read_bytes()
                for name in ("documents.jsonl", "embeddings.npy", "manifest.json", "index_manifest.json")
            }
            file_path.write_text("beta", encoding="utf-8")
            from cheragh.indexing import file_sha256 as real_file_sha256

            source_hash_calls = 0

            def unstable_digest(path):
                nonlocal source_hash_calls
                if Path(path) == file_path:
                    source_hash_calls += 1
                    if source_hash_calls == 2:
                        return "0" * 64
                return real_file_sha256(path)

            with patch("cheragh.indexing.file_sha256", side_effect=unstable_digest):
                with self.assertRaisesRegex(RuntimeError, "Source changed during indexing"):
                    index_path(source, output, embedding_model=HashingEmbedding(8))

            after = {name: (output / name).read_bytes() for name in before}
            self.assertEqual(after, before)


class CacheReliabilityTests(unittest.TestCase):
    def test_memory_cache_is_bounded_lru(self):
        cache = MemoryCache(max_entries=2)
        cache.set("a", 1)
        cache.set("b", 2)
        self.assertEqual(cache.get("a"), 1)
        cache.set("c", 3)

        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("c"), 3)
        self.assertEqual(cache.entry_count(), 2)
        self.assertEqual(cache.stats().evictions, 1)

    def test_empty_namespace_is_isolated_from_default(self):
        cache = MemoryCache(namespace="default")
        cache.set("key", "default")
        cache.set("key", "empty", namespace="")
        self.assertEqual(cache.get("key"), "default")
        self.assertEqual(cache.get("key", namespace=""), "empty")

    def test_invalid_ttls_are_rejected(self):
        for invalid in (-1, math.nan, math.inf, True, "1"):
            with self.subTest(default_ttl=invalid), self.assertRaises((TypeError, ValueError)):
                MemoryCache(default_ttl=invalid)  # type: ignore[arg-type]
        cache = MemoryCache()
        for invalid in (-1, math.nan, math.inf, True, "1"):
            with self.subTest(ttl=invalid), self.assertRaises((TypeError, ValueError)):
                cache.set("key", "value", ttl=invalid)  # type: ignore[arg-type]

    def test_backend_factory_rejects_lossy_numeric_coercions(self):
        invalid_configs = (
            {"backend": "memory", "ttl": True},
            {"backend": "memory", "ttl": math.nan},
            {"backend": "memory", "max_entries": 1.5},
            {"backend": "memory", "max_entries": True},
            {"backend": "sqlite", "max_entries": 1.5},
            {"backend": "sqlite", "max_entries": 0},
        )
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_cache_backend(config)
        with self.assertRaises(ValueError):
            SQLiteCache(":memory:", timeout=math.nan)

    def test_get_or_set_is_single_flight_per_key(self):
        cache = MemoryCache()
        barrier = threading.Barrier(8)
        calls = 0
        calls_lock = threading.Lock()
        results = []
        errors = []

        def factory():
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.02)
            return "value"

        def worker():
            try:
                barrier.wait(timeout=2)
                results.append(cache.get_or_set("same", factory))
            except BaseException as exc:  # pragma: no cover - assertion aid
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(errors, [])
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(results, ["value"] * 8)
        self.assertEqual(calls, 1)

    def test_recursive_same_key_factory_fails_instead_of_deadlocking(self):
        cache = MemoryCache()
        with self.assertRaisesRegex(RuntimeError, "recursive cache factory"):
            cache.get_or_set(
                "same",
                lambda: cache.get_or_set("same", lambda: "unreachable"),
            )

    def test_cold_get_or_set_counts_one_request_miss(self):
        cache = MemoryCache()
        self.assertEqual(cache.get_or_set("key", lambda: "value"), "value")
        stats = cache.stats()
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.hits, 0)
        self.assertEqual(stats.requests, 1)

        self.assertEqual(cache.get_or_set("key", lambda: "other"), "value")
        stats = cache.stats()
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.requests, 2)

    def test_sqlite_shared_connection_is_thread_safe_and_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = SQLiteCache(Path(tmp) / "cache.sqlite")
            barrier = threading.Barrier(6)
            errors = []

            def worker(worker_id):
                try:
                    barrier.wait(timeout=2)
                    for index in range(30):
                        key = f"{worker_id}:{index}"
                        cache.set(key, index)
                        if cache.get(key) != index:
                            raise AssertionError(key)
                except BaseException as exc:  # pragma: no cover - assertion aid
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(worker_id,)) for worker_id in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            self.assertEqual(errors, [])
            self.assertEqual(cache.entry_count(), 180)
            cache.close()
            cache.close()

    def test_sqlite_is_bounded_by_persistent_lru(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = SQLiteCache(Path(tmp) / "cache.sqlite", max_entries=2)
            cache.set("a", 1)
            cache.set("b", 2)
            with cache._db_lock:
                cache._conn.execute(
                    "UPDATE cache_entries SET accessed_at=1 WHERE key='a'"
                )
                cache._conn.execute(
                    "UPDATE cache_entries SET accessed_at=2 WHERE key='b'"
                )
                cache._conn.commit()

            self.assertEqual(cache.get("a"), 1)
            cache.set("c", 3)

            self.assertIsNone(cache.get("b"))
            self.assertEqual(cache.get("a"), 1)
            self.assertEqual(cache.get("c"), 3)
            self.assertEqual(cache.entry_count(), 2)
            self.assertEqual(cache.stats().evictions, 1)
            cache.close()

    def test_sqlite_write_purges_expired_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = SQLiteCache(Path(tmp) / "cache.sqlite")
            cache.set("expired", "old", ttl=60)
            with cache._db_lock:
                cache._conn.execute(
                    "UPDATE cache_entries SET expires_at=0 WHERE key='expired'"
                )
                cache._conn.commit()

            cache.set("fresh", "new")

            self.assertEqual(cache.entry_count(), 1)
            self.assertEqual(cache.get("fresh"), "new")
            self.assertEqual(cache.stats().expired, 1)
            cache.close()

    def test_sqlite_bound_is_preserved_under_concurrent_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = SQLiteCache(Path(tmp) / "cache.sqlite", max_entries=25)
            barrier = threading.Barrier(4)
            errors = []

            def worker(worker_id):
                try:
                    barrier.wait(timeout=2)
                    for index in range(50):
                        cache.set(f"{worker_id}:{index}", index)
                except BaseException as exc:  # pragma: no cover - assertion aid
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            self.assertEqual(errors, [])
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(cache.entry_count(), 25)
            self.assertEqual(cache.stats().evictions, 175)
            cache.close()

    def test_sqlite_migrates_legacy_schema_for_lru(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.sqlite"
            connection = sqlite3.connect(path)
            connection.execute(
                """
                CREATE TABLE cache_entries (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    payload BLOB NOT NULL,
                    PRIMARY KEY(namespace, key)
                )
                """
            )
            connection.commit()
            connection.close()

            cache = SQLiteCache(path)
            with cache._db_lock:
                columns = {
                    row[1]
                    for row in cache._conn.execute("PRAGMA table_info(cache_entries)")
                }
                indexes = {
                    row[1]
                    for row in cache._conn.execute("PRAGMA index_list(cache_entries)")
                }
            self.assertIn("accessed_at", columns)
            self.assertIn("idx_cache_lru", indexes)
            cache.close()

    def test_sqlite_applies_bound_immediately_when_reopened(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.sqlite"
            unbounded = SQLiteCache(path, max_entries=None)
            for index in range(5):
                unbounded.set(str(index), index)
            unbounded.close()

            bounded = SQLiteCache(path, max_entries=2)
            self.assertEqual(bounded.entry_count(), 2)
            self.assertEqual(bounded.stats().evictions, 3)
            self.assertEqual(bounded.get("3"), 3)
            self.assertEqual(bounded.get("4"), 4)
            bounded.close()

    def test_corrupt_sqlite_entry_is_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.sqlite"
            cache = SQLiteCache(path)
            cache.set("key", "value")
            with cache._db_lock:
                cache._conn.execute(
                    "UPDATE cache_entries SET payload=? WHERE namespace=? AND key=?",
                    (sqlite3.Binary(b"not-json"), "default", "key"),
                )
                cache._conn.commit()

            self.assertIsNone(cache.get("key"))
            self.assertEqual(cache.entry_count(), 0)
            self.assertEqual(cache.stats().errors, 1)
            cache.close()

    def test_backend_factory_supports_memory_bound(self):
        cache = build_cache_backend({"backend": "memory", "max_entries": 1})
        self.assertIsInstance(cache, MemoryCache)
        self.assertEqual(MemoryCache().max_entries, 10_000)
        cache.set("a", 1)  # type: ignore[union-attr]
        cache.set("b", 2)  # type: ignore[union-attr]
        self.assertIsNone(cache.get("a"))  # type: ignore[union-attr]

    def test_backend_factory_forwards_sqlite_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = build_cache_backend(
                {
                    "backend": "sqlite",
                    "path": str(Path(tmp) / "cache.sqlite"),
                    "max_entries": "2",
                }
            )
            self.assertIsInstance(cache, SQLiteCache)
            self.assertEqual(cache.max_entries, 2)
            cache.set("a", 1)
            cache.set("b", 2)
            cache.set("c", 3)
            self.assertEqual(cache.entry_count(), 2)
            cache.close()


class TraceReliabilityTests(unittest.TestCase):
    def test_concurrent_appends_produce_complete_json_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            barrier = threading.Barrier(12)
            errors = []

            def worker(index):
                try:
                    trace = RAGTrace(request_id=f"request-{index}")
                    trace.metadata["payload"] = "x" * 4096
                    trace.finish()
                    barrier.wait(timeout=2)
                    append_trace_jsonl(path, trace)
                except BaseException as exc:  # pragma: no cover - assertion aid
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(index,)) for index in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)

            self.assertEqual(errors, [])
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 12)
            self.assertEqual({record["request_id"] for record in records}, {f"request-{i}" for i in range(12)})

    def test_partial_os_writes_are_completed_and_fd_is_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            trace = RAGTrace(request_id="partial")
            trace.metadata["payload"] = "x" * 256
            trace.finish()
            real_write = os.write
            real_open = os.open
            real_close = os.close
            opened = []
            closed = []

            def partial_write(fd, data):
                return real_write(fd, data[:7])

            def recording_open(*args, **kwargs):
                fd = real_open(*args, **kwargs)
                opened.append(fd)
                return fd

            def recording_close(fd):
                closed.append(fd)
                return real_close(fd)

            with patch("cheragh.tracing.os.open", side_effect=recording_open), patch(
                "cheragh.tracing.os.write", side_effect=partial_write
            ), patch("cheragh.tracing.os.close", side_effect=recording_close):
                append_trace_jsonl(path, trace, durable=True)

            self.assertEqual(opened, closed)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["request_id"], "partial")

    def test_non_finite_trace_data_is_rejected_before_opening_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            trace = RAGTrace()
            trace.metadata["invalid"] = math.nan
            with self.assertRaises(ValueError):
                append_trace_jsonl(path, trace)
            self.assertFalse(path.exists())

    def test_trace_duration_uses_monotonic_clock(self):
        trace = RAGTrace(started_at_unix=1000.0)
        with patch("cheragh.tracing.time", return_value=-1000.0):
            trace.finish()
        self.assertEqual(trace.ended_at_unix, -1000.0)
        self.assertGreaterEqual(trace.duration_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
