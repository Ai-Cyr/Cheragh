import os
import tempfile
import time
import unittest
from pathlib import Path

from cheragh import Document, HashingEmbedding, RAGEngine, StaticLLMClient
from cheragh.cache import (
    CachedLLMClient,
    MemoryCache,
    SQLiteCache,
    build_cache_backend,
    make_cache_key,
)


class CountingLLM(StaticLLMClient):
    def __init__(self):
        super().__init__("cached answer [source: doc-1]")
        self.calls = 0

    def generate(self, prompt: str, **kwargs):
        self.calls += 1
        return super().generate(prompt, **kwargs)


class V071CacheTests(unittest.TestCase):
    def test_memory_cache_ttl_and_invalidation(self):
        cache = MemoryCache(default_ttl=0.05)
        key = make_cache_key("a", 1)
        cache.set(key, {"value": 1})
        self.assertEqual(cache.get(key), {"value": 1})
        time.sleep(0.07)
        self.assertIsNone(cache.get(key))
        self.assertGreaterEqual(cache.stats().expired, 1)
        cache.set("x", 1, namespace="n")
        self.assertEqual(cache.invalidate_namespace("n"), 1)
        self.assertIsNone(cache.get("x", namespace="n"))

    def test_sqlite_cache_persists_and_clears(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.sqlite"
            cache = SQLiteCache(path)
            cache.set("k", [1, 2, 3], namespace="ns")
            cache.close()
            reopened = SQLiteCache(path)
            self.assertEqual(reopened.get("k", namespace="ns"), [1, 2, 3])
            self.assertEqual(reopened.entry_count(), 1)
            self.assertEqual(reopened.invalidate_namespace("ns"), 1)
            self.assertIsNone(reopened.get("k", namespace="ns"))

    def test_cached_llm_client(self):
        cache = MemoryCache()
        llm = CountingLLM()
        cached = CachedLLMClient(llm, cache)
        prompt = "Question"
        self.assertEqual(cached.generate(prompt), "cached answer [source: doc-1]")
        self.assertEqual(cached.generate(prompt), "cached answer [source: doc-1]")
        self.assertEqual(llm.calls, 1)
        self.assertGreaterEqual(cache.stats().hits, 1)

    def test_rag_engine_from_documents_with_cache_backend(self):
        cache = MemoryCache()
        llm = CountingLLM()
        docs = [Document("Alpha contient la clause A.", doc_id="doc-1")]
        engine = RAGEngine.from_documents(
            docs,
            embedding_model=HashingEmbedding(64),
            retriever_type="memory",
            llm_client=llm,
            cache_backend=cache,
        )
        engine.ask("clause Alpha")
        engine.ask("clause Alpha")
        self.assertEqual(llm.calls, 1)
        stats = cache.stats().to_dict()
        self.assertGreater(stats["hits"], 0)
        self.assertGreater(stats["sets"], 0)

    def test_build_cache_backend_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = build_cache_backend({"enabled": True, "backend": "sqlite", "path": str(Path(tmp) / "c.sqlite"), "ttl": 60})
            self.assertIsInstance(cache, SQLiteCache)
            cache.set("k", "v")
            self.assertEqual(cache.get("k"), "v")


if __name__ == "__main__":
    unittest.main()
