"""Cache identities must retain the metadata and configuration value types."""

import gc
import os
import subprocess
import sys
import textwrap
import weakref

import pytest

from cheragh import Document, HashingEmbedding
from cheragh.cache import CachedEmbeddingModel, CachedLLMClient, CachedReranker, CachedRetriever, MemoryCache
from cheragh.reranking import BaseReranker
from cheragh.vectorstores import MemoryVectorStore


@pytest.mark.parametrize("first,second", [(1.0, "1.0"), ([1], (1,)), ({1: "a"}, {"1": "a"})])
def test_retrieval_cache_distinguishes_metadata_types(first, second):
    cache = MemoryCache()
    embedder = HashingEmbedding(8)
    stores = [MemoryVectorStore(embedder), MemoryVectorStore(embedder)]
    for store, value in zip(stores, (first, second)):
        store.add_documents([Document("same", doc_id="same", metadata={"value": value})])
    a, b = [CachedRetriever(store.as_retriever(), cache) for store in stores]
    assert a.retrieve("same")[0].metadata["value"] == first
    actual = b.retrieve("same")[0].metadata["value"]
    assert type(actual) is type(second)
    assert actual == second
    assert cache.stats().misses == 2


def test_reranking_cache_does_not_reuse_results_for_different_metadata_types():
    class TypeReranker(BaseReranker):
        def rerank(self, query, documents, top_k=5):
            return [Document(str(type(doc.metadata["value"]).__name__), doc_id=doc.doc_id) for doc in documents]

    cached = CachedReranker(TypeReranker(), MemoryCache())
    first = cached.rerank("query", [Document("same", doc_id="same", metadata={"value": 1.0})])
    second = cached.rerank("query", [Document("same", doc_id="same", metadata={"value": "1.0"})])
    assert first[0].content == "float"
    assert second[0].content == "str"


def test_opaque_instances_remain_isolated_when_object_ids_are_reused(monkeypatch):
    from cheragh.cache import decorators

    class OpaqueClient:
        def __init__(self, answer):
            self._answer = answer

        def generate(self, prompt, **kwargs):
            return self._answer

    monkeypatch.setattr(decorators, "id", lambda _: 17, raising=False)
    cache = MemoryCache()
    first = OpaqueClient("one")
    cached_first = CachedLLMClient(first, cache)
    assert cached_first.generate("same prompt") == "one"
    second = OpaqueClient("two")
    cached_second = CachedLLMClient(second, cache)
    assert cached_second.generate("same prompt") == "two"
    fingerprint = cached_second.fingerprint
    reference = weakref.ref(first)
    del first, cached_first
    gc.collect()
    assert reference() is None, "the identity registry must not keep clients alive"
    assert cached_second.fingerprint == fingerprint
    assert cached_second.generate("same prompt") == "two"
    assert cache.stats().hits == 1


def test_non_weakref_opaque_values_prefer_misses_but_explicit_fingerprints_stay_stable():
    class OpaqueClient:
        __slots__ = ("calls",)

        def __init__(self):
            self.calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            return str(self.calls)

    cache = MemoryCache()
    client = OpaqueClient()
    cached = CachedLLMClient(client, cache)
    assert cached.generate("prompt") == "1"
    assert cached.generate("prompt") == "2"
    explicit = CachedLLMClient(client, cache, fingerprint="external-version")
    assert explicit.fingerprint == "external-version"
    assert explicit.generate("prompt") == explicit.generate("prompt") == "3"
    first = CachedEmbeddingModel(HashingEmbedding(8), cache)
    second = CachedEmbeddingModel(HashingEmbedding(8), cache)
    assert first.cache_fingerprint == second.cache_fingerprint


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX worker-fork contract")
def test_inherited_cache_wrapper_does_not_share_opaque_provider_identity_between_workers(tmp_path):
    script = textwrap.dedent("""
        import os
        import sys
        from cheragh.cache import CachedLLMClient, MemoryCache, SQLiteCache

        class OpaqueClient:
            def __init__(self):
                self._answer = 'parent'
            def generate(self, prompt, **kwargs):
                return self._answer

        client = OpaqueClient()
        wrapped = CachedLLMClient(client, MemoryCache())
        # Initialize the registry before forking: child object addresses and
        # the wrapper's previous identity are exact copies of the parent's.
        assert wrapped.generate('prime') == 'parent'
        for answer in ('first worker', 'second worker'):
            pid = os.fork()
            if pid == 0:
                try:
                    client._answer = answer
                    with SQLiteCache(sys.argv[1]) as cache:
                        wrapped.cache = cache
                        assert wrapped.generate('same prompt') == answer
                    os._exit(0)
                except BaseException:
                    os._exit(1)
            _, status = os.waitpid(pid, 0)
            assert os.waitstatus_to_exitcode(status) == 0
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "shared.sqlite")],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
