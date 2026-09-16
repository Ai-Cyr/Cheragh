"""Regression tests for cache isolation and incremental index recovery."""
import sqlite3

import numpy as np
import pytest

from cheragh import Document, HashingEmbedding, index_path
from cheragh.base import BaseRetriever, EmbeddingModel
from cheragh.cache import CachedEmbeddingModel, CachedRetriever, MemoryCache, SQLiteCache, make_cache_key
from cheragh.cache.base import CacheEntry, dumps_entry, loads_entry
from cheragh.security import AccessControlledRetriever, AccessPolicy, Principal
from cheragh.vectorstores.memory import MemoryVectorStore


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ((b"a", b"b"), (b"a\x1fb",)),
        (([b"a,b", b"c"],), ([b"a", b"b,c"],)),
        (({1: "value"},), ({"1": "value"},)),
        ((1,), (b"1",)),
        (([1, 2],), ((1, 2),)),
        (({1, 2},), ([1, 2],)),
    ],
)
def test_cache_keys_preserve_component_boundaries_and_types(first, second):
    assert make_cache_key(*first) != make_cache_key(*second)


def test_cache_keys_are_independent_of_mapping_and_set_order():
    assert make_cache_key({"a": 1, "b": 2}) == make_cache_key({"b": 2, "a": 1})
    assert make_cache_key({"a", "b"}) == make_cache_key(set(["b", "a"]))


@pytest.mark.parametrize("marker", ["bytes", "Document", "ndarray", "mapping"])
def test_json_cache_preserves_user_mappings_with_reserved_type_markers(marker):
    value = {"__cheragh_type__": marker, "content": "user data", "data": "YWJj"}
    original = CacheEntry(key="key", namespace="test", value=value, metadata={"nested": value})
    restored = loads_entry(dumps_entry(original))
    assert restored.value == value
    assert restored.metadata == original.metadata


def test_sqlite_corrupt_cleanup_preserves_concurrent_replacement(tmp_path, monkeypatch):
    import cheragh.cache.sqlite as sqlite_module

    with SQLiteCache(tmp_path / "cache.sqlite") as reader, SQLiteCache(tmp_path / "cache.sqlite") as writer:
        reader.set("key", "old")
        reader._conn.execute("UPDATE cache_entries SET payload=?", (sqlite3.Binary(b"invalid JSON"),))
        reader._conn.commit()
        decode = sqlite_module.loads_entry

        def replace_before_decode(raw, **kwargs):
            writer.set("key", "fresh")
            return decode(raw, **kwargs)

        monkeypatch.setattr(sqlite_module, "loads_entry", replace_before_decode)
        assert reader.get("key") is None
        monkeypatch.setattr(sqlite_module, "loads_entry", decode)
        assert reader.get("key") == "fresh"
        assert reader.stats().errors == 1


@pytest.mark.parametrize("empty_source", [False, True])
def test_index_without_source_manifest_does_not_reuse_foreign_documents(tmp_path, empty_source):
    source = tmp_path / "source"
    source.mkdir()
    if not empty_source:
        (source / "current.txt").write_text("current corpus", encoding="utf-8")
    output = tmp_path / "index"
    store = MemoryVectorStore(HashingEmbedding(16))
    store.add_documents([Document("foreign corpus", metadata={"source": str(tmp_path / "foreign.txt")})])
    store.save(output)

    result = index_path(source, output, embedding_model=HashingEmbedding(16))

    restored = MemoryVectorStore.load(output)
    assert result["indexed_documents"] == (0 if empty_source else 1)
    assert all(document.content != "foreign corpus" for document in restored.documents)


class _MutableEmbedding(HashingEmbedding):
    def __init__(self):
        super().__init__(2)
        self.query_vector = np.asarray([1.0, 2.0])
        self.document_vectors = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        self.query_calls = 0

    def embed_query(self, text):
        self.query_calls += 1
        return self.query_vector

    def embed_documents(self, texts):
        return self.document_vectors


def test_embedding_cache_isolates_provider_and_caller_mutations():
    model = _MutableEmbedding()
    cached = CachedEmbeddingModel(model, MemoryCache())
    query = cached.embed_query("query")
    query[0] = 99
    model.query_vector[1] = 99
    np.testing.assert_array_equal(cached.embed_query("query"), [1.0, 2.0])
    assert model.query_calls == 1

    vectors = cached.embed_documents(["first", "second"])
    vectors[0, 0] = 99
    model.document_vectors[1, 0] = 99
    np.testing.assert_array_equal(cached.embed_documents(["first", "second"]), [[1, 2], [3, 4]])


@pytest.mark.parametrize(
    "invalid",
    [np.zeros((1, 2)), np.zeros((3, 2)), np.zeros((2, 0)), np.asarray([[1, 2], [np.nan, 2]])],
)
def test_invalid_embedding_batch_does_not_partially_populate_cache(invalid):
    model = _MutableEmbedding()
    model.document_vectors = invalid
    cache = MemoryCache()
    cached = CachedEmbeddingModel(model, cache)
    with pytest.raises(ValueError, match="embedding"):
        cached.embed_documents(["first", "second"])
    assert cache.entry_count() == 0


@pytest.mark.parametrize("invalid", [np.asarray([np.inf, 0]), np.asarray([1j, 0]), np.asarray([True, False])])
def test_invalid_query_embedding_is_not_cached(invalid):
    model = _MutableEmbedding()
    model.query_vector = invalid
    cache = MemoryCache()
    with pytest.raises(ValueError, match="embedding"):
        CachedEmbeddingModel(model, cache).embed_query("query")
    assert cache.entry_count() == 0


class _EmptyDimensionEmbedding(EmbeddingModel):
    def embed_documents(self, texts):
        return np.zeros((len(texts), 0))

    def embed_query(self, text):
        return np.zeros(0)


def test_memory_store_rejects_zero_dimensional_vectors_before_mutation():
    store = MemoryVectorStore(_EmptyDimensionEmbedding())
    with pytest.raises(ValueError, match="dimension"):
        store.add_documents([Document("knowledge")])
    assert store.documents == []
    assert store.embeddings is None


def test_memory_store_allocation_failure_preserves_document_vector_alignment(monkeypatch):
    store = MemoryVectorStore(HashingEmbedding(16))
    store.add_documents([Document("first")])
    original_vectors = store.embeddings.copy()
    original_vstack = np.vstack

    def fail_store_concatenation(arrays):
        # The hashing provider also stacks rows. Fail only when the store joins
        # the previous and new matrices, after embedding has succeeded.
        if len(arrays) == 2 and all(array.ndim == 2 for array in arrays):
            raise MemoryError("allocation failed")
        return original_vstack(arrays)

    monkeypatch.setattr(np, "vstack", fail_store_concatenation)
    with pytest.raises(MemoryError, match="allocation failed"):
        store.add_documents([Document("second")])
    assert [document.content for document in store.documents] == ["first"]
    np.testing.assert_array_equal(store.embeddings, original_vectors)


def _access_store():
    store = MemoryVectorStore(HashingEmbedding(16))
    store.add_documents([
        Document("first private evidence", {"tenant_id": "a", "allowed_users": ["A"], "category": "one"}, "a"),
        Document("second private evidence", {"tenant_id": "a", "allowed_users": ["B"], "category": "two"}, "b"),
    ])
    return store


def test_cached_access_retrievers_do_not_share_authorized_results_between_users():
    source = _access_store().as_retriever()
    cache = MemoryCache()
    first = CachedRetriever(AccessControlledRetriever(source, Principal("A", tenant_ids={"a"})), cache)
    second = CachedRetriever(AccessControlledRetriever(source, Principal("B", tenant_ids={"a"})), cache)
    assert [doc.doc_id for doc in first.retrieve("evidence")] == ["a"]
    assert [doc.doc_id for doc in second.retrieve("evidence")] == ["b"]


def test_cached_access_retriever_rechecks_mutated_principal_and_policy():
    principal = Principal("A", tenant_ids={"a"})
    policy = AccessPolicy()
    cached = CachedRetriever(AccessControlledRetriever(_access_store().as_retriever(), principal, policy), MemoryCache())
    assert [doc.doc_id for doc in cached.retrieve("evidence")] == ["a"]
    principal.user_id = "B"
    assert [doc.doc_id for doc in cached.retrieve("evidence")] == ["b"]
    policy.metadata_equals["category"] = "one"
    assert cached.retrieve("evidence") == []
    principal.tenant_ids.clear()
    policy.metadata_equals.clear()
    assert cached.retrieve("evidence") == []


def test_cached_access_retrievers_preserve_distinct_policy_decisions():
    source = _access_store().as_retriever()
    principal = Principal("A", tenant_ids={"a"})
    cache = MemoryCache()
    first = CachedRetriever(AccessControlledRetriever(source, principal, AccessPolicy()), cache)
    second = CachedRetriever(
        AccessControlledRetriever(source, principal, AccessPolicy(metadata_equals={"category": "two"})), cache,
    )
    assert [doc.doc_id for doc in first.retrieve("evidence")] == ["a"]
    assert second.retrieve("evidence") == []


def test_cached_scope_wrappers_preserve_selected_tenant_and_collection():
    from cheragh.tenancy.engine import _ScopedRetriever

    store = MemoryVectorStore(HashingEmbedding(16))
    store.add_documents([Document("registry-scoped knowledge")])
    source = store.as_retriever()
    cache = MemoryCache()
    first = CachedRetriever(_ScopedRetriever(source, "a", "one"), cache)
    second = CachedRetriever(_ScopedRetriever(source, "b", "two"), cache)
    assert first.retrieve("knowledge")[0].metadata == {"tenant_id": "a", "collection_id": "one"}
    assert second.retrieve("knowledge")[0].metadata == {"tenant_id": "b", "collection_id": "two"}


class _VersionedRetrieverWrapper(BaseRetriever):
    def __init__(self, retriever):
        self.retriever = retriever

    def get_fingerprint(self):
        return "index-v1"

    def retrieve(self, query, top_k=5):
        return self.retriever.retrieve(query, top_k=top_k)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_corpus_fingerprint_cannot_override_cached_authorization(explicit, nested):
    source = _access_store().as_retriever()
    cache = MemoryCache()

    class VersionedAccessRetriever(AccessControlledRetriever):
        def get_fingerprint(self):
            return "index-v1"

    first_principal = Principal("A", tenant_ids={"a"})
    first = VersionedAccessRetriever(source, first_principal)
    second = VersionedAccessRetriever(source, Principal("B", tenant_ids={"a"}))
    if nested:
        first = _VersionedRetrieverWrapper(first)
        second = _VersionedRetrieverWrapper(second)
    options = {"fingerprint": "index-v1"} if explicit else {}
    cached_first = CachedRetriever(first, cache, **options)
    cached_second = CachedRetriever(second, cache, **options)
    assert [doc.doc_id for doc in cached_first.retrieve("evidence")] == ["a"]
    assert [doc.doc_id for doc in cached_second.retrieve("evidence")] == ["b"]
    first_principal.tenant_ids.clear()
    assert cached_first.retrieve("evidence") == []
    if explicit:
        assert cached_first.fingerprint == "index-v1"


def test_explicit_corpus_fingerprint_preserves_nested_collection_scope():
    from cheragh.tenancy.engine import _ScopedRetriever

    store = MemoryVectorStore(HashingEmbedding(16))
    store.add_documents([Document("registry-scoped knowledge")])
    source = store.as_retriever()
    cache = MemoryCache()
    first = CachedRetriever(
        _VersionedRetrieverWrapper(_ScopedRetriever(source, "a", "one")), cache, fingerprint="index-v1",
    )
    second = CachedRetriever(
        _VersionedRetrieverWrapper(_ScopedRetriever(source, "b", "two")), cache, fingerprint="index-v1",
    )
    assert first.retrieve("knowledge")[0].metadata == {"tenant_id": "a", "collection_id": "one"}
    assert second.retrieve("knowledge")[0].metadata == {"tenant_id": "b", "collection_id": "two"}


def test_custom_policy_with_external_decisions_bypasses_authorized_result_cache():
    allowed = True

    class ExternalPolicy(AccessPolicy):
        def filter_documents(self, documents, principal=None):
            return super().filter_documents(documents, principal) if allowed else []

    cache = MemoryCache()
    guarded = AccessControlledRetriever(_access_store().as_retriever(), Principal("A", tenant_ids={"a"}), ExternalPolicy())
    cached = CachedRetriever(_VersionedRetrieverWrapper(guarded), cache, fingerprint="index-v1")
    assert [doc.doc_id for doc in cached.retrieve("evidence")] == ["a"]
    allowed = False
    assert cached.retrieve("evidence") == []
    assert cache.entry_count() == 0


def test_authorization_snapshot_preserves_numeric_and_string_policy_types():
    store = MemoryVectorStore(HashingEmbedding(16))
    store.add_documents([Document("numeric policy evidence", {"tenant_id": "a", "level": 1.0}, "numeric")])
    principal = Principal("A", tenant_ids={"a"})
    cache = MemoryCache()
    first = CachedRetriever(
        AccessControlledRetriever(store.as_retriever(), principal, AccessPolicy(metadata_equals={"level": 1.0})),
        cache, fingerprint="index-v1",
    )
    second = CachedRetriever(
        AccessControlledRetriever(store.as_retriever(), principal, AccessPolicy(metadata_equals={"level": "1.0"})),
        cache, fingerprint="index-v1",
    )
    assert [doc.doc_id for doc in first.retrieve("evidence")] == ["numeric"]
    assert second.retrieve("evidence") == []
