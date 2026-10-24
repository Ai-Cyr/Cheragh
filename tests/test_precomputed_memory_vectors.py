"""Contextual vectors remain intact when inserted into the memory store."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from cheragh import Document, EmbeddingModel, MemoryVectorStore


class QueryOnlyEmbedding(EmbeddingModel):
    dimension = 2

    def embed_documents(self, texts):
        raise AssertionError("precomputed insertion must not re-embed documents")

    def embed_query(self, text):
        return np.array([1.0, 0.0])

    def get_fingerprint(self):
        return "tests.QueryOnlyEmbedding::2"


class UnknownDimensionEmbedding(QueryOnlyEmbedding):
    dimension = None


def test_precomputed_vectors_retain_distinct_contexts_for_identical_text():
    store = MemoryVectorStore(QueryOnlyEmbedding())
    store.add_embeddings(
        [Document("same passage", doc_id="context-a"), Document("same passage", doc_id="context-b")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    assert [doc.doc_id for doc in store.documents] == ["context-a", "context-b"]
    result = store.similarity_search("direction a", top_k=2)
    assert [doc.doc_id for doc in result] == ["context-a", "context-b"]
    assert [doc.score for doc in result] == [1.0, 0.0]


def test_precomputed_insertion_copies_document_metadata_and_caller_matrix():
    source = Document("original", metadata={"acl": {"roles": ["staff"]}}, doc_id="source")
    vectors = np.array([[1.0, 0.0]])
    documents = [source]
    store = MemoryVectorStore(QueryOnlyEmbedding())
    store.add_embeddings(documents, vectors)
    source.content = "mutated"
    source.metadata["acl"]["roles"].append("public")
    documents.clear()
    vectors[:] = np.nan
    assert store.documents[0].content == "original"
    assert store.documents[0].metadata == {"acl": {"roles": ["staff"]}}
    np.testing.assert_array_equal(store.embeddings, [[1.0, 0.0]])


def test_precomputed_array_conversion_cannot_rewrite_document_snapshot():
    document = Document("before", metadata={"nested": ["before"]})

    class MutatingArray:
        def __array__(self, dtype=None, copy=None):
            document.content = "after"
            document.metadata["nested"].append("after")
            return np.array([[1.0, 0.0]], dtype=dtype)

    store = MemoryVectorStore(QueryOnlyEmbedding())
    store.add_embeddings([document], MutatingArray())
    assert store.documents[0].content == "before"
    assert store.documents[0].metadata["nested"] == ["before"]


@pytest.mark.parametrize("vectors", [
    [1.0, 0.0], [[[1.0, 0.0]]], [[1.0]], [[1.0, 0.0], [0.0, 1.0]],
    [[]], [[np.nan, 0.0]], [[np.inf, 0.0]], [[True, False]],
    [[1.0j, 0.0j]], [["1", "0"]], np.array([[1.0, 0.0]], dtype=object),
])
def test_invalid_precomputed_batches_leave_existing_state_unchanged(vectors):
    store = MemoryVectorStore(QueryOnlyEmbedding())
    store.add_embeddings([Document("existing", doc_id="original")], [[1.0, 0.0]])
    documents, embeddings = store.documents, store.embeddings
    with pytest.raises((TypeError, ValueError)):
        store.add_embeddings([Document("invalid", doc_id="new")], vectors)
    assert store.documents is documents
    assert store.embeddings is embeddings
    np.testing.assert_array_equal(store.embeddings, [[1.0, 0.0]])


def test_precomputed_batches_must_match_unknown_models_established_store_dimension():
    store = MemoryVectorStore(UnknownDimensionEmbedding())
    store.add_embeddings([Document("first")], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="store contains 2, got 3"):
        store.add_embeddings([Document("second")], [[1.0, 0.0, 0.0]])
    assert len(store.documents) == 1


@pytest.mark.parametrize("documents", [[object()], [Document("")], [Document("x", metadata=None)]])
def test_precomputed_insertion_validates_documents_before_array_conversion(documents):
    class UnexpectedArray:
        def __array__(self, dtype=None, copy=None):
            raise AssertionError("array conversion should not run")

    store = MemoryVectorStore(QueryOnlyEmbedding())
    with pytest.raises((TypeError, ValueError)):
        store.add_embeddings(documents, UnexpectedArray())
    assert store.documents == []
    assert store.embeddings is None


def test_precomputed_empty_batch_is_validated_and_does_not_mutate_store():
    store = MemoryVectorStore(QueryOnlyEmbedding())
    store.add_embeddings([], np.empty((0, 2)))
    assert store.embeddings is None
    store.add_embeddings([Document("first")], [[1.0, 0.0]])
    documents, embeddings = store.documents, store.embeddings
    store.add_embeddings([], np.empty((0, 2)))
    assert store.documents is documents
    assert store.embeddings is embeddings
    for invalid in ([], np.empty((0, 0)), np.empty((0, 3)), [[1.0, 0.0]]):
        with pytest.raises(ValueError):
            store.add_embeddings([], invalid)
    assert store.documents is documents
    assert store.embeddings is embeddings


def test_precomputed_insertion_preserves_empty_persisted_store_dimension():
    store = MemoryVectorStore(UnknownDimensionEmbedding())
    store.embeddings = np.empty((0, 2))
    with pytest.raises(ValueError, match="store contains 2, got 3"):
        store.add_embeddings([Document("new")], [[1.0, 0.0, 0.0]])
    assert store.embeddings.shape == (0, 2)


def test_precomputed_allocation_failure_keeps_old_snapshot_aligned(monkeypatch):
    store = MemoryVectorStore(QueryOnlyEmbedding())
    store.add_embeddings([Document("first")], [[1.0, 0.0]])
    documents, embeddings = store.documents, store.embeddings

    def fail_stack(*args, **kwargs):
        raise MemoryError("allocation failed")

    monkeypatch.setattr(np, "vstack", fail_stack)
    with pytest.raises(MemoryError, match="allocation failed"):
        store.add_embeddings([Document("second")], [[0.0, 1.0]])
    assert store.documents is documents
    assert store.embeddings is embeddings


def test_precomputed_concurrent_appends_preserve_document_vector_alignment():
    store = MemoryVectorStore(QueryOnlyEmbedding())
    barrier = threading.Barrier(4)

    def append(index):
        barrier.wait(timeout=2)
        store.add_embeddings([Document(str(index), doc_id=str(index))], [[float(index), 1.0]])

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append, range(4)))
    assert len(store.documents) == len(store.embeddings) == 4
    for document, vector in zip(store.documents, store.embeddings):
        np.testing.assert_array_equal(vector, [float(document.content), 1.0])


def test_precomputed_vectors_persist_and_load_without_reembedding(tmp_path):
    model = QueryOnlyEmbedding()
    store = MemoryVectorStore(model)
    store.add_embeddings([Document("contextual", doc_id="source")], [[0.75, 0.25]])
    store.save(tmp_path)
    restored = MemoryVectorStore.load(tmp_path, model)
    np.testing.assert_array_equal(restored.embeddings, [[0.75, 0.25]])
    assert restored.similarity_search("query")[0].doc_id == "source"


def test_add_documents_still_snapshots_before_embedding_provider_runs():
    document = Document("before", metadata={"acl": ["staff"]})

    class MutatingProvider(QueryOnlyEmbedding):
        def embed_documents(self, texts):
            assert texts == ["before"]
            document.content = "after"
            document.metadata["acl"].append("public")
            return np.array([[1.0, 0.0]])

    store = MemoryVectorStore(MutatingProvider())
    store.add_documents([document])
    assert store.documents[0].content == "before"
    assert store.documents[0].metadata["acl"] == ["staff"]
