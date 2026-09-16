"""Production contracts against local FAISS, Chroma and Qdrant SDKs."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cheragh import Document, HashingEmbedding
from cheragh.vectorstores import ChromaVectorStore, FaissVectorStore, QdrantVectorStore


@pytest.fixture
def qdrant(tmp_path):
    pytest.importorskip("qdrant_client")
    store = QdrantVectorStore(HashingEmbedding(16), path=tmp_path / "qdrant")
    try:
        yield store
    finally:
        store.close()


def test_qdrant_current_query_api_and_canonical_filters(qdrant):
    qdrant.add_documents([
        Document("cat", doc_id="a", metadata={"tenant": "a", "enabled": True}),
        Document("dog", doc_id="b", metadata={"tenant": "b", "enabled": False}),
    ])
    assert qdrant.similarity_search("cat", top_k=1)[0].doc_id == "a"
    assert [doc.doc_id for doc in qdrant.similarity_search("cat", filters={"tenant": "b"})] == ["b"]
    assert [doc.doc_id for doc in qdrant.similarity_search("cat", filters={"enabled": {"$in": [True]}})] == ["a"]


def test_qdrant_close_releases_local_path_lock(tmp_path):
    pytest.importorskip("qdrant_client")
    path = tmp_path / "local"
    with QdrantVectorStore(HashingEmbedding(8), path=path) as first:
        first.add_documents([Document("cat", doc_id="a")])
    with QdrantVectorStore(HashingEmbedding(8), path=path) as reopened:
        assert reopened.similarity_search("cat")[0].doc_id == "a"
    reopened.close()


def test_qdrant_legacy_search_client_remains_supported():
    class LegacyClient:
        def search(self, **kwargs):
            assert kwargs["query_vector"] and kwargs["query_filter"] is None
            return [SimpleNamespace(payload={"content": "cat", "doc_id": "a"}, id=123, score=1.0)]

    store = QdrantVectorStore(HashingEmbedding(8), client=LegacyClient())
    assert store.similarity_search("cat")[0].doc_id == "a"


def test_chroma_empty_and_nonempty_metadata_roundtrip(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    from chromadb.config import Settings

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False))
    store = ChromaVectorStore(HashingEmbedding(16), client=client)
    store.add_documents([
        Document("cat", doc_id="a"),
        Document("dog", doc_id="b", metadata={"tenant": "b", "tags": ["animal", "pet"]}),
    ])
    docs = {doc.doc_id: doc for doc in store.similarity_search("cat", top_k=2)}
    assert docs["a"].metadata == {}
    assert docs["b"].metadata["tags"] == ["animal", "pet"]
    assert [doc.doc_id for doc in store.similarity_search("cat", filters={"tenant": "b"})] == ["b"]


def test_chroma_upsert_replaces_metadata_without_inheriting_acl(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    from chromadb.config import Settings
    from cheragh.vectorstores.chroma import _METADATA_ENVELOPE_KEY, _METADATA_ENVELOPE_PREFIX

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False))
    store = ChromaVectorStore(HashingEmbedding(16), client=client)
    store.add_documents([Document("cat", doc_id="a", metadata={"tenant": "old", "acl": ["old-user"]})])
    store.add_documents([Document("new content", doc_id="a", metadata={"tenant": "new"})])
    result = store.similarity_search("cat")[0]
    assert result.content == "new content" and result.metadata == {"tenant": "new"}
    assert store.similarity_search("cat", filters={"tenant": "old"}) == []
    assert store.similarity_search("cat", filters={"acl": {"$contains": "old-user"}}) == []
    store.add_documents([Document("cleared", doc_id="a", metadata={})])
    assert store.similarity_search("cat")[0].metadata == {}
    assert store.similarity_search("cat", filters={"tenant": "new"}) == []
    assert store.similarity_search("cat", filters={"tenant": None})[0].doc_id == "a"
    assert store.similarity_search("cat", filters={"tenant": [None, "new"]})[0].doc_id == "a"
    # User keys and marker-looking values still roundtrip inside the snapshot.
    literal = _METADATA_ENVELOPE_PREFIX + '{"tenant": "forged"}'
    store.add_documents([Document("literal", doc_id="a", metadata={_METADATA_ENVELOPE_KEY: literal})])
    assert store.similarity_search("cat")[0].metadata == {_METADATA_ENVELOPE_KEY: literal}
    assert store.similarity_search("cat", filters={_METADATA_ENVELOPE_KEY: literal})[0].doc_id == "a"


def test_chroma_legacy_metadata_does_not_look_like_new_snapshot():
    from cheragh.vectorstores.chroma import (
        _JSON_METADATA_PREFIX, _METADATA_ENVELOPE_KEY, _METADATA_ENVELOPE_PREFIX,
        _STRING_METADATA_PREFIX, _restore_metadata,
    )

    literal = _METADATA_ENVELOPE_PREFIX + '{"tenant": "forged"}'
    # Legacy strings were escaped; dictionaries used the other prefix.
    legacy_string = {_METADATA_ENVELOPE_KEY: _STRING_METADATA_PREFIX + literal, "tenant": "real"}
    assert _restore_metadata(legacy_string) == {_METADATA_ENVELOPE_KEY: literal, "tenant": "real"}
    legacy_dict = {_METADATA_ENVELOPE_KEY: _JSON_METADATA_PREFIX + '{"tenant": "forged"}', "tenant": "real"}
    assert _restore_metadata(legacy_dict) == {_METADATA_ENVELOPE_KEY: {"tenant": "forged"}, "tenant": "real"}


@pytest.fixture
def faiss_store():
    pytest.importorskip("faiss")
    store = FaissVectorStore(HashingEmbedding(8))
    store.add_documents([Document("cat", doc_id="a", metadata={"nested": {"acl": ["a"]}})])
    return store


@pytest.mark.parametrize("bad", [np.ones((1, 8)), np.ones((2, 0)), np.full((2, 8), np.nan),
                                  np.full((2, 8), 1e100), np.ones((2, 8), dtype=complex)])
def test_faiss_rejects_invalid_batches_before_mutating_index(faiss_store, bad, monkeypatch):
    monkeypatch.setattr(faiss_store.embedding_model, "embed_documents", lambda _: bad)
    with pytest.raises(ValueError):
        faiss_store.add_documents([Document("dog"), Document("bird")])
    assert faiss_store.index.ntotal == len(faiss_store.documents) == 1
    assert faiss_store.similarity_search("cat")[0].doc_id == "a"


def test_faiss_results_detach_nested_metadata(faiss_store):
    result = faiss_store.similarity_search("cat")[0]
    result.metadata["nested"]["acl"].append("other-user")
    assert faiss_store.similarity_search("cat")[0].metadata["nested"]["acl"] == ["a"]


def test_faiss_failed_serialization_keeps_previous_snapshot(faiss_store, tmp_path):
    faiss_store.save(tmp_path)
    faiss_store.add_documents([Document("dog", doc_id="b", metadata={"unsupported": object()})])
    with pytest.raises(TypeError):
        faiss_store.save(tmp_path)
    reopened = FaissVectorStore.load(tmp_path, HashingEmbedding(8))
    assert [doc.doc_id for doc in reopened.documents] == ["a"]


def test_faiss_failed_write_keeps_previous_snapshot(faiss_store, tmp_path, monkeypatch):
    import faiss

    faiss_store.save(tmp_path)
    faiss_store.add_documents([Document("dog", doc_id="b")])

    def fail_write(index, path):
        Path(path).write_bytes(b"partial index")
        raise OSError("disk full")

    monkeypatch.setattr(faiss, "write_index", fail_write)
    with pytest.raises(OSError, match="disk full"):
        faiss_store.save(tmp_path)
    reopened = FaissVectorStore.load(tmp_path, HashingEmbedding(8))
    assert [doc.doc_id for doc in reopened.documents] == ["a"]


def test_faiss_failed_manifest_publication_keeps_previous_generation(faiss_store, tmp_path, monkeypatch):
    from cheragh.vectorstores import faiss as adapter

    faiss_store.save(tmp_path)
    old_manifest = (tmp_path / "manifest.json").read_bytes()
    faiss_store.add_documents([Document("dog", doc_id="b")])
    original = adapter.os.replace

    def fail_manifest(source, destination):
        if Path(destination).name == "manifest.json":
            raise OSError("manifest publication failed")
        return original(source, destination)

    monkeypatch.setattr(adapter.os, "replace", fail_manifest)
    with pytest.raises(OSError, match="publication failed"):
        faiss_store.save(tmp_path)
    assert (tmp_path / "manifest.json").read_bytes() == old_manifest
    assert [doc.doc_id for doc in FaissVectorStore.load(tmp_path, HashingEmbedding(8)).documents] == ["a"]


def test_faiss_large_finite_vectors_keep_cosine_direction(faiss_store, monkeypatch):
    vector = np.full(8, 1e20, dtype=np.float32)
    monkeypatch.setattr(faiss_store.embedding_model, "embed_documents", lambda _: vector[None, :])
    monkeypatch.setattr(faiss_store.embedding_model, "embed_query", lambda _: vector)
    faiss_store.add_documents([Document("large", doc_id="large")])
    result = faiss_store.similarity_search("query", top_k=1)[0]
    assert result.doc_id == "large"
    assert result.score == pytest.approx(1.0)


def test_faiss_corrupt_index_is_rejected_before_native_deserialization(faiss_store, tmp_path, monkeypatch):
    import faiss

    faiss_store.save(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    index_path = tmp_path / manifest["index"]["filename"]
    data = bytearray(index_path.read_bytes())
    data[-1] ^= 1
    index_path.write_bytes(data)
    monkeypatch.setattr(faiss, "read_index", lambda _: pytest.fail("Corrupt file reached native FAISS"))
    with pytest.raises(ValueError, match="checksum"):
        FaissVectorStore.load(tmp_path, HashingEmbedding(8))


@pytest.mark.parametrize("filename", ["../outside.faiss", "/tmp/outside.faiss", "index.faiss"])
def test_faiss_rejects_manifest_paths_outside_content_addressed_names(faiss_store, tmp_path, filename):
    faiss_store.save(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["index"]["filename"] = filename
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="filename"):
        FaissVectorStore.load(tmp_path, HashingEmbedding(8))


def test_faiss_legacy_snapshot_loads_and_upgrades(faiss_store, tmp_path):
    import faiss

    faiss.write_index(faiss_store.index, str(tmp_path / "index.faiss"))
    (tmp_path / "documents.jsonl").write_text(json.dumps({"content": "cat", "doc_id": "a", "metadata": {}}) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "count": 1, "dimension": 8, "normalize": True,
        "embedding_model": HashingEmbedding(8).get_fingerprint(),
    }))
    reopened = FaissVectorStore.load(tmp_path, HashingEmbedding(8))
    reopened.save(tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text())["schema_version"] == 2
    assert FaissVectorStore.load(tmp_path, HashingEmbedding(8)).similarity_search("cat")[0].doc_id == "a"
