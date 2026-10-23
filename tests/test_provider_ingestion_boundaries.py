"""Fail malformed provider responses and bound ingestion reads before indexing."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cheragh.embeddings import CohereEmbedding, OpenAIEmbedding, VoyageEmbedding
from cheragh.indexing import scan_indexable_files
from cheragh.ingestion.pipeline import _looks_binary, load_documents
from cheragh.vectorstores.memory import MemoryVectorStore
from cheragh.vectorstores.chroma import ChromaVectorStore, _restore_metadata
from cheragh.vectorstores.qdrant import QdrantVectorStore
from cheragh import Document


def _openai(items, *, normalize=True):
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(data=items)))
    return OpenAIEmbedding(client=client, normalize=normalize)


@pytest.mark.parametrize("indices", [[0, 0], [0], [0, 2], [False, 1], [0.0, 1], [None, 1]])
def test_openai_response_requires_exact_input_indices(indices):
    model = _openai([SimpleNamespace(index=index, embedding=[1.0, 0.0]) for index in indices])
    store = MemoryVectorStore(model)
    with pytest.raises(ValueError, match="each input index exactly once"):
        store.add_documents([Document("first"), Document("second")])
    assert store.documents == []
    assert store.embeddings is None


def test_openai_response_reorders_by_input_and_normalizes_without_overflow():
    model = _openai([
        SimpleNamespace(index=1, embedding=[0.0, 3e30]),
        SimpleNamespace(index=0, embedding=[3e30, 0.0]),
    ])
    np.testing.assert_allclose(model.embed_documents(["first", "second"]), np.eye(2))


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("vector", [[float("nan"), 1], [float("inf"), 1], [1e100, 0], ["1", "2"], []])
def test_openai_response_rejects_invalid_coordinates(vector, normalize):
    model = _openai([SimpleNamespace(index=0, embedding=vector)], normalize=normalize)
    with pytest.raises(ValueError):
        model.embed_query("query")


@pytest.mark.parametrize("model_type", [CohereEmbedding, VoyageEmbedding])
@pytest.mark.parametrize("method", ["embed_documents", "embed_query"])
@pytest.mark.parametrize("vectors", [[], [[1, 0], [0, 1]], [[float("nan"), 0]], [[True, False]]])
def test_other_providers_reject_incomplete_or_invalid_batches(model_type, method, vectors):
    client = SimpleNamespace(embed=lambda *args, **kwargs: SimpleNamespace(embeddings=vectors))
    model = model_type(client=client)
    with pytest.raises(ValueError):
        getattr(model, method)(["first"] if method == "embed_documents" else "query")


def test_binary_detection_reads_only_sample(monkeypatch, tmp_path):
    reads = []

    class RecordingFile(BytesIO):
        def read(self, size=-1):
            reads.append(size)
            assert size == 4096
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: RecordingFile(b"text" * 2000))
    assert _looks_binary(tmp_path / "file.txt") is False
    assert reads == [4096]


@pytest.mark.parametrize("loader", [load_documents, scan_indexable_files])
def test_oversized_documents_are_rejected_before_any_content_read(loader, monkeypatch, tmp_path):
    path = tmp_path / "large.txt"
    path.write_bytes(b"a" * 1024)

    def forbidden_open(*args, **kwargs):
        raise AssertionError("oversized document must not be opened")

    monkeypatch.setattr(Path, "open", forbidden_open)
    result = loader(path, max_file_size_mb=0.0001)
    if loader is load_documents:
        assert result == []
    else:
        assert result == ({}, [str(path)])


@pytest.mark.parametrize("loader", [load_documents, scan_indexable_files])
@pytest.mark.parametrize("limit", [True, False, 0, -1, float("nan"), float("inf"), "50"])
def test_ingestion_size_limit_cannot_be_disabled_by_invalid_values(loader, limit, tmp_path):
    with pytest.raises(ValueError, match="max_file_size_mb"):
        loader(tmp_path / "not-yet-created", max_file_size_mb=limit)


@pytest.mark.parametrize("store_type", [ChromaVectorStore, QdrantVectorStore])
def test_external_store_detaches_document_before_embedding(store_type, monkeypatch):
    document = Document("original content", doc_id="original", metadata={"acl": ["alice"]})

    class MutatingEmbedder:
        def embed_documents(self, texts):
            assert texts == ["original content"]
            document.content = "replacement content"
            document.doc_id = "replacement"
            document.metadata["acl"].append("bob")
            return np.array([[1.0, 0.0]])

    captured = {}

    def upsert(**kwargs):
        captured.update(kwargs)

    if store_type is ChromaVectorStore:
        collection = SimpleNamespace(upsert=upsert)
        client = SimpleNamespace(get_or_create_collection=lambda _: collection)
    else:
        models = SimpleNamespace(PointStruct=lambda **kwargs: SimpleNamespace(**kwargs))
        monkeypatch.setattr("cheragh.vectorstores.qdrant.require_qdrant_client", lambda: (object, models))
        client = SimpleNamespace(
            get_collections=lambda: SimpleNamespace(collections=[SimpleNamespace(name="cheragh")]),
            upsert=upsert,
        )
    store_type(MutatingEmbedder(), client=client).add_documents([document])
    if store_type is ChromaVectorStore:
        assert captured["documents"] == ["original content"]
        assert captured["ids"] == ["original"]
        assert _restore_metadata(captured["metadatas"][0]) == {"acl": ["alice"]}
    else:
        assert captured["points"][0].payload == {
            "content": "original content", "doc_id": "original", "acl": ["alice"],
        }
