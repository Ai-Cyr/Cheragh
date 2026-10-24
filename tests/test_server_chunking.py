"""Chunking selection is authenticated, validated and applied during live indexing."""
from __future__ import annotations

import importlib.util
import json
import re
import time
from unittest.mock import patch

import pytest
import numpy as np

from cheragh import Document, EmbeddingModel, HashingEmbedding, MemoryVectorStore, RAGEngine
from cheragh.base import ExtractiveLLMClient, LLMClient, StaticLLMClient
from cheragh.cache import CachedEmbeddingModel, CachedLLMClient, MemoryCache
from cheragh.ingestion.chunkers import CHUNKING_STRATEGIES
from cheragh.server.app import create_app


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None, reason="FastAPI optional dependency is not installed",
)
AUTH = {"X-API-Key": "chunking-test-key"}
ADVANCED = {"semantic", "late", "contextual"}


class ConfiguredEmbedding(EmbeddingModel):
    dimension = 2

    def __init__(self):
        self.batches = []

    def embed_documents(self, texts):
        self.batches.append(list(texts))
        return np.asarray([self.embed_query(text) for text in texts])

    def embed_query(self, text):
        return np.asarray([1.0, 0.0] if "coral" in text.lower() else [0.0, 1.0])


class ContextualTokenEmbedding(ConfiguredEmbedding):
    def __init__(self):
        super().__init__()
        self.token_inputs = []

    def encode_tokens(self, text):
        self.token_inputs.append(text)
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {"offsets": offsets, "embeddings": np.tile([1.0, 2.0], (len(offsets), 1))}


class ConfiguredLLM(LLMClient):
    model = "test-contextual-model"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return "Generated retrieval context."


def configured_app(root, model, llm=None, **options):
    store = MemoryVectorStore(model)
    app = create_app(
        RAGEngine(store.as_retriever(), llm_client=llm),
        api_key=AUTH["X-API-Key"], require_auth=True,
        enable_indexing=True, allowed_index_root=root, **options,
    )
    return app, store


@pytest.fixture
def client_and_root(tmp_path):
    from fastapi.testclient import TestClient

    store = MemoryVectorStore(HashingEmbedding(dimension=64))
    store.add_documents([Document("The old deployment identifier is amber-lighthouse.", doc_id="old")])
    store.save(tmp_path / "index")
    (tmp_path / "corpus.txt").write_text(
        "The deployment identifier is coral-garden.\n\n"
        "Releases are checked before deployment.\n\n"
        "The staging environment has separate data.",
        encoding="utf-8",
    )
    app = create_app(
        index_path=str(tmp_path / "index"), api_key=AUTH["X-API-Key"],
        require_auth=True, enable_indexing=True, allowed_index_root=tmp_path,
    )
    with TestClient(app) as client:
        yield client, tmp_path


def test_ui_advertises_shared_strategy_catalogue_behind_authentication(client_and_root):
    client, _ = client_and_root
    assert client.get("/ui/api/settings").status_code == 401
    options = client.get("/ui/api/settings", headers=AUTH).json()["chunking_strategies"]
    assert {option["id"] for option in options} == set(CHUNKING_STRATEGIES)
    assert next(option for option in options if option["id"] == "token")["unit"] == "tokens"
    assert all(option["label"] and option["description"] for option in options)
    assert all(option["available"] is False and option["reason"] for option in options if option["id"] in ADVANCED)


@pytest.mark.parametrize("strategy", ["unknown", "Paragraph", True, 12, None, ["paragraph"]])
def test_invalid_strategy_is_rejected_before_index_writes(client_and_root, strategy):
    client, _ = client_and_root
    with patch("cheragh.indexing.index_path") as build:
        response = client.post(
            "/index", headers=AUTH,
            json={"path": "corpus.txt", "output": "index", "chunking_strategy": strategy},
        )
    assert response.status_code == 422
    build.assert_not_called()


@pytest.mark.parametrize("strategy", [name for name in CHUNKING_STRATEGIES if name not in ADVANCED])
def test_index_request_passes_selected_strategy_to_indexer(client_and_root, strategy):
    client, _ = client_and_root
    with patch("cheragh.indexing.index_path", return_value={"indexed_documents": 1}) as build:
        response = client.post(
            "/index", headers=AUTH,
            json={"path": "corpus.txt", "output": "other", "chunking_strategy": strategy},
        )
    assert response.status_code == 200
    assert build.call_args.kwargs["chunking_strategy"] == strategy


def test_omitted_strategy_preserves_recursive_default(client_and_root):
    client, _ = client_and_root
    with patch("cheragh.indexing.index_path", return_value={"indexed_documents": 1}) as build:
        response = client.post("/index", headers=AUTH, json={"path": "corpus.txt"})
    assert response.status_code == 200
    assert build.call_args.kwargs["chunking_strategy"] == "recursive"


def test_selected_paragraph_strategy_is_used_by_activated_index(client_and_root):
    client, root = client_and_root
    response = client.post("/index", headers=AUTH, json={
        "path": "corpus.txt", "output": "index", "activate": True,
        "chunking_strategy": "paragraph", "chunk_size": 50, "chunk_overlap": 0,
    })
    assert response.status_code == 200, response.text
    assert response.json()["activated"] is True
    documents = MemoryVectorStore.load(root / "index").documents
    assert len(documents) == 3
    assert all(document.metadata["chunker"] == "paragraph" for document in documents)
    assert all(len(document.content) <= 50 for document in documents)
    manifest = json.loads((root / "index" / "index_manifest.json").read_text())
    assert manifest["metadata"]["indexing_options"]["chunking_strategy"] == "paragraph"
    answer = client.post("/ask", headers=AUTH, json={"query": "deployment identifier"}).json()
    assert "coral-garden" in answer["answer"]
    assert "amber-lighthouse" not in answer["answer"]


@pytest.mark.parametrize("strategy", sorted(ADVANCED))
def test_unconfigured_advanced_strategy_cannot_fall_back_to_hashing(client_and_root, strategy):
    client, _ = client_and_root
    with patch("cheragh.indexing.index_path") as build:
        response = client.post("/index", headers=AUTH, json={
            "path": "corpus.txt", "output": "other", "chunking_strategy": strategy,
        })
    assert response.status_code == 400
    build.assert_not_called()


@pytest.mark.parametrize("strategy,options", [
    ("recursive", {"model": "remote-model"}),
    ("semantic", {"buffer_size": -1}),
    ("semantic", {"breakpoint_percentile": 100}),
    ("semantic", {"api_key": "not-accepted"}),
    ("late", {"checkpoint": "remote-model"}),
    ("contextual", {"max_document_chars": 0}),
])
def test_chunking_options_are_validated_before_provider_work(client_and_root, strategy, options):
    client, _ = client_and_root
    with patch("cheragh.indexing.index_path") as build:
        response = client.post("/index", headers=AUTH, json={
            "path": "corpus.txt", "chunking_strategy": strategy, "chunking_options": options,
        })
    assert response.status_code == 422
    build.assert_not_called()


@pytest.mark.parametrize("strategy,model,llm", [
    ("semantic", CachedEmbeddingModel(HashingEmbedding(), MemoryCache()), None),
    ("contextual", ConfiguredEmbedding(), CachedLLMClient(ExtractiveLLMClient(), MemoryCache())),
    ("contextual", ConfiguredEmbedding(), CachedLLMClient(StaticLLMClient("static"), MemoryCache())),
])
def test_cached_fallbacks_do_not_advertise_advanced_model_capabilities(tmp_path, strategy, model, llm):
    from fastapi.testclient import TestClient

    app, _ = configured_app(tmp_path, model, llm)
    with TestClient(app) as client, patch("cheragh.indexing.index_path") as build:
        strategies = client.get("/ui/api/settings", headers=AUTH).json()["chunking_strategies"]
        option = next(item for item in strategies if item["id"] == strategy)
        assert option["available"] is False
        assert option["reason"]
        response = client.post("/index", headers=AUTH, json={
            "path": "corpus.txt", "output": "snapshot", "chunking_strategy": strategy,
        })
    assert response.status_code == 400
    build.assert_not_called()


def test_semantic_index_uses_configured_model_and_requested_options(tmp_path):
    from fastapi.testclient import TestClient

    text = "Coral gardens shelter fish. Gardens bloom in summer."
    (tmp_path / "corpus.txt").write_text(text)
    model = ConfiguredEmbedding()
    app, _ = configured_app(tmp_path, model)
    with TestClient(app) as client:
        strategies = client.get("/ui/api/settings", headers=AUTH).json()["chunking_strategies"]
        assert next(item for item in strategies if item["id"] == "semantic")["available"] is True
        response = client.post("/index", headers=AUTH, json={
            "path": "corpus.txt", "output": "semantic-index", "chunking_strategy": "semantic",
            "chunk_size": 200, "chunk_overlap": 0,
            "chunking_options": {"min_chunk_size": 0, "buffer_size": 0, "breakpoint_threshold": 0.75},
        })
    assert response.status_code == 200, response.text
    assert response.json()["activated"] is False
    indexed = MemoryVectorStore.load(tmp_path / "semantic-index", model)
    assert len(indexed.documents) == 2
    assert any(len(batch) == 2 for batch in model.batches)
    assert all(document.metadata["chunk_method"] == "semantic" for document in indexed.documents)


@pytest.mark.parametrize("cache_depth", [0, 2])
def test_late_index_preserves_contextual_vectors_without_reembedding_passages(tmp_path, cache_depth):
    from fastapi.testclient import TestClient

    text = "Coral gardens shelter fish. Gardens bloom in summer."
    (tmp_path / "corpus.txt").write_text(text)
    model = ContextualTokenEmbedding()
    query_model = model
    for _ in range(cache_depth):
        query_model = CachedEmbeddingModel(query_model, MemoryCache())
    app, store = configured_app(tmp_path, query_model)
    with TestClient(app) as client:
        strategies = client.get("/ui/api/settings", headers=AUTH).json()["chunking_strategies"]
        assert next(item for item in strategies if item["id"] == "late")["available"] is True
        response = client.post("/index", headers=AUTH, json={
            "path": "corpus.txt", "output": "late-index", "chunking_strategy": "late",
            "chunk_size": 30, "chunk_overlap": 0,
        })
    assert response.status_code == 200, response.text
    assert store.embedding_model is query_model
    indexed = MemoryVectorStore.load(tmp_path / "late-index", query_model)
    assert len(indexed.documents) >= 2
    assert model.token_inputs == [text]
    assert model.batches == [], "Late passage vectors were independently re-embedded"
    assert np.allclose(indexed.embeddings, np.tile(np.asarray([1.0, 2.0]) / np.sqrt(5), (len(indexed.documents), 1)))


def test_contextual_index_uses_configured_llm_but_preserves_source_evidence(tmp_path):
    from fastapi.testclient import TestClient

    text = "Coral gardens shelter fish. Gardens bloom in summer."
    (tmp_path / "corpus.txt").write_text(text)
    model, llm = ConfiguredEmbedding(), ConfiguredLLM()
    app, _ = configured_app(tmp_path, model, llm)
    with TestClient(app) as client:
        response = client.post("/index", headers=AUTH, json={
            "path": "corpus.txt", "output": "contextual-index", "chunking_strategy": "contextual",
            "chunk_size": 30, "chunk_overlap": 0,
        })
    assert response.status_code == 200, response.text
    indexed = MemoryVectorStore.load(tmp_path / "contextual-index", model)
    assert len(llm.prompts) == len(indexed.documents)
    assert all(text in prompt for prompt in llm.prompts)
    assert all("Generated retrieval context." in item for batch in model.batches for item in batch)
    assert all("Generated retrieval context." not in document.content for document in indexed.documents)


def test_timeout_while_waiting_for_source_embedding_lock_cannot_start_late_index_write(tmp_path):
    from fastapi.testclient import TestClient

    app, store = configured_app(tmp_path, ConfiguredEmbedding(), index_timeout_seconds=0.03)
    body = {"path": "corpus.txt", "output": "snapshot", "chunking_strategy": "semantic"}
    with TestClient(app) as client, patch("cheragh.indexing.index_path", return_value={"indexed_documents": 1}) as build:
        with store._embedding_lock:
            assert client.post("/index", headers=AUTH, json=body).status_code == 504
            build.assert_not_called()
        deadline = time.monotonic() + 2
        while True:
            result = client.post("/index", headers=AUTH, json=body)
            if result.status_code != 503:
                break
            assert time.monotonic() < deadline, "Timed-out index did not release its write lease"
            time.sleep(0.005)
        assert result.status_code == 200, result.text
        assert build.call_count == 1, "The abandoned worker wrote after obtaining the embedding lock"
