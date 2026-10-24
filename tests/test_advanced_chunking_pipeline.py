"""Advanced ingestion must preserve the vectors that make each technique work."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pytest

from cheragh import Document, MemoryVectorStore, RAGEngine
from cheragh.base import EmbeddingModel, LLMClient
from cheragh.indexing import index_from_config, index_path, load_manifest
from cheragh.ingestion import ingest_path


SOURCE = (
    "Harbor ships dock beside the blue lighthouse every morning. "
    "Harbor crews unload the blue boats at the quay. "
    "Finance markets trade bonds during the afternoon. "
    "Finance traders study interest rates before closing."
)


class DocumentContextEmbedding(EmbeddingModel):
    """Token vectors depend on the full source, never on an isolated chunk."""

    dimension = 3

    def __init__(self, revision: str = "one") -> None:
        self.revision = revision
        self.encoded_texts: list[str] = []
        self.invalid_vectors = False
        self.query_calls = 0

    def encode_tokens(self, text: str) -> dict:
        self.encoded_texts.append(text)
        offsets = [match.span() for match in re.finditer(r"\S+", text)]
        rows = np.tile([1.0, float(len(text)), 3.0], (len(offsets), 1))
        if self.invalid_vectors:
            rows[0, 0] = np.nan
        return {"embeddings": rows, "offsets": offsets}

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise AssertionError("Late ingestion must preserve pooled vectors without re-embedding chunks")

    def embed_query(self, text: str) -> np.ndarray:
        self.query_calls += 1
        return np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)

    def get_fingerprint(self) -> str:
        return f"DocumentContextEmbedding::{self.revision}"


class ContextLLM(LLMClient):
    def __init__(self, model: str = "context-one") -> None:
        self.model = model
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return f"Project ORCHID context from {self.model}."


class ContextEmbedding(EmbeddingModel):
    dimension = 2

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.batches.append(list(texts))
        assert all(text.startswith("Project ORCHID context") for text in texts), (
            "Contextual ingestion must not replace enriched vectors with source-only embeddings"
        )
        return np.array([[7.0, float(len(text))] for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return np.array([0.0, 1.0])


class TopicEmbedding(EmbeddingModel):
    dimension = 2

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.batches.append(list(texts))
        return np.array([[float("Harbor" in text), float("Finance" in text)] for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return np.array([0.0, 1.0])


def write_source(tmp_path: Path, content: str = SOURCE) -> Path:
    source = tmp_path / "source.txt"
    source.write_text(content, encoding="utf-8")
    return source


def write_config(tmp_path: Path, strategy: str, **sections) -> Path:
    write_source(tmp_path)
    config = {
        "ingestion": {
            "path": "source.txt", "chunking_strategy": strategy,
            "chunk_size": 80, "chunk_overlap": 0,
        },
        "embedding": {
            "provider": "transformers-token", "model": "fixture-encoder",
            "revision": "revision-a", "device": "cpu", "max_tokens": 128,
        },
        "retriever": {"type": "memory"},
        "vectorstore": {"type": "memory", "path": "index"},
        "generation": {"provider": "extractive"},
    }
    for section, values in sections.items():
        config[section].update(values)
    path = tmp_path / "rag.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def assert_late_vectors(store: MemoryVectorStore, source: str) -> None:
    expected = np.array([1.0, float(len(source)), 3.0])
    expected /= np.linalg.norm(expected)
    assert len(store.documents) > 1
    np.testing.assert_allclose(store.embeddings, np.tile(expected, (len(store.documents), 1)))
    for document in store.documents:
        metadata = document.metadata
        assert metadata["chunk_method"] == "late"
        assert document.content == source[metadata["source_char_start"]:metadata["source_char_end"]]


@pytest.mark.parametrize("entrypoint", ["documents", "path", "config"])
def test_late_engine_factories_preserve_full_document_context(tmp_path, monkeypatch, entrypoint):
    model = DocumentContextEmbedding()
    if entrypoint == "documents":
        engine = RAGEngine.from_documents(
            [Document(SOURCE, doc_id="source")], embedding_model=model,
            retriever_type="memory", chunk=True, chunking_strategy="late",
            chunk_size=80, chunk_overlap=0,
        )
    elif entrypoint == "path":
        engine = RAGEngine.from_path(
            write_source(tmp_path), embedding_model=model, retriever_type="vector",
            chunking_strategy="late", chunk_size=80, chunk_overlap=0,
        )
    else:
        monkeypatch.setattr("cheragh.engine._embedding_from_config", lambda config: model)
        engine = RAGEngine.from_config(write_config(tmp_path, "late"))
    assert model.encoded_texts == [SOURCE]
    assert_late_vectors(engine.retriever.store, SOURCE)
    assert engine.retriever.retrieve("Where do the boats dock?", top_k=1)


def test_late_embedding_cache_does_not_hide_token_encoder_capability():
    model = DocumentContextEmbedding()
    engine = RAGEngine.from_documents(
        [Document(SOURCE)], embedding_model=model, retriever_type="memory", chunk=True,
        chunking_strategy="late", chunk_size=80, chunk_overlap=0,
        cache_config={
            "enabled": True, "backend": "memory", "cache_embeddings": True,
            "cache_retrieval": False, "cache_llm": False,
        },
    )
    assert model.encoded_texts == [SOURCE]
    assert_late_vectors(engine.retriever.store, SOURCE)
    engine.retriever.retrieve("harbor", top_k=1)
    engine.retriever.retrieve("harbor", top_k=1)
    assert model.query_calls == 1


def test_late_incremental_index_reuses_unchanged_vectors_and_rebuilds_for_model_changes(tmp_path):
    source = write_source(tmp_path)
    output = tmp_path / "index"
    model = DocumentContextEmbedding()
    options = {"chunking_strategy": "late", "chunk_size": 80, "chunk_overlap": 0}
    index_path(source, output, embedding_model=model, **options)
    reloaded = MemoryVectorStore.load(output, embedding_model=model)
    assert_late_vectors(reloaded, SOURCE)
    assert reloaded.similarity_search("harbor", top_k=1)
    second = index_path(source, output, embedding_model=model, **options)
    assert second["unchanged_files"] == 1
    assert second["changed_files"] == 0
    assert model.encoded_texts == [SOURCE]

    resized = index_path(source, output, embedding_model=model, **{**options, "chunk_size": 60})
    assert resized["indexing_options_changed"] is True
    assert model.encoded_texts == [SOURCE, SOURCE]
    revised_model = DocumentContextEmbedding(revision="two")
    revised = index_path(source, output, embedding_model=revised_model, **{**options, "chunk_size": 60})
    assert revised["embedding_changed"] is True
    assert revised_model.encoded_texts == [SOURCE]


def test_late_partial_update_retains_the_other_sources_vectors(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    first = corpus / "a.txt"
    second = corpus / "b.txt"
    first.write_text(SOURCE, encoding="utf-8")
    second_text = SOURCE + " Additional context for the second report."
    second.write_text(second_text, encoding="utf-8")
    output = tmp_path / "index"
    model = DocumentContextEmbedding()
    options = {"embedding_model": model, "chunking_strategy": "late", "chunk_size": 80, "chunk_overlap": 0}
    index_path(corpus, output, **options)
    original = MemoryVectorStore.load(output, embedding_model=model)
    kept = {
        doc.doc_id: vector.copy() for doc, vector in zip(original.documents, original.embeddings)
        if Path(doc.metadata["source"]).name == "b.txt"
    }
    updated_text = SOURCE + " The first report has changed."
    first.write_text(updated_text, encoding="utf-8")
    result = index_path(corpus, output, **options)
    assert (result["changed_files"], result["unchanged_files"]) == (1, 1)
    assert model.encoded_texts[-1] == updated_text
    assert len(model.encoded_texts) == 3
    updated = MemoryVectorStore.load(output, embedding_model=model)
    for doc, vector in zip(updated.documents, updated.embeddings):
        if doc.doc_id in kept:
            np.testing.assert_array_equal(vector, kept.pop(doc.doc_id))
    assert not kept

    second.unlink()
    result = index_path(corpus, output, **options)
    assert result["deleted_files"] == 1
    assert len(model.encoded_texts) == 3
    assert all(Path(doc.metadata["source"]).name == "a.txt" for doc in MemoryVectorStore.load(
        output, embedding_model=model,
    ).documents)


def test_failed_contextual_token_encoding_keeps_previous_index_snapshot(tmp_path):
    source = write_source(tmp_path)
    output = tmp_path / "index"
    model = DocumentContextEmbedding()
    options = {"embedding_model": model, "chunking_strategy": "late", "chunk_size": 80, "chunk_overlap": 0}
    index_path(source, output, **options)
    filenames = ("documents.jsonl", "embeddings.npy", "manifest.json", "index_manifest.json")
    snapshot = {name: (output / name).read_bytes() for name in filenames}
    source.write_text(SOURCE + " This new revision will fail validation.", encoding="utf-8")
    model.invalid_vectors = True
    with pytest.raises(ValueError, match="finite"):
        index_path(source, output, **options)
    assert snapshot == {name: (output / name).read_bytes() for name in filenames}
    assert_late_vectors(MemoryVectorStore.load(output, embedding_model=model), SOURCE)


def test_semantic_engine_uses_the_supplied_model_to_separate_topics():
    model = TopicEmbedding()
    engine = RAGEngine.from_documents(
        [Document(SOURCE)], embedding_model=model, retriever_type="memory", chunk=True,
        chunking_strategy="semantic", chunk_size=1000, chunk_overlap=0,
        chunking_options={"breakpoint_threshold": 0.75, "buffer_size": 0},
    )
    chunks = engine.retriever.store.documents
    assert len(chunks) == 2
    assert "Harbor" in chunks[0].content and "Finance" not in chunks[0].content
    assert "Finance" in chunks[1].content and "Harbor" not in chunks[1].content
    assert len(model.batches[0]) == 4
    assert model.batches[-1] == [doc.content for doc in chunks]
    assert engine.retriever.retrieve("finance", top_k=1)[0].doc_id == chunks[1].doc_id


def test_semantic_threshold_change_rebuilds_unchanged_sources(tmp_path):
    source = write_source(tmp_path)
    output = tmp_path / "index"
    model = TopicEmbedding()
    options = {
        "embedding_model": model, "chunking_strategy": "semantic",
        "chunk_size": 1000, "chunk_overlap": 0,
    }
    index_path(source, output, chunking_options={"breakpoint_threshold": 0.75}, **options)
    assert len(MemoryVectorStore.load(output, embedding_model=model).documents) == 2
    changed = index_path(source, output, chunking_options={"breakpoint_threshold": 0.0}, **options)
    assert changed["changed_files"] == 1
    assert changed["indexing_options_changed"] is True
    assert len(MemoryVectorStore.load(output, embedding_model=model).documents) == 1
    batch_count = len(model.batches)
    reused = index_path(source, output, chunking_options={"breakpoint_threshold": 0.0}, **options)
    assert reused["unchanged_files"] == 1
    assert len(model.batches) == batch_count


@pytest.mark.parametrize("entrypoint", ["documents", "path", "config"])
def test_contextual_engine_embeds_generated_context_but_cites_original_text(tmp_path, monkeypatch, entrypoint):
    model = ContextEmbedding()
    llm = ContextLLM()
    if entrypoint == "documents":
        engine = RAGEngine.from_documents(
            [Document(SOURCE)], embedding_model=model, llm_client=llm,
            retriever_type="memory", chunk=True, chunking_strategy="contextual",
            chunk_size=80, chunk_overlap=0,
        )
    elif entrypoint == "path":
        engine = RAGEngine.from_path(
            write_source(tmp_path), embedding_model=model, llm_client=llm,
            retriever_type="memory", chunking_strategy="contextual", chunk_size=80, chunk_overlap=0,
        )
    else:
        monkeypatch.setattr("cheragh.engine._embedding_from_config", lambda config: model)
        monkeypatch.setattr("cheragh.engine._llm_from_config", lambda config: llm)
        engine = RAGEngine.from_config(write_config(
            tmp_path, "contextual", generation={"provider": "openai", "model": "context-one"},
        ))
    store = engine.retriever.store
    embedded = [text for batch in model.batches for text in batch]
    assert len(embedded) == len(store.documents) == len(llm.prompts)
    assert len(store.documents) > 1
    for doc, text, vector in zip(store.documents, embedded, store.embeddings):
        metadata = doc.metadata
        assert doc.content == SOURCE[metadata["source_char_start"]:metadata["source_char_end"]]
        assert "ORCHID" not in doc.content
        assert metadata["contextual_context_generated"] is True
        assert text == metadata["contextual_context"] + "\n\n" + doc.content
        np.testing.assert_array_equal(vector, [7.0, len(text)])


def test_contextual_index_rebuilds_for_generation_and_strategy_option_changes(tmp_path):
    source = write_source(tmp_path)
    output = tmp_path / "index"
    model = ContextEmbedding()
    first_llm = ContextLLM("context-one")
    options = {"embedding_model": model, "chunking_strategy": "contextual", "chunk_size": 80, "chunk_overlap": 0}
    index_path(source, output, llm_client=first_llm, **options)
    prompt_count = len(first_llm.prompts)
    batch_count = len(model.batches)
    reused = index_path(source, output, llm_client=first_llm, **options)
    assert reused["unchanged_files"] == 1
    assert len(first_llm.prompts) == prompt_count
    assert len(model.batches) == batch_count
    second_llm = ContextLLM("context-two")
    changed = index_path(source, output, llm_client=second_llm, **options)
    assert changed["indexing_options_changed"] is True
    assert len(second_llm.prompts) == prompt_count
    bounded = index_path(
        source, output, llm_client=second_llm, chunking_options={"max_context_chars": 200}, **options,
    )
    assert bounded["indexing_options_changed"] is True
    assert len(second_llm.prompts) == 2 * prompt_count
    assert load_manifest(output).metadata["chunking_options"] == {"max_context_chars": 200}
    reloaded = MemoryVectorStore.load(output, embedding_model=model)
    assert all("context-two" in doc.metadata["contextual_context"] for doc in reloaded.documents)
    assert reloaded.similarity_search("harbor", top_k=1)


def test_contextual_config_fingerprint_tracks_generation_without_persisting_secrets(tmp_path, monkeypatch):
    model = ContextEmbedding()
    clients = []

    def llm_from_config(config):
        client = ContextLLM(config["model"])
        clients.append(client)
        return client

    monkeypatch.setattr("cheragh.engine._embedding_from_config", lambda config: model)
    monkeypatch.setattr("cheragh.engine._llm_from_config", llm_from_config)
    secret = "test-secret-that-must-never-be-written-to-an-index"
    config_path = write_config(
        tmp_path, "contextual",
        generation={"provider": "openai", "model": "context-one", "api_key": secret},
    )
    index_from_config(config_path)
    assert clients[0].prompts
    reused = index_from_config(config_path)
    assert reused["unchanged_files"] == 1
    assert not clients[1].prompts
    config = json.loads(config_path.read_text())
    config["generation"]["model"] = "context-two"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    rebuilt = index_from_config(config_path)
    assert rebuilt["indexing_options_changed"] is True
    assert clients[2].prompts
    assert all(secret not in path.read_text() for path in (tmp_path / "index").glob("*.json*"))


def test_contextual_explicit_fingerprint_invalidates_an_opaque_generation_deployment(tmp_path):
    source = write_source(tmp_path)
    output = tmp_path / "index"
    llm = ContextLLM()
    options = {
        "embedding_model": ContextEmbedding(), "llm_client": llm,
        "chunking_strategy": "contextual", "chunk_size": 80, "chunk_overlap": 0,
    }
    index_path(source, output, generation_fingerprint="deployment-r1", **options)
    prompt_count = len(llm.prompts)
    rebuilt = index_path(source, output, generation_fingerprint="deployment-r2", **options)
    assert rebuilt["indexing_options_changed"] is True
    assert len(llm.prompts) == 2 * prompt_count
    reused = index_path(source, output, generation_fingerprint="deployment-r2", **options)
    assert reused["unchanged_files"] == 1
    assert len(llm.prompts) == 2 * prompt_count


@pytest.mark.parametrize("strategy", ["late", "contextual"])
def test_non_vector_retrievers_reject_precomputed_chunking_before_model_work(strategy):
    model = DocumentContextEmbedding()
    llm = ContextLLM()
    with pytest.raises(ValueError, match="memory|vector|retriever"):
        RAGEngine.from_documents(
            [Document(SOURCE)], embedding_model=model, llm_client=llm,
            retriever_type="hybrid", chunk=True, chunking_strategy=strategy,
        )
    assert model.encoded_texts == []
    assert llm.prompts == []


@pytest.mark.parametrize("strategy", ["late", "contextual"])
def test_plain_ingestion_refuses_to_discard_advanced_vectors(tmp_path, strategy):
    with pytest.raises(ValueError, match="index_path|chunk_and_embed|RAGEngine"):
        ingest_path(write_source(tmp_path), chunking_strategy=strategy)


@pytest.mark.parametrize("entrypoint", ["engine", "index"])
@pytest.mark.parametrize("strategy, embedding, generation", [
    ("late", {"provider": "hashing"}, {"provider": "extractive"}),
    ("semantic", {"provider": "hashing"}, {"provider": "extractive"}),
    ("contextual", {"provider": "sentence-transformers"}, {"provider": "extractive"}),
])
def test_invalid_advanced_provider_combinations_fail_before_initialization(
    tmp_path, monkeypatch, entrypoint, strategy, embedding, generation,
):
    from unittest.mock import Mock

    embedding_factory = Mock(side_effect=AssertionError("Provider must not initialize before validation"))
    llm_factory = Mock(side_effect=AssertionError("Provider must not initialize before validation"))
    monkeypatch.setattr("cheragh.engine._embedding_from_config", embedding_factory)
    monkeypatch.setattr("cheragh.engine._llm_from_config", llm_factory)
    config_path = write_config(tmp_path, strategy, embedding=embedding, generation=generation)
    config = json.loads(config_path.read_text())
    config["embedding"] = embedding
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=f"(?i){strategy}"):
        if entrypoint == "engine":
            RAGEngine.from_config(config_path)
        else:
            index_from_config(config_path)
    embedding_factory.assert_not_called()
    llm_factory.assert_not_called()
    assert not (tmp_path / "index").exists()


@pytest.mark.parametrize("strategy", ["late", "contextual"])
def test_unsupported_configured_retriever_is_rejected_before_provider_initialization(
    tmp_path, monkeypatch, strategy,
):
    from unittest.mock import Mock

    embedding_factory = Mock(side_effect=AssertionError("Embedding initialization preceded validation"))
    llm_factory = Mock(side_effect=AssertionError("Generation initialization preceded validation"))
    monkeypatch.setattr("cheragh.engine._embedding_from_config", embedding_factory)
    monkeypatch.setattr("cheragh.engine._llm_from_config", llm_factory)
    config_path = write_config(
        tmp_path, strategy, retriever={"type": "hybrid"}, vectorstore={"type": "hybrid"},
        generation={"provider": "openai", "model": "context-one"},
    )
    with pytest.raises(ValueError, match="memory|vector|retriever"):
        RAGEngine.from_config(config_path)
    embedding_factory.assert_not_called()
    llm_factory.assert_not_called()


def test_transformers_token_provider_uses_validated_model_settings(tmp_path, monkeypatch):
    from unittest.mock import Mock

    model = DocumentContextEmbedding()
    constructor = Mock(return_value=model)
    monkeypatch.setattr("cheragh.ingestion.chunkers.late.TransformersTokenEmbedding", constructor)
    engine = RAGEngine.from_config(write_config(tmp_path, "late"))
    constructor.assert_called_once_with(
        model_name="fixture-encoder", revision="revision-a", device="cpu", max_tokens=128,
    )
    assert_late_vectors(engine.retriever.store, SOURCE)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("changed_identity", ["model", "wrapper_fingerprint"])
def test_contextual_cached_llm_identity_rebuilds_then_reuses_across_fresh_instances(
    tmp_path, depth, changed_identity,
):
    from cheragh.cache import CachedLLMClient, MemoryCache

    source = write_source(tmp_path)
    output = tmp_path / "index"
    options = {
        "embedding_model": ContextEmbedding(), "chunking_strategy": "contextual",
        "chunk_size": 80, "chunk_overlap": 0,
    }
    calls = []
    secret_fingerprint = "opaque-deployment-secret-must-only-be-hashed"
    for revision in ("one", "two", "two"):
        model_name = f"context-{revision}" if changed_identity == "model" else "context-one"
        original = ContextLLM(model_name)
        wrapped = original
        for _ in range(depth):
            wrapped = CachedLLMClient(
                wrapped, MemoryCache(),
                fingerprint=(
                    f"{secret_fingerprint}-{revision}"
                    if changed_identity == "wrapper_fingerprint" else None
                ),
            )
        result = index_path(source, output, llm_client=wrapped, **options)
        calls.append((result, original))
    assert calls[0][1].prompts
    assert calls[1][0]["indexing_options_changed"] is True
    assert calls[1][0]["changed_files"] == 1
    assert calls[1][1].prompts
    assert calls[2][0]["indexing_options_changed"] is False
    assert calls[2][0]["unchanged_files"] == 1
    assert not calls[2][1].prompts
    assert all(secret_fingerprint not in path.read_text() for path in output.glob("*.json*"))
    loaded = MemoryVectorStore.load(output, options["embedding_model"])
    expected_model = "context-two" if changed_identity == "model" else "context-one"
    assert all(expected_model in doc.metadata["contextual_context"] for doc in loaded.documents)
