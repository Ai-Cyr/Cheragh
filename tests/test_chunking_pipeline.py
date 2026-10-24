"""Chunking choices reach every ingestion entrypoint and invalidate snapshots."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cheragh import Document, HashingEmbedding, MemoryVectorStore, RAGEngine
from cheragh.cli.main import main
from cheragh.config.schema import validate_config
from cheragh.indexing import index_from_config, index_path, load_manifest, save_manifest


def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text(
        "Le phare éclaire le port. Les bateaux arrivent.\n\n"
        "Le jardin accueille les visiteurs. La visite commence ici.",
        encoding="utf-8",
    )
    return path


def config_file(tmp_path: Path, *, strategy: str = "paragraph") -> Path:
    corpus(tmp_path)
    path = tmp_path / "rag.json"
    path.write_text(json.dumps({
        "ingestion": {
            "path": "corpus.txt", "chunking_strategy": strategy,
            "chunk_size": 60, "chunk_overlap": 0,
        },
        "embedding": {"provider": "hashing", "dimension": 32},
        "retriever": {"type": "memory"},
        "vectorstore": {"type": "memory", "path": "index"},
        "generation": {"provider": "extractive"},
    }), encoding="utf-8")
    return path


@pytest.mark.parametrize("strategy", ["recursive", "sentence", "paragraph", "token"])
def test_configuration_accepts_the_supported_chunking_strategies(strategy):
    assert validate_config({"ingestion": {"chunking_strategy": strategy}}).ingestion.chunking_strategy == strategy


@pytest.mark.parametrize("strategy", ["sentences", "unknown", "Python-AST", True, 1, None, []])
def test_configuration_rejects_unknown_or_untyped_chunking_choices(strategy):
    with pytest.raises(ValidationError):
        validate_config({"ingestion": {"chunking_strategy": strategy}})


@pytest.mark.parametrize("options", [
    {"chunking_strategy": "unknown"},
    {"chunking_strategy": "paragraph", "chunk_size": 0},
    {"chunking_strategy": "token", "chunk_size": 3, "chunk_overlap": 3},
])
def test_invalid_chunking_options_fail_before_scanning_or_creating_output(tmp_path, options):
    output = tmp_path / "new-index"
    with pytest.raises(ValueError):
        index_path(tmp_path / "does-not-exist", output, **options)
    assert not output.exists()


@pytest.mark.parametrize("entrypoint", ["documents", "path", "config"])
def test_engine_factories_really_apply_paragraph_chunking(tmp_path, entrypoint):
    source = corpus(tmp_path)
    if entrypoint == "documents":
        engine = RAGEngine.from_documents(
            [Document(source.read_text(), doc_id="guide")],
            embedding_model=HashingEmbedding(32), retriever_type="memory", chunk=True,
            chunking_strategy="paragraph", chunk_size=60, chunk_overlap=0,
        )
    elif entrypoint == "path":
        engine = RAGEngine.from_path(
            source, embedding_model=HashingEmbedding(32), retriever_type="memory",
            chunking_strategy="paragraph", chunk_size=60, chunk_overlap=0,
        )
    else:
        engine = RAGEngine.from_config(config_file(tmp_path))
    documents = engine.retriever.store.documents
    assert len(documents) == 2
    assert {document.metadata["chunker"] for document in documents} == {"paragraph"}
    assert documents[0].content.startswith("Le phare")
    assert documents[1].content.startswith("Le jardin")
    response = engine.ask(documents[1].content, top_k=1)
    assert "jardin" in response.answer
    assert response.retrieved_documents[0].doc_id == documents[1].doc_id


def test_configured_strategy_does_not_rechunk_explicitly_supplied_documents(tmp_path):
    engine = RAGEngine.from_config(
        config_file(tmp_path, strategy="paragraph"),
        documents=[Document("This supplied document is already chunked.\n\nIts paragraphs must remain together.", doc_id="given")],
    )
    assert [doc.doc_id for doc in engine.retriever.store.documents] == ["given"]


def test_strategy_change_rebuilds_unchanged_sources_then_reuses_new_chunks(tmp_path):
    source = corpus(tmp_path)
    output = tmp_path / "index"
    index_path(source, output, chunk_size=60, chunk_overlap=0)
    changed = index_path(source, output, chunking_strategy="paragraph", chunk_size=60, chunk_overlap=0)
    assert changed["changed_files"] == 1
    assert changed["indexing_options_changed"] is True
    manifest = load_manifest(output)
    assert manifest.metadata["chunking_strategy"] == "paragraph"
    assert manifest.metadata["indexing_options"]["chunking_strategy"] == "paragraph"
    assert {doc.metadata["chunker"] for doc in MemoryVectorStore.load(output).documents} == {"paragraph"}
    unchanged = index_path(source, output, chunking_strategy="paragraph", chunk_size=60, chunk_overlap=0)
    assert unchanged["unchanged_files"] == 1
    assert unchanged["indexing_options_changed"] is False


def test_legacy_manifest_without_strategy_keeps_recursive_incremental_reuse(tmp_path):
    source = corpus(tmp_path)
    output = tmp_path / "index"
    index_path(source, output, chunk_size=60, chunk_overlap=0)
    legacy = load_manifest(output)
    legacy.metadata.pop("chunking_strategy")
    legacy.metadata["indexing_options"].pop("chunking_strategy")
    legacy.metadata["indexing_options"].pop("chunking_options")
    save_manifest(output, legacy)
    result = index_path(source, output, chunk_size=60, chunk_overlap=0)
    assert result["unchanged_files"] == 1
    assert result["changed_files"] == 0
    assert result["indexing_options_changed"] is False


def test_invalid_strategy_does_not_replace_a_previous_good_snapshot(tmp_path):
    source = corpus(tmp_path)
    output = tmp_path / "index"
    index_path(source, output, chunk_size=60, chunk_overlap=0)
    snapshot_files = ("documents.jsonl", "embeddings.npy", "manifest.json", "index_manifest.json")
    snapshot = {name: (output / name).read_bytes() for name in snapshot_files}
    with pytest.raises(ValueError):
        index_path(source, output, chunking_strategy="invalid", chunk_size=60, chunk_overlap=0)
    assert snapshot == {name: (output / name).read_bytes() for name in snapshot_files}
    assert MemoryVectorStore.load(output).documents


def test_index_from_config_applies_strategy_and_explicit_override(tmp_path):
    config = config_file(tmp_path)
    index_from_config(config)
    assert load_manifest(tmp_path / "index").metadata["chunking_strategy"] == "paragraph"
    result = index_from_config(config, chunking_strategy="token", chunk_size=4, chunk_overlap=1)
    assert result["indexing_options_changed"] is True
    assert load_manifest(tmp_path / "index").metadata["chunking_strategy"] == "token"
    assert {doc.metadata["chunker"] for doc in MemoryVectorStore.load(tmp_path / "index").documents} == {"token"}


@pytest.mark.parametrize("configured", [False, True])
def test_cli_strategy_selects_real_chunker_and_overrides_config(tmp_path, capsys, configured):
    if configured:
        arguments = ["--config", str(config_file(tmp_path))]
    else:
        arguments = [str(corpus(tmp_path)), "--output", str(tmp_path / "index")]
    assert main([
        "index", *arguments, "--chunking-strategy", "token", "--chunk-size", "4", "--chunk-overlap", "1",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["indexed_documents"] > 0
    chunks = MemoryVectorStore.load(tmp_path / "index").documents
    assert {doc.metadata["chunker"] for doc in chunks} == {"token"}
    assert max(len(doc.content) for doc in chunks) > 4, "Token budgets must not be interpreted as characters"


@pytest.mark.parametrize("embedding", [
    {"max_tokens": 0}, {"max_tokens": True}, {"max_tokens": "512"},
    {"revision": " "}, {"device": " "}, {"model": " "},
])
def test_token_encoder_config_rejects_invalid_settings_before_provider_creation(embedding):
    with pytest.raises(ValidationError):
        validate_config({"embedding": {"provider": "transformers-token", **embedding}})


@pytest.mark.parametrize("options", [
    {"breakpoint_threshold": True}, {"breakpoint_percentile": 100},
    {"buffer_size": -1}, {"buffer_size": 1.5}, {"embedding_batch_size": 0},
    {"api_key": "must-not-be-echoed"},
])
def test_invalid_strategy_options_fail_config_validation_without_echoing_values(options):
    with pytest.raises(ValidationError) as error:
        validate_config({
            "ingestion": {"chunking_strategy": "semantic", "chunking_options": options},
            "embedding": {"provider": "sentence-transformers"},
        })
    assert "must-not-be-echoed" not in str(error.value)
