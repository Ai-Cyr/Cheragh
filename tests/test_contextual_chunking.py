"""Contextual retrieval keeps evidence unchanged and indexes contextual vectors."""
from __future__ import annotations

import json
from unittest.mock import Mock

import numpy as np
import pytest

from cheragh import Document, EmbeddingModel
from cheragh.ingestion.chunkers.contextual import ContextualChunker


class RecordingEmbedding(EmbeddingModel):
    def __init__(self):
        self.batches = []

    def embed_documents(self, texts):
        self.batches.append(list(texts))
        return np.array([[float("Aurora" in text), float(len(text))] for text in texts])

    def embed_query(self, text):
        return np.array([float("Aurora" in text), float(len(text))])


class ContextLLM:
    def __init__(self, answer="Budget information for the Aurora programme."):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.answer


def test_contextual_embedding_uses_full_document_context_but_preserves_source_evidence():
    text = "The programme is Aurora.\r\nIts annual budget is 42 euros."
    source = Document(text, doc_id="budget", metadata={"acl": {"allow": ["finance"]}})
    embedding = RecordingEmbedding()
    llm = ContextLLM()
    chunker = ContextualChunker(embedding, llm, chunk_size=32, chunk_overlap=0, embedding_batch_size=1)
    chunks, vectors = chunker.chunk_and_embed([source])
    assert len(chunks) == len(llm.prompts) == len(embedding.batches) == 2
    assert vectors.shape == (2, 2)
    assert "Aurora" not in chunks[1].content
    assert vectors[1, 0] == 1, "The second passage must be embedded with its generated document context"
    for index, chunk in enumerate(chunks):
        assert chunk.content == text[chunk.metadata["source_char_start"]:chunk.metadata["source_char_end"]]
        assert chunk.metadata["parent_doc_id"] == "budget"
        assert chunk.metadata["chunker"] == "contextual"
        assert chunk.metadata["contextual_context_generated"] is True
        assert chunk.metadata["contextual_context"] == llm.answer
        assert embedding.batches[index][0].endswith(chunk.content)
        prompt = llm.prompts[index]
        source_line = prompt.split("SOURCE_DOCUMENT_JSON: ")[1].split("\n")[0]
        assert json.loads(source_line) == text
        assert json.loads(prompt.split("SOURCE_PASSAGE_JSON: ")[1]) == chunk.content
    chunks[0].metadata["acl"]["allow"].append("outsider")
    assert chunks[1].metadata["acl"] == source.metadata["acl"] == {"allow": ["finance"]}
    assert "contextual_context" not in source.metadata


def test_all_document_bounds_are_checked_before_any_provider_calls():
    embedding = RecordingEmbedding()
    llm = ContextLLM()
    chunker = ContextualChunker(embedding, llm, max_document_chars=10)
    with pytest.raises(ValueError, match="max_document_chars"):
        chunker.chunk_and_embed([Document("short"), Document("x" * 11)])
    assert llm.prompts == embedding.batches == []


def test_pathological_overlap_is_rejected_before_context_generation():
    embedding = RecordingEmbedding()
    llm = ContextLLM()
    chunker = ContextualChunker(
        embedding, llm, chunk_size=100, chunk_overlap=99, max_chunks_per_document=3,
    )
    with pytest.raises(ValueError, match="max_chunks_per_document"):
        chunker.chunk_and_embed([Document("x" * 500)])
    assert llm.prompts == embedding.batches == []


@pytest.mark.parametrize("response", [None, 42, [], "", " \n", "x" * 21])
def test_invalid_or_oversized_generated_context_never_reaches_embeddings(response):
    embedding = RecordingEmbedding()
    chunker = ContextualChunker(embedding, ContextLLM(response), max_context_chars=20)
    with pytest.raises(ValueError, match="context|ContextualChunker"):
        chunker.chunk_and_embed([Document("Short evidence.")])
    assert embedding.batches == []


@pytest.mark.parametrize("matrix", [
    [], [[1], [2]], [[float("nan")]], [[float("inf")]], [[]], [1, 2],
    [[True, False]], [["1", "2"]], [[1 + 2j, 3 + 0j]],
])
def test_contextual_embedding_shape_and_values_are_validated(matrix):
    embedding = Mock(embed_documents=Mock(return_value=matrix))
    chunker = ContextualChunker(embedding, ContextLLM())
    with pytest.raises(ValueError, match="finite embedding"):
        chunker.chunk_and_embed([Document("One chunk.")])


def test_embedding_dimensions_cannot_change_between_batches():
    embedding = Mock(embed_documents=Mock(side_effect=[[[1, 2]], [[1, 2, 3]]]))
    chunker = ContextualChunker(embedding, ContextLLM(), chunk_size=8, chunk_overlap=0, embedding_batch_size=1)
    with pytest.raises(ValueError, match="dimensions changed"):
        chunker.chunk_and_embed([Document("First. Second.")])


def test_empty_input_has_no_provider_calls_and_no_fabricated_vectors():
    embedding = RecordingEmbedding()
    llm = ContextLLM()
    chunker = ContextualChunker(embedding, llm)
    chunks, vectors = chunker.chunk_and_embed([Document(" \r\n ")])
    assert chunks == [] and vectors.shape == (0, 0)
    assert llm.prompts == embedding.batches == []


def test_plain_split_cannot_silently_discard_contextual_embeddings():
    with pytest.raises(RuntimeError, match="chunk_and_embed"):
        ContextualChunker(RecordingEmbedding(), ContextLLM()).split_documents([Document("Evidence")])


@pytest.mark.parametrize("field", ["max_document_chars", "max_context_chars", "max_chunks_per_document", "embedding_batch_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_contextual_resource_budgets_require_positive_integers(field, value):
    with pytest.raises((TypeError, ValueError)):
        ContextualChunker(RecordingEmbedding(), ContextLLM(), **{field: value})


def test_contextual_retrieval_requires_explicit_generation_and_embedding_providers():
    with pytest.raises(TypeError, match="configured llm_client"):
        ContextualChunker(RecordingEmbedding(), None)
    with pytest.raises(TypeError, match="embedding_model"):
        ContextualChunker(None, ContextLLM())
