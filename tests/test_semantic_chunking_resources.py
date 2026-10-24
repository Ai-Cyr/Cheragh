"""Semantic neighborhood budgets, effective cutoffs and exact fallback evidence."""
from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from cheragh import Document, EmbeddingModel, HashingEmbedding
from cheragh.ingestion.chunkers.advanced import SemanticChunker


class TopicEmbedding(EmbeddingModel):
    def __init__(self):
        self.batches = []

    def embed_documents(self, texts):
        self.batches.append(list(texts))
        vectors = {"First.": [1, 0], "Second.": [0.8, 0.6], "Third.": [0, 1], "Fourth.": [0, 1]}
        return np.array([vectors[text] for text in texts], dtype=float)

    def embed_query(self, text):
        return np.array([1.0, 0.0])


def test_percentile_cuts_at_topic_shift_and_records_the_actual_cosine_cutoff():
    embedding = TopicEmbedding()
    chunker = SemanticChunker(
        embedding, breakpoint_threshold=0.1, breakpoint_percentile=50,
        min_chunk_size=0, embedding_batch_size=2,
    )
    chunks = chunker.split_documents([Document("First. Second. Third. Fourth.", doc_id="topics")])
    assert [chunk.content for chunk in chunks] == ["First. Second.", "Third. Fourth."]
    assert [len(batch) for batch in embedding.batches] == [2, 2]
    for chunk in chunks:
        assert chunk.metadata["semantic_breakpoint_threshold"] == pytest.approx(0.8)
        assert chunk.metadata["semantic_breakpoint_percentile"] == 50
        assert chunk.metadata["semantic_buffer_size"] == 0


def test_buffered_sentence_neighborhoods_are_encoded_in_bounded_batches():
    embedding = Mock(embed_documents=Mock(side_effect=lambda texts: np.ones((len(texts), 2))))
    chunker = SemanticChunker(embedding, buffer_size=1, embedding_batch_size=2, max_buffer_chars=22)
    chunker.split_text("First. Second. Third. Fourth.")
    batches = [call.args[0] for call in embedding.embed_documents.call_args_list]
    assert batches == [
        ["First. Second.", "First. Second. Third."],
        ["Second. Third. Fourth.", "Third. Fourth."],
    ]


def test_oversized_buffer_is_rejected_before_constructing_or_encoding_neighborhoods():
    embedding = Mock()
    chunker = SemanticChunker(embedding, buffer_size=100_000, max_buffer_chars=15)
    with pytest.raises(ValueError, match="max_buffer_chars"):
        chunker.split_text("First. Second. Third. Fourth.")
    embedding.embed_documents.assert_not_called()


def test_sentence_count_limit_fails_before_provider_calls():
    embedding = Mock()
    with pytest.raises(ValueError, match="max_sentences"):
        SemanticChunker(embedding, max_sentences=2).split_text("First. Second. Third.")
    embedding.embed_documents.assert_not_called()


def test_batched_semantic_vectors_cannot_change_dimension():
    embedding = Mock(embed_documents=Mock(side_effect=[[[1, 2]], [[1, 2, 3]]]))
    with pytest.raises(ValueError, match="dimensions changed"):
        SemanticChunker(embedding, embedding_batch_size=1).split_text("First. Second.")


@pytest.mark.parametrize("matrix", [
    [[float("nan"), 1], [1, 0]], [[1, 0]], [[], []],
    [[True], [False]], [["1"], ["2"]], [[1 + 2j], [3 + 0j]],
])
def test_bad_semantic_embeddings_are_rejected(matrix):
    embedding = Mock(embed_documents=Mock(return_value=matrix))
    with pytest.raises(ValueError, match="finite embedding"):
        SemanticChunker(embedding).split_text("First. Second.")


def test_hard_size_limit_preserves_raw_source_and_short_tail_when_minimums_cannot_fit():
    text = "  Une\t\tphrase très longue" + " contenant des faits" * 8 + ".\r\n\r\nFin. "
    source = Document(text, doc_id="source", metadata={"acl": {"allow": ["a"]}})
    chunks = SemanticChunker(
        HashingEmbedding(16), max_chunk_size=24, min_chunk_size=100, min_sentences=10,
    ).split_documents([source])
    covered = set()
    for chunk in chunks:
        start, end = chunk.metadata["source_char_start"], chunk.metadata["source_char_end"]
        assert chunk.content == text[start:end]
        assert len(chunk.content) <= 24
        assert chunk.metadata["parent_doc_id"] == "source"
        assert chunk.metadata["chunk_method"] in {"semantic", "semantic+sentence"}
        covered.update(range(start, end))
    assert {index for index, character in enumerate(text) if not character.isspace()} <= covered
    assert chunks[-1].content.endswith("Fin.")
    chunks[0].metadata["acl"]["allow"].append("b")
    assert source.metadata["acl"] == chunks[-1].metadata["acl"] == {"allow": ["a"]}
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_single_sentence_percentile_does_not_invent_a_computed_threshold():
    embedding = Mock()
    chunks = SemanticChunker(embedding, breakpoint_percentile=90).split_documents([Document("One sentence.")])
    assert chunks[0].metadata["semantic_breakpoint_threshold"] is None
    embedding.embed_documents.assert_not_called()


@pytest.mark.parametrize("name", ["embedding_batch_size", "max_buffer_chars", "max_sentences"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_semantic_resource_budgets_require_positive_integers(name, value):
    with pytest.raises((TypeError, ValueError)):
        SemanticChunker(HashingEmbedding(), **{name: value})


@pytest.mark.parametrize("name", ["breakpoint_threshold", "breakpoint_percentile"])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "0.5"])
def test_ambiguous_or_non_finite_thresholds_are_rejected(name, value):
    with pytest.raises(ValueError):
        SemanticChunker(HashingEmbedding(), **{name: value})
