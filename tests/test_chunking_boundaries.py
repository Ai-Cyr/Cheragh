"""Coverage and provenance contracts for sentence/paragraph chunking."""
from __future__ import annotations

import random

import pytest

from cheragh import Document
from cheragh.ingestion.chunkers.boundaries import ParagraphChunker, SentenceTextChunker


CHUNKERS = (SentenceTextChunker, ParagraphChunker)


def assert_complete_source_coverage(chunker, text):
    chunks = chunker.split_text_with_offsets(text)
    covered = set()
    previous_start = -1
    for chunk in chunks:
        assert chunk.text == text[chunk.source_char_start:chunk.source_char_end]
        assert 0 < len(chunk.text) <= chunker.chunk_size
        assert chunk.source_char_start > previous_start
        assert chunk.normalized_char_start == chunk.source_char_start
        assert chunk.normalized_char_end == chunk.source_char_end
        covered.update(range(chunk.source_char_start, chunk.source_char_end))
        previous_start = chunk.source_char_start
    assert {i for i, character in enumerate(text) if not character.isspace()} <= covered
    return chunks


def test_sentence_packing_retains_whole_sentence_overlap():
    chunker = SentenceTextChunker(chunk_size=24, chunk_overlap=10)
    assert chunker.split_text("Alpha done. Beta done. Gamma done.") == [
        "Alpha done. Beta done.", "Beta done. Gamma done.",
    ]


def test_paragraph_packing_retains_whole_paragraph_overlap():
    chunker = ParagraphChunker(chunk_size=20, chunk_overlap=8)
    assert chunker.split_text("Alpha123\n\nBeta1234\n\nGamma123") == [
        "Alpha123\n\nBeta1234", "Beta1234\n\nGamma123",
    ]


def test_sentence_boundaries_keep_abbreviations_decimals_and_closing_quotes():
    text = 'Dr. Martin note 3.14. "Très bien!" Mme Dupont confirme. Fin.'
    chunks = SentenceTextChunker(chunk_size=25, chunk_overlap=0).split_text(text)
    assert chunks == ['Dr. Martin note 3.14.', '"Très bien!"', "Mme Dupont confirme. Fin."]


def test_sentence_boundaries_understand_initials_and_cjk_without_spaces():
    assert SentenceTextChunker(chunk_size=25, chunk_overlap=0).split_text(
        "M. J. Dupont arrive. Fin."
    ) == ["M. J. Dupont arrive. Fin."]
    text = "第一句话。第二句话！「第三句话？」结束。"
    chunks = SentenceTextChunker(chunk_size=8, chunk_overlap=0).split_text(text)
    assert chunks == ["第一句话。", "第二句话！", "「第三句话？」", "结束。"]


def test_paragraph_blank_lines_allow_spaces_and_crlf_without_splitting_single_newlines():
    text = "  Première ligne\r\nseconde ligne.\r\n \t\r\nAutre paragraphe.\r\n\r\nFin. "
    chunks = ParagraphChunker(chunk_size=35, chunk_overlap=0).split_text(text)
    assert chunks == ["Première ligne\r\nseconde ligne.", "Autre paragraphe.\r\n\r\nFin."]
    assert_complete_source_coverage(ParagraphChunker(chunk_size=35, chunk_overlap=0), text)


@pytest.mark.parametrize("chunker_class", CHUNKERS)
def test_long_units_and_short_tail_preserve_every_non_whitespace_character(chunker_class):
    text = "é" * 43 + "\r\n" + "🙂" * 20 + ".\n\nZ."
    chunks = assert_complete_source_coverage(chunker_class(chunk_size=16, chunk_overlap=5), text)
    assert chunks[-1].text.endswith("Z.")


@pytest.mark.parametrize("chunker_class", CHUNKERS)
def test_high_overlap_makes_progress_and_never_loops_over_context_only(chunker_class):
    text = "Alpha.\n\nBeta.\n\n" + "x" * 90 + ".\n\nGamma.\n\nFin."
    chunks = assert_complete_source_coverage(chunker_class(chunk_size=12, chunk_overlap=11), text)
    assert len(chunks) <= len(text)
    assert len({(chunk.source_char_start, chunk.source_char_end) for chunk in chunks}) == len(chunks)


@pytest.mark.parametrize("chunker_class", CHUNKERS)
def test_repeated_text_offsets_refer_to_the_actual_occurrence(chunker_class):
    text = "Encore.\r\n\r\nEncore.\r\n\r\nEncore."
    chunks = assert_complete_source_coverage(chunker_class(chunk_size=7, chunk_overlap=0), text)
    assert [chunk.source_char_start for chunk in chunks] == [0, 11, 22]


@pytest.mark.parametrize("chunker_class", CHUNKERS)
def test_document_ids_and_nested_authorization_metadata_are_isolated(chunker_class):
    source = Document("First.\n\nSecond.\n\nThird.", metadata={"acl": {"allow": ["a"]}}, doc_id="source")
    chunker = chunker_class(chunk_size=9, chunk_overlap=0)
    chunks = chunker.split_documents([source])
    assert len(chunks) == 3
    assert len({chunk.doc_id for chunk in chunks}) == 3
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["parent_doc_id"] == "source"
        assert chunk.metadata["chunk_index"] == index
        assert chunk.content == source.content[chunk.metadata["source_char_start"]:chunk.metadata["source_char_end"]]
    chunks[0].metadata["acl"]["allow"].append("b")
    assert source.metadata["acl"] == {"allow": ["a"]}
    assert chunks[1].metadata["acl"] == {"allow": ["a"]}
    assert [chunk.doc_id for chunk in chunks] == [chunk.doc_id for chunk in chunker.split_documents([source])]


@pytest.mark.parametrize("chunker_class", CHUNKERS)
@pytest.mark.parametrize("text", ["", " ", "\r\n\t\r\n"])
def test_empty_and_whitespace_documents_do_not_create_chunks(chunker_class, text):
    assert chunker_class().split_text(text) == []


@pytest.mark.parametrize("chunker_class", CHUNKERS)
@pytest.mark.parametrize("settings", [
    {"chunk_size": 0}, {"chunk_size": -1}, {"chunk_size": True}, {"chunk_size": 2.5},
    {"chunk_overlap": -1}, {"chunk_overlap": False}, {"chunk_overlap": 0.5},
    {"chunk_size": 3, "chunk_overlap": 3},
])
def test_invalid_budgets_fail_before_splitting(chunker_class, settings):
    with pytest.raises((TypeError, ValueError)):
        chunker_class(**settings)


@pytest.mark.parametrize("chunker_class", CHUNKERS)
def test_varied_whitespace_unicode_and_budgets_preserve_coverage(chunker_class):
    randomizer = random.Random(42)
    for _ in range(100):
        text = "".join(randomizer.choice(["é", "🙂", "word.", "\r\n", "\n\n", " ", "終。", "?"])
                       for _ in range(50))
        size = randomizer.randrange(1, 31)
        overlap = randomizer.randrange(size)
        assert_complete_source_coverage(chunker_class(size, overlap), text)
