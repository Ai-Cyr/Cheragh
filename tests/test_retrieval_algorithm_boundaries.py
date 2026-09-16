from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cheragh import Document, HashingEmbedding, StaticLLMClient
from cheragh.base import cosine_similarity
from cheragh.contextual_compression import ContextualCompressionRetriever
from cheragh.conversation import ConversationalRAGEngine, ConversationTurn, InMemoryConversationStore
from cheragh.evaluation.retrieval import RetrievalExample, evaluate_retrieval
from cheragh.federated import FederatedRetriever
from cheragh.hybrid_search import BM25Retriever
from cheragh.ingestion.chunkers import RecursiveTextChunker, SemanticChunker
from cheragh.ingestion.chunkers.structured import HTMLSectionChunker
from cheragh.parent_document import ParentDocumentRetriever
from cheragh.retrieval.parent_child import ParentChildRetriever
from cheragh.reranking import CrossEncoderReranker
from cheragh.self_query import SelfQueryRetriever
from cheragh.tokenization import RetrievalTokenizer


class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append(top_k)
        return self.documents[:top_k]


@pytest.mark.parametrize("text", ["A" * 60 + " final fact.", "Repeated phrase. " * 15,
                                   "Line one.\r\n\r\nLine two.\r\nLast fact.", "词语" * 50])
def test_recursive_chunks_have_real_offsets_bounded_size_and_preserve_all_nonwhitespace(text):
    chunks = RecursiveTextChunker(chunk_size=30, chunk_overlap=7, min_chunk_size=20).split_text_with_offsets(text)
    covered = set()
    for chunk in chunks:
        assert len(chunk.text) <= 30
        source = text[chunk.source_char_start:chunk.source_char_end]
        assert " ".join(chunk.text.split()) == " ".join(source.split())
        covered.update(range(chunk.source_char_start, chunk.source_char_end))
    assert all(index in covered for index, char in enumerate(text) if not char.isspace())


def test_semantic_chunker_preserves_short_tail_and_original_offsets():
    text = "  Long animal sentence about cats.\r\n Another animal sentence.\n\nNo. "
    docs = SemanticChunker(HashingEmbedding(16), max_chunk_size=40, min_chunk_size=20).split_documents([Document(text, doc_id="a")])
    assert any("No." in document.content for document in docs)
    for document in docs:
        start, end = document.metadata["source_char_start"], document.metadata["source_char_end"]
        assert " ".join(text[start:end].split()) == " ".join(document.content.split())
        assert document.metadata["parent_doc_id"] == "a"
        assert len(document.content) <= 40


def test_semantic_chunker_can_embed_sentence_neighborhoods_and_use_percentiles():
    class Embeddings(HashingEmbedding):
        def embed_documents(self, texts):
            self.texts = texts
            return super().embed_documents(texts)

    embedding = Embeddings(16)
    SemanticChunker(embedding, buffer_size=1, breakpoint_percentile=90).split_text("First sentence. Second sentence. Third sentence.")
    assert embedding.texts == ["First sentence. Second sentence.",
                               "First sentence. Second sentence. Third sentence.", "Second sentence. Third sentence."]


def test_html_hierarchy_does_not_discard_preamble_before_first_heading():
    sections = HTMLSectionChunker().split_html("<p>Critical opening fact.</p><h1>Section</h1><p>Later.</p>")
    assert sections[0]["content"] == "Critical opening fact."


def test_cosine_and_mmr_inputs_do_not_require_normalized_embeddings():
    result = cosine_similarity(np.array([10.0, 0]), np.array([[100, 0], [3, 4], [0, 0], [-10, 0]]))
    assert result == pytest.approx([1, 0.6, 0, -1])
    assert cosine_similarity(np.array([1e300, 0]), np.array([[1e300, 1e300]]))[0] == pytest.approx(1 / math.sqrt(2))


@pytest.mark.parametrize("matrix", [[[float("nan"), 0]], [[1, 2, 3]]])
def test_cosine_rejects_malformed_provider_output(matrix):
    with pytest.raises(ValueError):
        cosine_similarity(np.array([1, 0]), np.array(matrix))


def test_bm25_positive_idf_and_term_frequency_have_analytical_scores():
    model = BM25Retriever([Document("alpha alpha beta", doc_id="a"), Document("beta", doc_id="b")], k1=1.5, b=0.75,
                          tokenizer=RetrievalTokenizer(ngram_range=(1, 1)))
    results = model.retrieve("alpha")
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    expected = idf * (2 * 2.5) / (2 + 1.5 * (0.25 + 0.75 * 3 / 2))
    assert results[0].score == pytest.approx(expected)
    assert results[1].score == 0
    # A term in every document must still have positive evidence weight.
    assert all(document.score > 0 for document in model.retrieve("beta"))


def test_parent_child_expands_candidates_until_distinct_parents_are_found():
    parents = [Document("Parent A", doc_id="a"), Document("Parent B", doc_id="b")]
    children = [Document(f"A fact {index}", doc_id=f"a-{index}", metadata={"parent_doc_id": "a"}, score=1)
                for index in range(30)] + [Document("B fact", doc_id="b-0", metadata={"parent_doc_id": "b"}, score=0.5)]
    source = Retriever(children)
    retriever = ParentChildRetriever(parents, children, child_retriever=source, top_k_children=4)
    assert [doc.doc_id for doc in retriever.retrieve("query", top_k=2)] == ["a", "b"]
    assert source.calls == [4, 8, 16, 32]


def test_legacy_parent_retriever_does_not_starve_other_parents_or_mutate_inputs():
    parents = [Document("alpha " * 100, doc_id="a"), Document("beta")]
    retriever = ParentDocumentRetriever(parents, HashingEmbedding(16), child_chunk_size=1, child_chunk_overlap=0)
    assert len(retriever.retrieve("alpha", top_k=2)) == 2
    assert parents[1].doc_id is None


@pytest.mark.parametrize("filters", [{"year": {"$unknown": 2024}}, {"unknown": 1}, {"year": {"$in": "2024"}}])
def test_self_query_rejects_uninterpretable_structured_constraints(filters):
    source = SelfQueryRetriever([Document("Fact", metadata={"year": 2024})], HashingEmbedding(16),
                                StaticLLMClient(json.dumps({"cleaned_query": "fact", "filters": filters})), {"year": "integer"})
    with pytest.raises(ValueError):
        source.retrieve("query")


def test_self_query_comparisons_do_not_crash_on_missing_or_mixed_types():
    assert not SelfQueryRetriever._match_filters({"year": "2024"}, {"year": {"$gte": 2023}})
    assert not SelfQueryRetriever._match_filters({}, {"year": {"$gte": 2023}})


def test_compression_rejects_fabricated_evidence_and_removed_negation():
    retriever = ContextualCompressionRetriever(Retriever([Document("This action is not permitted.")]),
                                               StaticLLMClient("This action is permitted."), min_compressed_length=0)
    with pytest.raises(ValueError, match="verbatim"):
        retriever.retrieve("Is this allowed?")


def test_conversation_can_rewrite_reference_without_retrieving_on_the_history_transcript():
    class Engine:
        def ask(self, query, **kwargs):
            return SimpleNamespace(answer="answer", sources=[], metadata={}, query=query)

    memory = InMemoryConversationStore()
    memory.append("s", ConversationTurn("Explain contract A", "It is a contract."))
    engine = ConversationalRAGEngine(Engine(), memory=memory, query_rewriter=StaticLLMClient("What is contract A's notice period?"))
    result = engine.ask("And its notice period?", session_id="s")
    assert result.query == "What is contract A's notice period?"
    assert memory.get("s", limit=0) == []
    assert engine.ask("unrelated", session_id="other").query == "unrelated"


def test_federation_does_not_compare_uncalibrated_source_scores():
    source = FederatedRetriever({"a": Retriever([Document("A1", score=1000), Document("A2", score=900)]),
                                 "b": Retriever([Document("B1", score=-10)])})
    assert [doc.content for doc in source.retrieve("query", top_k=2)] == ["A1", "B1"]


def test_retrieval_evaluation_uses_positive_graded_labels_and_enforces_depth():
    class Overfetch:
        def retrieve(self, query, top_k=5):
            return [Document("wrong", doc_id="x"), Document("right", doc_id="a")]

    result = evaluate_retrieval([RetrievalExample("query", set(), {"a": 2})], Overfetch(), top_k=1)
    assert result.metrics["recall@1"] == 0
    assert result.metrics["precision@1"] == 0


@pytest.mark.parametrize("scores", [[1.0], [1.0, float("nan")], [[1, 2], [3, 4]]])
def test_cross_encoder_requires_one_finite_score_per_document(scores):
    reranker = CrossEncoderReranker(model=SimpleNamespace(predict=lambda pairs: scores))
    with pytest.raises(ValueError, match="one finite"):
        reranker.rerank("query", [Document("a"), Document("b")])
