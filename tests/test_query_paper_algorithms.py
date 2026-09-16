"""Numerical regressions for HyQE (Eq. 2/5), HyDE (Eq. 7/8), and RRF."""
from __future__ import annotations

import numpy as np
import pytest

from cheragh.base import Document, EmbeddingModel, LLMClient
from cheragh.cache import MemoryCache
from cheragh.hyde import HyDERetriever
from cheragh.hyqe import HyQEReranker, HyQERetriever
from cheragh.rag_fusion import RAGFusionRetriever


class Embeddings(EmbeddingModel):
    def __init__(self, documents, queries):
        self.documents = documents
        self.queries = queries
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return np.asarray([self.documents[text] for text in texts], dtype=float)

    def embed_query(self, text):
        self.query_calls.append(text)
        return np.asarray(self.queries[text], dtype=float)


class Generator(LLMClient):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return next(self.responses)


def test_hyqe_adds_context_cosine_and_question_cosine_instead_of_max_over_index():
    # Old max(original, hypothesis) ranks A first; Eq. 2 ranks B first.
    embedding = Embeddings({"A": [0, 5], "B": [4, 3]}, {
        "query": [10, 0], "Question for A?": [7, 0], "Question for B?": [3, 4],
    })
    retriever = HyQERetriever([Document("A", doc_id="a"), Document("B", doc_id="b")],
                            embedding, Generator(["Question for A?", "Question for B?"]), question_weight=0.5)
    results = retriever.retrieve("query")
    assert [doc.doc_id for doc in results] == ["b", "a"]
    assert [doc.score for doc in results] == pytest.approx([1.1, 0.5])
    assert results[0].metadata["hyqe_context_score"] == pytest.approx(0.8)
    assert "Question for B?" in embedding.query_calls
    assert all("Question for B?" not in texts for texts in embedding.document_calls)


@pytest.mark.parametrize("aggregation,expected", [("max", 1.5), ("mean", 1.0)])
def test_hyqe_paper_max_and_mean_aggregations(aggregation, expected):
    reranker = HyQEReranker(Embeddings({"context": [1, 0]}, {
        "query": [1, 0], "Positive question?": [1, 0], "Negative question?": [-1, 0],
    }), Generator(["Positive question?\nNegative question?"]), aggregation=aggregation)
    assert reranker.rerank("query", [Document("context")])[0].score == pytest.approx(expected)


def test_hyqe_cache_reuses_questions_for_different_queries_and_preserves_candidate_metadata():
    llm = Generator(["Cached question?"])
    reranker = HyQEReranker(Embeddings({"context": [1, 0]}, {
        "query one": [1, 0], "query two": [0, 1], "Cached question?": [1, 0],
    }), llm, cache=MemoryCache())
    first = reranker.rerank("query one", [Document("context", doc_id="first", metadata={"tenant_id": "a"})])
    second = reranker.rerank("query two", [Document("context", doc_id="second", metadata={"tenant_id": "b"})])
    assert len(llm.calls) == 1
    assert first[0].doc_id == "first"
    assert second[0].doc_id == "second"
    assert second[0].metadata["tenant_id"] == "b"


def test_hyqe_long_context_questions_cover_every_partition():
    content = "A" * 20 + "B" * 20 + "Tail fact."
    llm = Generator(["First question?", "Second question?", "Last question?"])
    embedding = Embeddings({content: [1, 0]}, {
        "query": [1, 0], "First question?": [1, 0], "Second question?": [0, 1], "Last question?": [1, 0],
    })
    reranker = HyQEReranker(embedding, llm, max_context_chars=20, n_questions_per_doc=1)
    result = reranker.rerank("query", [Document(content)])[0]
    assert result.metadata["hyqe_question_count"] == 3
    assert "Tail fact." in llm.calls[-1][0]
    assert len(llm.calls) == 3


def test_hyqe_scores_all_sources_even_if_one_source_has_many_questions():
    questions = [f"First question {i}?" for i in range(25)]
    embedding = Embeddings({"A": [1, 0], "B": [0, 1]}, {
        "query": [1, 0], **{question: [1, 0] for question in questions}, "Other question?": [0, 1],
    })
    retriever = HyQERetriever([Document("A"), Document("B")], embedding,
                            Generator(["\n".join(questions), "Other question?"]), n_questions_per_doc=25)
    assert len(retriever.retrieve("query", top_k=2)) == 2


def test_hyqe_missing_questions_keeps_original_similarity():
    reranker = HyQEReranker(Embeddings({"context": [3, 4]}, {"query": [1, 0]}), Generator([""]))
    document = Document("context", metadata={"nested": {"value": 1}})
    result = reranker.rerank("query", [document])[0]
    assert result.score == pytest.approx(0.6)
    result.metadata["nested"]["value"] = 2
    assert document.metadata["nested"]["value"] == 1


@pytest.mark.parametrize("values", [[float("nan"), 0], []])
def test_hyqe_rejects_invalid_embeddings(values):
    reranker = HyQEReranker(Embeddings({"context": values}, {}), Generator([""]))
    with pytest.raises(ValueError, match="finite embeddings"):
        reranker.rerank("query", [Document("context")])


def test_hyde_uses_arithmetic_mean_and_unnormalized_inner_product():
    embedding = Embeddings({"A": [10, 0], "B": [1, 1], "hyp one": [2, 0], "hyp two": [0, 4]}, {})
    llm = Generator(["hyp one", "hyp two"])
    retriever = HyDERetriever([Document("A"), Document("B")], embedding, llm, n_hypotheses=2)
    results = retriever.retrieve("query")
    assert [doc.score for doc in results] == pytest.approx([10, 3])
    assert all(kwargs == {"temperature": 0.7} for _, kwargs in llm.calls)
    assert embedding.query_calls == []


def test_hyde_query_inclusion_uses_document_encoder_and_cosine_is_explicit():
    embedding = Embeddings({"A": [10, 0], "B": [1, 1], "hyp": [2, 0], "query": [0, 4]}, {"query": [-1, -1]})
    retriever = HyDERetriever([Document("A"), Document("B")], embedding, Generator(["hyp"]),
                            include_query=True, similarity="cosine", generation_kwargs={"temperature": 0.2})
    results = retriever.retrieve("query")
    assert results[0].content == "B"
    assert results[0].score == pytest.approx(3 / np.sqrt(10))
    assert embedding.document_calls[-1] == ["hyp", "query"]
    assert embedding.query_calls == []


def test_hyde_empty_corpus_does_not_generate_hallucinations_or_embed():
    embedding, llm = Embeddings({}, {}), Generator([])
    assert HyDERetriever([], embedding, llm).retrieve("query") == []
    assert not llm.calls and not embedding.document_calls


@pytest.mark.parametrize("count", [True, 1.5, 0, -1])
def test_hyde_hypothesis_count_is_a_positive_integer(count):
    with pytest.raises((ValueError, TypeError)):
        HyDERetriever([], Embeddings({}, {}), Generator([]), n_hypotheses=count)


def test_rag_fusion_never_votes_twice_for_the_same_query_or_result():
    class Retriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k=5):
            self.calls.append(query)
            return [Document("A", doc_id="a"), Document("A", doc_id="a"), Document("B", doc_id="b")]

    base = Retriever()
    fusion = RAGFusionRetriever(base, Generator(["1. Original query\n2. New query\n3. new query"]), n_queries=4)
    results = fusion.retrieve("Original query")
    assert base.calls == ["Original query", "New query"]
    assert results[0].score == pytest.approx(2 / 61)
    assert results[0].metadata["rrf_sources"] == 2
