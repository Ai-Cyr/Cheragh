"""Behavioral regressions for the grouped retriever and long reader in LongRAG."""
from __future__ import annotations

import numpy as np
import pytest

from cheragh.base import Document, EmbeddingModel, HashingEmbedding, LLMClient
from cheragh.long_rag import LongRAGEngine, LongRAGRetriever, group_documents
from cheragh.security.access_control import AccessDecision, AccessPolicy, Principal


class TableEmbedding(EmbeddingModel):
    def __init__(self, table=None):
        self.table = table or {}
        self.texts = []
        self.queries = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return np.asarray([self.table.get(text.strip(), [1.0, 0.0]) for text in texts])

    def embed_query(self, text):
        self.queries.append(text)
        return np.array([1.0, 0.0])


class ScriptedReader(LLMClient):
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return next(self.answers)


def test_grouping_processes_degree_then_smallest_related_groups_and_keeps_every_source():
    docs = [Document(source, doc_id=source) for source in "abcdef"]
    # a,b form a two-document group; c remains alone. d links both groups,
    # and must absorb c first despite a being earlier lexicographically.
    adjacency = {"b": ["a"], "d": ["a", "c"], "e": ["a", "b", "c"], "f": []}
    groups = group_documents(docs, adjacency, max_group_size=3)
    assert [[doc.doc_id for doc in group] for group in groups] == [["a", "b", "e"], ["c", "d"], ["f"]]
    assert sorted(doc.doc_id for group in groups for doc in group) == list("abcdef")
    assert [[doc.doc_id for doc in group] for group in group_documents(list(reversed(docs)), adjacency, max_group_size=3)] == [
        [doc.doc_id for doc in group] for group in groups
    ]


def test_retrieval_uses_raw_dot_products_and_all_chunks_not_cosine_or_a_candidate_prefix():
    encoder = TableEmbedding({"ordinary": [1, 0], "needle": [8, 100], "competitor": [7, 0]})
    sources = [Document("ordinary " * 25 + "needle", doc_id="a"), Document("linked detail", doc_id="b"),
               Document("competitor", doc_id="c")]
    retriever = LongRAGRetriever(sources, encoder, adjacency={"b": ["a"]}, max_group_size=2,
                                chunk_tokens=1, token_counter=lambda text: len(text.split()) or 1,
                                embedding_batch_size=3)
    group = retriever.retrieve_groups("semantic query", top_k=1)[0]
    assert group.source_doc_ids == ("a", "b")
    assert group.score == 8
    assert group.matched_source_doc_id == "a"
    assert sources[0].content[group.matched_chunk_start:group.matched_chunk_end] == "needle"
    assert len(encoder.texts) > 25
    assert group.documents[0].content == sources[0].content
    standard = retriever.retrieve("semantic query", top_k=1)[0]
    assert sources[0].content in standard.content
    assert sources[1].content in standard.content
    assert standard.metadata["source_doc_ids"] == ["a", "b"]


def test_chunking_is_lossless_and_encoder_inputs_obey_budget():
    text = "Éléphants 🐘 vivent\nensemble.\n\n" * 4
    encoder = TableEmbedding()
    retriever = LongRAGRetriever([Document(text, doc_id="a")], encoder, chunk_tokens=12)
    assert "".join(encoder.texts) == text
    assert all(0 < len(chunk.encode()) <= 12 for chunk in encoder.texts)
    assert retriever.retrieve_groups("habitat")[0].documents[0].content == text


def test_tokenizers_that_do_not_encode_whitespace_are_supported():
    text = " " * 100 + "first " + "\n" * 100 + "last"
    encoder = TableEmbedding()
    LongRAGRetriever([Document(text, doc_id="a")], encoder, chunk_tokens=1,
                     token_counter=lambda text: len(text.split()))
    assert "".join(encoder.texts) == text


def test_sources_embeddings_and_outputs_are_snapshots():
    document = Document("original", doc_id="a", metadata={"nested": {"value": "original"}})
    retriever = LongRAGRetriever([document], TableEmbedding())
    document.content = "changed"
    document.metadata["nested"]["value"] = "changed"
    first = retriever.retrieve_groups("query")[0]
    first.documents[0].content = "changed again"
    first.documents[0].metadata["nested"]["value"] = "changed again"
    second = retriever.retrieve_groups("query")[0]
    assert second.documents[0].content == "original"
    assert second.documents[0].metadata["nested"]["value"] == "original"


def test_acl_precedes_grouping_and_private_scores_cannot_promote_public_group():
    sources = [Document("public", doc_id="a", metadata={"tenant_id": "alpha"}),
               Document("secret", doc_id="b", metadata={"tenant_id": "beta"}),
               Document("better", doc_id="c", metadata={"tenant_id": "alpha"})]
    encoder = TableEmbedding({"public": [1, 0], "secret": [100, 0], "better": [2, 0]})
    retriever = LongRAGRetriever(sources, encoder, adjacency={"b": ["a"]})
    assert retriever.retrieve_groups("query", top_k=1)[0].source_doc_ids == ("a", "b")
    groups = retriever.retrieve_groups("query", top_k=3, principal=Principal("alice", tenant_ids={"alpha"}))
    assert [group.source_doc_ids for group in groups] == [("c",), ("a",)]
    assert [group.score for group in groups] == [2, 1]
    # Filtering a private neighbor also changes the induced degree ordering.
    assert "secret" not in repr(groups)


def test_selected_tenant_collection_scope_applies_even_to_admin():
    docs = [Document("one", doc_id="a", metadata={"tenant_id": "alpha", "collection_id": "one"}),
            Document("two", doc_id="b", metadata={"tenant_id": "alpha", "collection_id": "two"}),
            Document("other", doc_id="c", metadata={"tenant_id": "beta", "collection_id": "one"})]
    retriever = LongRAGRetriever(docs, TableEmbedding(), adjacency={"c": ["a", "b"]})
    selected = retriever.retrieve_groups("query", principal=Principal("admin", roles={"admin"}),
                                         tenant_id="alpha", collection_id="one")
    assert selected[0].source_doc_ids == ("a",)
    assert retriever.retrieve_groups("query", allowed_doc_ids=[]) == []


def test_custom_policy_is_preserved_and_cannot_mutate_index():
    class CustomPolicy(AccessPolicy):
        def authorize(self, document, principal=None):
            document.content = "tampered"
            return AccessDecision(document.metadata["custom"])

    retriever = LongRAGRetriever([Document("allowed", doc_id="a", metadata={"custom": True}),
                                 Document("denied", doc_id="b", metadata={"custom": False})], TableEmbedding())
    assert retriever.retrieve_groups("query", access_policy=CustomPolicy())[0].documents[0].content == "allowed"
    assert len(retriever.retrieve_groups("query")) == 2


def test_reader_receives_complete_original_sources_and_preserves_citations():
    sources = [Document("A " * 200 + "important ending", doc_id="a", metadata={"page": 3}),
               Document("Complementary evidence", doc_id="b")]
    retriever = LongRAGRetriever(sources, HashingEmbedding(), adjacency={"b": ["a"]})
    reader = ScriptedReader("Supported conclusion [source: a] [source: b].")
    response = LongRAGEngine(retriever, reader).ask("What is the conclusion?", top_k=1)
    assert sources[0].content in reader.calls[0][0]
    assert sources[1].content in reader.calls[0][0]
    assert [source.doc_id for source in response.sources] == ["a", "b"]
    assert response.sources[0].location == "page=3"
    assert response.citation_validation.ok
    assert response.metadata["group_source_doc_ids"] == [["a", "b"]]
    assert response.trace.metadata["generation_calls"] == 1


def test_reader_budget_raises_before_generation_or_excludes_whole_group_with_diagnostics():
    sources = [Document("big " * 200, doc_id="a"), Document("linked", doc_id="b"), Document("small", doc_id="c")]
    retriever = LongRAGRetriever(sources, TableEmbedding(), adjacency={"b": ["a"]}, chunk_tokens=1024)
    reader = ScriptedReader("Answer [source: c].")
    with pytest.raises(ValueError, match="complete LongRAG group"):
        LongRAGEngine(retriever, reader, max_input_tokens=600).ask("query", top_k=2)
    assert reader.calls == []
    result = LongRAGEngine(retriever, reader, max_input_tokens=600, budget_policy="skip_groups").ask("query", top_k=2)
    assert result.metadata["excluded_source_doc_ids"] == ["a", "b"]
    assert result.metadata["group_source_doc_ids"] == [["c"]]
    assert "long_rag_budget_excluded_whole_groups" in result.warnings
    assert "linked" not in reader.calls[0][0]
    assert len(reader.calls[0][0].encode()) <= 600


def test_two_stage_reader_validates_draft_then_extracts_short_cited_answer():
    retriever = LongRAGRetriever([Document("Paris is the capital of France.", doc_id="a")], TableEmbedding())
    reader = ScriptedReader("The source establishes Paris as France's capital [source: a].", "Paris [source: a].")
    response = LongRAGEngine(retriever, reader, short_answer=True, short_answer_max_tokens=80).ask("Capital?")
    assert response.answer == "Paris [source: a]."
    assert len(reader.calls) == 2
    assert "The source establishes Paris" in reader.calls[1][0]
    assert reader.calls[1][1]["max_tokens"] == 80
    assert response.metadata["generation_calls"] == 2


@pytest.mark.parametrize("answers", [("Invented [source: unknown].",),
                                     ("Supported [source: a].", "Invented [source: unknown].")])
def test_invalid_draft_or_short_answer_citations_are_withheld(answers):
    retriever = LongRAGRetriever([Document("Evidence", doc_id="a")], TableEmbedding())
    reader = ScriptedReader(*answers)
    response = LongRAGEngine(retriever, reader, short_answer=True).ask("query")
    assert response.answer.startswith("Je ne sais pas")
    assert "Invented" not in response.answer
    assert response.citations == []
    assert not response.citation_validation.ok
    assert len(reader.calls) == len(answers)


def test_no_authorized_context_makes_no_model_call():
    encoder = TableEmbedding()
    retriever = LongRAGRetriever([Document("secret", doc_id="a")], encoder)
    reader = ScriptedReader()
    result = LongRAGEngine(retriever, reader).ask("query", allowed_doc_ids=[])
    assert encoder.queries == []
    assert reader.calls == []
    assert result.sources == []
    assert result.metadata["access_control_enabled"]


@pytest.mark.parametrize("vectors", [np.ones((1, 0)), np.ones(3), np.array([[np.nan, 0]]),
                                    np.array([[True, False]]), np.array([[1 + 2j, 0]])])
def test_invalid_embedding_matrices_fail_closed(vectors):
    class BrokenEncoder(TableEmbedding):
        def embed_documents(self, texts):
            return vectors
    with pytest.raises(ValueError):
        LongRAGRetriever([Document("source", doc_id="a")], BrokenEncoder())


@pytest.mark.parametrize("vectors", [np.ones((1, 2)), np.ones(3), np.array([np.inf, 0]), np.array([True, False])])
def test_invalid_query_vectors_fail_closed(vectors):
    class BrokenEncoder(TableEmbedding):
        def embed_query(self, text):
            return vectors
    retriever = LongRAGRetriever([Document("source", doc_id="a")], BrokenEncoder())
    with pytest.raises(ValueError):
        retriever.retrieve("query")


def test_ids_adjacency_and_token_budgets_are_strict():
    with pytest.raises(ValueError, match="source IDs"):
        group_documents([Document("source")], {})
    with pytest.raises(ValueError, match="duplicate"):
        group_documents([Document("one", doc_id="a"), Document("two", doc_id="a")], {})
    with pytest.raises(ValueError, match="unknown"):
        group_documents([Document("one", doc_id="a")], {"a": ["missing"]})
    with pytest.raises(TypeError, match="not strings"):
        group_documents([Document("one", doc_id="a")], {"a": "a"})
    with pytest.raises(ValueError, match="positive"):
        LongRAGRetriever([Document("one", doc_id="a")], TableEmbedding(), token_counter=lambda _: 0)
    assert LongRAGRetriever([], TableEmbedding()).retrieve("query") == []


def test_reader_enforces_output_and_refinement_input_budgets():
    retriever = LongRAGRetriever([Document("source", doc_id="a")], TableEmbedding())
    with pytest.raises(ValueError, match="output budget"):
        LongRAGEngine(retriever, ScriptedReader("very long answer [source: a]"), max_output_tokens=5).ask("query")
    # The first prompt fits, but a long grounded draft cannot bypass the same
    # explicit input ceiling when supplied to the second reader call.
    reader = ScriptedReader("draft " * 100 + "[source: a]")
    with pytest.raises(ValueError, match="prompt exceeds"):
        LongRAGEngine(retriever, reader, max_input_tokens=600, short_answer=True).ask("query")
    assert len(reader.calls) == 1
