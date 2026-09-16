"""Cross-method regressions for derived evidence and authorization boundaries."""
import json

import numpy as np
import pytest

from cheragh.base import Document, EmbeddingModel
from cheragh.community_graph.engine import CommunityGraphRAGEngine
from cheragh.community_graph.local import LocalGraphSearchConfig
from cheragh.corrective.semantic import SemanticKnowledgeRefiner
from cheragh.graph.engine import KnowledgeGraph, KnowledgeTriple
from cheragh.raptor_engine import RAPTOREngine, RAPTORIndex, RAPTORNode, RAPTORRetrieverV2
from cheragh.security.access_control import AccessControlledRAGEngine, AccessControlledRetriever, AccessDecision, AccessPolicy, Principal


class ConstantEmbedding(EmbeddingModel):
    def embed_documents(self, texts):
        return np.array([[1., 0.] for _ in texts])

    def embed_query(self, text):
        return np.array([1., 0.])


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if "Résumé:" in prompt:
            return "Public and SECRET combined facts"
        assert "SECRET" not in prompt
        return "Public evidence [source: a]."


class SourcePolicy(AccessPolicy):
    def __bool__(self):
        return False  # Explicit policies must not be replaced because falsy.

    def authorize(self, document, principal=None):
        permitted = document.metadata.get("permit") is True
        document.content = "policy mutated its input"
        return AccessDecision(permitted)


def tree():
    return RAPTORIndex([
        RAPTORNode(Document("Public evidence", doc_id="a", metadata={"permit": True, "tenant_id": "alpha", "collection_id": "one"}), 0),
        RAPTORNode(Document("SECRET", doc_id="b", metadata={"permit": False, "tenant_id": "beta", "collection_id": "two"}), 0),
        RAPTORNode(Document("Public summary", doc_id="safe"), 1, ["a"]),
        RAPTORNode(Document("SECRET mixed summary", doc_id="mixed"), 1, ["a", "b"]),
        RAPTORNode(Document("SECRET root", doc_id="root"), 2, ["safe", "mixed"]),
        RAPTORNode(Document("SECRET unknown provenance", doc_id="orphan"), 1),
    ])


@pytest.mark.parametrize("mode", ["flat", "tree", "paper_tree"])
def test_raptor_authorizes_all_original_dependencies_and_removes_private_paths_before_scoring(mode):
    retriever = RAPTORRetrieverV2(tree(), ConstantEmbedding(), retrieval_mode=mode)
    sources = retriever.retrieve("query", top_k=10, allowed_doc_ids=["a"])
    assert sources
    assert {document.doc_id for document in sources} <= {"a", "safe"}
    assert "SECRET" not in repr(sources)
    for document in sources:
        assert document.metadata["source_doc_ids"] == ["a"]
        assert set(document.metadata["raptor_path"]) <= {"a", "safe"}
    # Filtering must not alter the shared index or a later unscoped request.
    unscoped = retriever.retrieve("query", top_k=10, retrieval_mode="flat")
    assert {document.doc_id for document in unscoped} == {"a", "b", "safe", "mixed", "root", "orphan"}
    root = next(document for document in unscoped if document.doc_id == "root")
    assert root.metadata["source_doc_ids"] == ["a", "b"]


def test_raptor_native_tenant_and_collection_scope_applies_even_to_admin():
    retriever = RAPTORRetrieverV2(tree(), ConstantEmbedding())
    selected = retriever.retrieve("query", top_k=10, principal=Principal("admin", roles={"admin"}),
                                  tenant_id="alpha", collection_id="one")
    assert {doc.doc_id for doc in selected} == {"a", "safe"}
    assert retriever.retrieve("query", allowed_doc_ids=[]) == []


def test_acl_wrapper_evaluates_source_policy_for_derived_summaries_and_retains_safe_summaries():
    retriever = RAPTORRetrieverV2(tree(), ConstantEmbedding())
    wrapped = AccessControlledRetriever(retriever, Principal("alice"), policy=SourcePolicy())
    docs = wrapped.retrieve("query", top_k=10)
    assert {document.doc_id for document in docs} == {"a", "safe"}
    assert all(document.metadata["source_doc_ids"] == ["a"] for document in docs)
    assert "policy mutated" not in repr(docs)
    assert "policy mutated" not in repr(retriever.retrieve("query", top_k=10))


@pytest.mark.parametrize("native", [False, True])
def test_raptor_engine_cannot_bypass_authorization_through_its_original_inner_engine(native):
    llm = RecordingLLM()
    engine = RAPTOREngine([
        Document("Public evidence", doc_id="a", metadata={"permit": True}),
        Document("SECRET", doc_id="b", metadata={"permit": False}),
    ], embedding_model=ConstantEmbedding(), llm_client=llm, levels=1, require_citations=True)
    llm.calls.clear()
    result = (engine.ask("query", access_policy=SourcePolicy()) if native else
              AccessControlledRAGEngine(engine, policy=SourcePolicy()).ask("query", principal=Principal("alice")))
    assert result.answer == "Public evidence [source: a]."
    assert [source.doc_id for source in result.sources] == ["a"]
    assert "SECRET" not in json.dumps(result.to_dict(include_prompt=True))
    assert "raptor_index" not in result.metadata
    assert all("access_policy" not in kwargs and "principal" not in kwargs for _, kwargs in llm.calls)


def test_raptor_enforces_summary_output_limit_and_rejects_zero_cost_counter():
    llm = RecordingLLM()
    with pytest.raises(ValueError, match="summary_max_tokens"):
        RAPTOREngine([Document("a"), Document("b")], llm_client=llm,
                     token_estimator=len, summary_max_tokens=5)
    with pytest.raises(ValueError, match="positive"):
        RAPTOREngine([Document("a"), Document("b")], token_estimator=lambda _: 0)


@pytest.mark.parametrize("dependency", ["private", "unknown-private"])
def test_graph_reports_propagate_all_consolidated_relationship_dependencies(dependency):
    graph = KnowledgeGraph()
    graph.add_triple(KnowledgeTriple("Alice", "SECRET consolidated relation", "Bob", "public",
                                     metadata={"source_doc_ids": ["public", dependency]}))
    docs = [Document("Public material", doc_id="public"), Document("SECRET evidence", doc_id="private")]
    llm = RecordingLLM()
    engine = CommunityGraphRAGEngine(docs, graph=graph, llm_client=llm)
    assert set(engine.reports[0].doc_ids) == {"public", dependency}
    result = engine.ask_global_map_reduce("Overview", allowed_doc_ids=["public"])
    assert result.sources == []
    assert result.metadata["map_calls"] == 0
    assert llm.calls == []
    assert "SECRET" not in json.dumps(result.to_dict(include_prompt=True))
    if dependency.startswith("unknown"):
        assert engine.reports[0].metadata["provenance_complete"] is False


@pytest.mark.parametrize("method", ["ask_local", "ask_global_map_reduce"])
def test_graph_explicit_falsy_policy_is_honored(method):
    graph = KnowledgeGraph()
    graph.add_triple(KnowledgeTriple("Alice", "knows", "Bob", "source"))
    docs = [Document("SECRET evidence", doc_id="source", metadata={"permit": False, "tenant_id": "alpha"})]
    llm = RecordingLLM()
    engine = CommunityGraphRAGEngine(docs, graph=graph, llm_client=llm, embedding_model=ConstantEmbedding(),
                                     local_search_config=LocalGraphSearchConfig())
    result = getattr(engine, method)("Alice", principal=Principal("alice", tenant_ids={"alpha"}), access_policy=SourcePolicy())
    assert result.sources == []
    assert llm.calls == []
    assert "SECRET" not in result.prompt


def test_semantic_refiner_uses_immutable_source_strips_despite_mutating_custom_grader():
    class MutatingGrader:
        def score_documents(self, query, strips):
            for strip in strips:
                strip.content = "FABRICATED"
                strip.metadata["crag_strip"]["start"] = 99
            return [1.] * len(strips)

    original = Document("First fact. Second fact.", doc_id="source")
    refined = SemanticKnowledgeRefiner(MutatingGrader(), sentences_per_strip=1).refine("query", [original])[0]
    assert refined.content == "First fact.\n\nSecond fact."
    spans = refined.metadata["corrective_provenance"]["refinement"]["retained_strips"]
    assert [original.content[span["start"]:span["end"]] for span in spans] == ["First fact.", "Second fact."]
