"""Authorization and coherent index snapshots for production requests."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from cheragh.base import Document
from cheragh.community_graph.engine import CommunityGraphRAGEngine
from cheragh.graph.engine import GraphRAGEngine, KnowledgeGraph, KnowledgeTriple
from cheragh.multimodal import CallableMultimodalEmbedding, MultimodalDocument, MultimodalRetriever
from cheragh.multimodal.colpali import CallableVisualLateInteractionEncoder, ColPaliRetriever
from cheragh.retrieval.learned import ColBERTRetriever, LearnedSparseRetriever
from cheragh.security.access_control import AccessControlledRAGEngine, AccessDecision, AccessPolicy, Principal


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        assert "SECRET" not in prompt
        return "Public evidence [source: public]"


class SourcePolicy(AccessPolicy):
    def __bool__(self):
        return False

    def authorize(self, document, principal=None):
        allowed = document.doc_id == "public"
        document.content = "MUTATED BY POLICY"
        return AccessDecision(allowed)


def graph_fixture():
    documents = [Document("Public Alice evidence", doc_id="public"),
                 Document("SECRET Alice evidence", doc_id="private")]
    graph = KnowledgeGraph()
    graph.add_triple(KnowledgeTriple("Alice", "knows", "Bob", "public"))
    graph.add_triple(KnowledgeTriple("Alice", "SECRET relationship", "Bob", "private"))
    return documents, graph


@pytest.mark.parametrize("mode", ["global", "local"])
@pytest.mark.parametrize("scope", [{"allowed_doc_ids": ["public"]}, {"access_policy": SourcePolicy()}])
def test_community_lexical_queries_apply_scopes_before_generation(mode, scope):
    documents, graph = graph_fixture()
    llm = RecordingLLM()
    engine = CommunityGraphRAGEngine(documents, graph=graph, llm_client=llm)
    response = engine.ask("Alice", mode=mode, **scope)
    assert "SECRET" not in repr(response.to_dict(include_prompt=True))
    assert "MUTATED" not in repr(response.to_dict(include_prompt=True))
    if mode == "global":
        # The single report combines both sources and must be withheld entirely.
        assert response.sources == []
        assert llm.calls == []
    else:
        assert [source.doc_id for source in response.sources] == ["public"]
        assert llm.calls[0][1] == {}
    assert "SECRET" in repr(engine.documents)  # Request did not mutate the corpus.


@pytest.mark.parametrize("method", ["global_search", "local_search", "search", "retrieve"])
def test_community_empty_scope_is_fail_closed(method):
    documents, graph = graph_fixture()
    engine = CommunityGraphRAGEngine(documents, graph=graph)
    assert getattr(engine, method)("Alice", allowed_doc_ids=[]) == []
    with pytest.raises(TypeError, match="iterable"):
        getattr(engine, method)("Alice", allowed_doc_ids="public")


def test_graph_engine_uses_wrapped_retriever_and_does_not_leak_graph_metadata():
    documents, graph = graph_fixture()
    llm = RecordingLLM()
    engine = GraphRAGEngine(documents, graph=graph, llm_client=llm)
    wrapped = AccessControlledRAGEngine(engine, policy=SourcePolicy())
    response = wrapped.ask("Alice", principal=Principal("alice"))
    assert [source.doc_id for source in response.sources] == ["public"]
    assert "SECRET" not in repr(response.to_dict(include_prompt=True))
    assert {triple["doc_id"] for triple in response.metadata["graph_triples"]} == {"public"}
    assert llm.calls[0][1] == {}


def test_multimodal_encoder_and_result_mutations_cannot_change_index_metadata():
    def encode(documents):
        for doc in documents:
            doc.content = "CORRUPTED"
            doc.metadata["tenant"]["id"] = "other"
        return [[1.0, 0.0] for _ in documents]

    retriever = MultimodalRetriever(
        [MultimodalDocument("original", metadata={"tenant": {"id": "one"}}, doc_id="public")],
        CallableMultimodalEmbedding(encode, lambda query: [1.0, 0.0]),
    )
    result = retriever.retrieve("query")[0]
    assert result.content == "original"
    assert result.metadata["tenant"]["id"] == "one"
    result.metadata["tenant"]["id"] = "other"
    assert retriever.retrieve("query")[0].metadata["tenant"]["id"] == "one"


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -1.0])
def test_multimodal_rejects_invalid_weights(weight):
    with pytest.raises(ValueError, match="modality_weights"):
        MultimodalRetriever([], CallableMultimodalEmbedding(lambda docs: [], lambda query: []),
                            modality_weights={"text": weight})


@pytest.mark.parametrize("kind", ["sparse", "colbert", "colpali", "multimodal"])
def test_retrieval_uses_one_index_snapshot_during_concurrent_ingestion(kind):
    query_started = Event()
    finish_query = Event()

    def query_vectors(texts):
        query_started.set()
        assert finish_query.wait(5)
        return [[[1.0, 0.0]] for _ in texts]

    if kind in {"sparse", "colbert"}:
        class Encoder:
            def encode_documents(self, texts):
                vectors = [[1.0, 0.0] for _ in texts]
                return vectors if kind == "sparse" else [[vector] for vector in vectors]

            def encode_queries(self, texts):
                vectors = query_vectors(texts)
                return [vector[0] for vector in vectors] if kind == "sparse" else vectors

        cls = LearnedSparseRetriever if kind == "sparse" else ColBERTRetriever
        retriever = cls([Document("first", doc_id="first")], Encoder())
        def add():
            retriever.add_documents([Document("second", doc_id="second")])
    elif kind == "colpali":
        encoder = CallableVisualLateInteractionEncoder(lambda pages: [[[1.0, 0.0]] for _ in pages], query_vectors)
        retriever = ColPaliRetriever([MultimodalDocument("first", doc_id="first", modality="image")], encoder)
        def add():
            retriever.add_pages([MultimodalDocument("second", doc_id="second", modality="image")])
    else:
        encoder = CallableMultimodalEmbedding(
            lambda docs: [[1.0, 0.0] for _ in docs], lambda query: query_vectors([query])[0][0],
        )
        retriever = MultimodalRetriever([MultimodalDocument("first", doc_id="first")], encoder)
        def add():
            retriever.add_documents([MultimodalDocument("second", doc_id="second")])
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(retriever.retrieve, "query", 10)
        try:
            assert query_started.wait(5)
            add()
        finally:
            finish_query.set()
        assert [doc.doc_id for doc in future.result(timeout=5)] == ["first"]
    assert {doc.doc_id for doc in retriever.retrieve("query", 10)} == {"first", "second"}


def test_legacy_raptor_snapshots_inputs_and_returns_independent_metadata():
    from cheragh import HashingEmbedding, RAPTORRetriever, StaticLLMClient

    source = Document("Public original", doc_id="public", metadata={"nested": {"value": "original"}})
    retriever = RAPTORRetriever([source], HashingEmbedding(), StaticLLMClient("unused"), n_levels=0)
    assert "raptor_level" not in source.metadata
    source.content = "CHANGED"
    source.metadata["nested"]["value"] = "CHANGED"
    result = retriever.retrieve("Public")[0]
    assert result.content == "Public original"
    assert result.metadata["nested"]["value"] == "original"
    result.metadata["nested"]["value"] = "CHANGED"
    assert retriever.retrieve("Public")[0].metadata["nested"]["value"] == "original"


def test_legacy_raptor_authorizes_all_summary_dependencies(monkeypatch):
    import sys
    from types import ModuleType

    from cheragh import HashingEmbedding, RAPTORRetriever, StaticLLMClient
    from cheragh.security.access_control import AccessControlledRetriever

    class KMeans:
        def __init__(self, **kwargs):
            pass

        def fit(self, embeddings):
            self.labels_ = [0] * len(embeddings)
            return self

    cluster = ModuleType("sklearn.cluster")
    cluster.KMeans = KMeans
    monkeypatch.setitem(sys.modules, "sklearn.cluster", cluster)
    documents = [Document("Public", doc_id="public"), Document("SECRET", doc_id="private")]
    retriever = RAPTORRetriever(documents, HashingEmbedding(), StaticLLMClient("SECRET mixed summary"),
                                min_cluster_size=1, branching_factor=1, n_levels=1)
    assert retriever.all_nodes[-1].metadata["source_doc_ids"] == ["private", "public"]
    scoped = AccessControlledRetriever(retriever, Principal("alice"), policy=SourcePolicy())
    results = scoped.retrieve("Public", top_k=10)
    assert [document.doc_id for document in results] == ["public"]
    assert "SECRET" not in repr(results)
    assert len(retriever.retrieve("Public", top_k=10)) == 3


def test_legacy_raptor_empty_corpus_does_not_call_providers():
    from cheragh import RAPTORRetriever

    class Unused:
        def embed_documents(self, texts):
            raise AssertionError("empty corpus must not call the provider")

        def embed_query(self, text):
            raise AssertionError("empty corpus must not call the provider")

    assert RAPTORRetriever([], Unused(), object()).retrieve("query") == []


def test_graph_results_cannot_mutate_stored_access_rules():
    graph = KnowledgeGraph()
    graph.add_triple(KnowledgeTriple("Alice", "knows", "Bob", "protected"))
    engine = GraphRAGEngine(
        [Document("protected payload", doc_id="protected", metadata={"allowed_users": ["alice"]})],
        graph=graph,
    )
    wrapped = AccessControlledRAGEngine(engine, policy=AccessPolicy(require_tenant_match=False))
    assert wrapped.ask("Alice", principal=Principal("mallory")).sources == []
    engine.retrieve("Alice")[0].metadata["allowed_users"].append("mallory")
    assert wrapped.ask("Alice", principal=Principal("mallory")).sources == []


def test_graph_identity_collisions_cannot_merge_private_provenance():
    with pytest.raises(ValueError, match="unique"):
        GraphRAGEngine([Document("Public", doc_id="same"), Document("SECRET", doc_id="same")])
    engine = GraphRAGEngine([Document("Alice knows Bob."), Document("Carol knows Dave.", doc_id="doc-0")])
    assert len({doc.doc_id for doc in engine.documents}) == 2
    assert len(engine.retrieve("Alice", top_k=10)) == 2
