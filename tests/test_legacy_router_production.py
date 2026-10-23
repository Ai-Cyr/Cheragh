"""Bounded, isolated legacy routing and weighted ensemble retrieval."""
import math

import pytest

from cheragh import Document, StaticLLMClient
from cheragh.router import EnsembleRetriever, QueryRouter


class SharedRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.documents


class BoundedGeneratorRetriever:
    def retrieve(self, query, top_k=5):
        for index in range(top_k):
            yield Document(f"source {index}", doc_id=str(index))
        raise AssertionError("The provider iterator was consumed beyond the requested budget")


def test_router_bounds_generator_and_does_not_mutate_shared_documents():
    bounded = QueryRouter({"safe": BoundedGeneratorRetriever()}, {"safe": "safe"}, StaticLLMClient("safe"))
    assert len(bounded.retrieve("question", top_k=2)) == 2
    source = Document("source", doc_id="source", metadata={"allowed_users": ["alice"]})
    shared = SharedRetriever([source])
    router = QueryRouter({"safe": shared}, {"safe": "safe"}, StaticLLMClient("safe"))
    result = router.retrieve("question")[0]
    result.metadata["allowed_users"].append("mallory")
    assert result.metadata["route_used"] == "safe"
    assert source.metadata == {"allowed_users": ["alice"]}


def test_router_uses_exact_route_or_configured_fallback_and_copies_configuration():
    safe = SharedRetriever([Document("safe", doc_id="safe")])
    unrelated = SharedRetriever([Document("private", doc_id="private")])
    routes = {"hr": unrelated, "general": safe}
    descriptions = {"hr": "HR", "general": "general"}
    router = QueryRouter(routes, descriptions, StaticLLMClient("through general information"), default_route="general")
    routes["general"] = unrelated
    descriptions["general"] = "mutated"
    assert router.retrieve("question")[0].doc_id == "safe"
    assert unrelated.calls == []
    assert router.route_descriptions["general"] == "general"


def test_ensemble_bounds_generator_and_counts_a_document_once_per_member():
    source = Document("one", doc_id="one", metadata={"allowed_users": ["alice"]})
    first = SharedRetriever([source, source, Document("two", doc_id="two")])
    second = SharedRetriever([source])
    ensemble = EnsembleRetriever([first, second], weights=[2.0, 3.0], rrf_k=60, per_retriever_top_k=3)
    result = ensemble.retrieve("question")[0]
    assert result.doc_id == "one"
    assert result.score == pytest.approx(5.0 / 61)
    assert result.metadata["ensemble_sources"] == [0, 1]
    result.metadata["allowed_users"].append("mallory")
    assert source.metadata == {"allowed_users": ["alice"]}
    bounded = EnsembleRetriever([BoundedGeneratorRetriever()], per_retriever_top_k=2)
    assert len(bounded.retrieve("question", top_k=5)) == 2


def test_ensemble_copies_configuration_and_uses_stable_anonymous_identity():
    shared = SharedRetriever([Document("same content"), Document("same content")])
    retrievers, weights = [shared], [1.0]
    ensemble = EnsembleRetriever(retrievers, weights=weights)
    retrievers.clear()
    weights[0] = math.nan
    first = ensemble.retrieve("question")
    second = EnsembleRetriever([shared]).retrieve("question")
    assert len(first) == 1
    assert first[0].doc_id == second[0].doc_id
    assert first[0].score == pytest.approx(1 / 61)


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, -1.0, True, "1"])
def test_ensemble_rejects_invalid_weights(weight):
    with pytest.raises((TypeError, ValueError), match="weight"):
        EnsembleRetriever([SharedRetriever([])], weights=[weight])


@pytest.mark.parametrize("name", ["rrf_k", "per_retriever_top_k"])
@pytest.mark.parametrize("value", [-1, 0, True, math.nan, 1.5])
def test_ensemble_rejects_invalid_budgets_before_provider_calls(name, value):
    with pytest.raises((TypeError, ValueError)):
        EnsembleRetriever([SharedRetriever([])], **{name: value})


@pytest.mark.parametrize("kind", ["router", "ensemble"])
def test_legacy_orchestration_rejects_invalid_provider_documents(kind):
    provider = SharedRetriever([Document("text", score=math.nan)])
    retriever = (QueryRouter({"safe": provider}, {"safe": "safe"}, StaticLLMClient("safe"))
                 if kind == "router" else EnsembleRetriever([provider]))
    with pytest.raises(ValueError, match="finite"):
        retriever.retrieve("question")
    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve("  ")


def test_router_rejects_ambiguous_names_and_invalid_provider_output():
    provider = SharedRetriever([])
    with pytest.raises(ValueError, match="unique"):
        QueryRouter({"HR": provider, "hr": provider}, {"HR": "one", "hr": "two"}, StaticLLMClient("HR"))

    class InvalidLLM:
        def generate(self, prompt):
            return {"route": "safe"}

    router = QueryRouter({"safe": provider}, {"safe": "safe"}, InvalidLLM())
    with pytest.raises(TypeError, match="string"):
        router.retrieve("question")
    assert provider.calls == []
