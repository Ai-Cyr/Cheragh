"""Regression coverage for bounded, isolated architecture extension points."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from cheragh import Document, HashingEmbedding, StaticLLMClient
from cheragh.adaptive import AdaptiveRAGRoute, HeuristicComplexityClassifier
from cheragh.agentic import AgentAction, AgentTool, AgenticRAGEngine, RetrievalToolAdapter, ScriptedPlanner, ToolRegistry
from cheragh.self_rag import (
    EvidenceRelevance,
    LexicalEvidenceCritic,
    RelevanceAssessment,
    RetrievalDecision,
    ScriptedEvidenceCritic,
    SelfRAGEngine,
    SupportAssessment,
)
from cheragh.temporal import TemporalRetriever
from cheragh.contextual_compression import ContextualCompressionRetriever
from cheragh.mmr import MMRRetriever


class GuardedRetriever:
    def __init__(self, document):
        self.document = document
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        for _ in range(top_k):
            yield self.document
        raise AssertionError("architecture consumed beyond its retrieval budget")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("constructor", [
    lambda value: RetrievalDecision(True, value),
    lambda value: EvidenceRelevance(0, "doc", value, True),
    lambda value: SupportAssessment(value, True),
    lambda value: LexicalEvidenceCritic(relevance_threshold=value),
])
def test_self_rag_rejects_non_finite_critic_confidence(constructor, value):
    with pytest.raises(ValueError, match="finite"):
        constructor(value)


@pytest.mark.parametrize("constructor", [
    lambda: RetrievalDecision("false"),
    lambda: EvidenceRelevance(0, "doc", 1.0, "false"),
    lambda: SupportAssessment(1.0, "false"),
    lambda: EvidenceRelevance(True, "doc", 1.0, True),
    lambda: EvidenceRelevance(0.5, "doc", 1.0, True),
])
def test_self_rag_rejects_ambiguous_critic_decisions(constructor):
    with pytest.raises(TypeError):
        constructor()


def test_self_rag_bounds_retrieval_and_isolates_critic_and_result_documents():
    source = Document("original evidence", metadata={"nested": {"value": "original"}}, doc_id="doc")
    retriever = GuardedRetriever(source)

    class MutatingCritic(ScriptedEvidenceCritic):
        def assess_relevance(self, query, documents):
            documents[0].content = "corrupted relevance evidence"
            documents[0].metadata["nested"]["value"] = "corrupted relevance"
            return super().assess_relevance(query, documents)

        def assess_support(self, query, answer, documents):
            assert documents[0].content == "original evidence"
            documents[0].content = "corrupted support evidence"
            documents[0].metadata["nested"]["value"] = "corrupted support"
            return super().assess_support(query, answer, documents)

    engine = SelfRAGEngine(retriever, StaticLLMClient("original evidence"), evidence_critic=MutatingCritic())
    result = engine.ask("evidence", top_k=1)
    assert result.trace.retrieved_count == 1
    assert result.documents[0].content == "original evidence"
    assert source.metadata["nested"]["value"] == "original"
    result.documents[0].metadata["nested"]["value"] = "changed by caller"
    assert source.metadata["nested"]["value"] == "original"


@pytest.mark.parametrize("grade", [EvidenceRelevance(1, "doc", 1.0, True), EvidenceRelevance(0, "other", 1.0, True)])
def test_self_rag_rejects_misattributed_relevance_before_generation(grade):
    class Critic(ScriptedEvidenceCritic):
        def assess_relevance(self, query, documents):
            return RelevanceAssessment((grade,))

    class UnusedLLM:
        def generate(self, prompt, **kwargs):
            raise AssertionError("invalid evidence was passed to generation")

    engine = SelfRAGEngine(GuardedRetriever(Document("evidence", doc_id="doc")), UnusedLLM(), evidence_critic=Critic())
    with pytest.raises(ValueError, match="assessment"):
        engine.ask("evidence", top_k=1)


def test_self_rag_does_not_turn_invalid_generator_output_into_an_answer():
    class InvalidLLM:
        def generate(self, prompt, **kwargs):
            return None

    engine = SelfRAGEngine(GuardedRetriever(Document("evidence")), InvalidLLM(), evidence_critic=ScriptedEvidenceCritic())
    with pytest.raises(TypeError, match="return a string"):
        engine.ask("evidence", top_k=1)


def test_relevance_assessment_rejects_conflicting_duplicate_indices():
    with pytest.raises(ValueError, match="duplicate"):
        RelevanceAssessment((EvidenceRelevance(0, "doc", 1.0, True), EvidenceRelevance(0, "doc", 0.0, False)))


def test_agentic_retrieval_adapter_enforces_budget_even_for_lazy_retrievers():
    adapter = RetrievalToolAdapter(GuardedRetriever(Document("evidence", doc_id="doc")), top_k=1)
    assert len(adapter("evidence")["documents"]) == 1


def test_scripted_agentic_planner_has_request_local_progress():
    planner = ScriptedPlanner([AgentAction.call_tool("lookup", "proof"), AgentAction.finish("answer")])
    engine = AgenticRAGEngine(planner, [AgentTool("lookup", "retrieve proof", lambda value: value)], max_steps=2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(engine.ask, ["one", "two", "three", "four"]))
    assert all(result.answer == "answer" and len(result.trace.steps) == 2 for result in results)
    assert all(result.trace.steps[0].observation.output == "proof" for result in results)


def test_agentic_rejects_non_text_tool_names_without_executing():
    with pytest.raises(TypeError, match="tool_name"):
        AgentAction("tool", tool_name=["lookup"])
    observation = ToolRegistry().execute(["lookup"], "input")
    assert observation.error == "tool_name_must_be_text"
    assert not observation.ok


@pytest.mark.parametrize("query", ["Bonjour, quel est le délai de remboursement ?", "Hello what is our retention policy?"])
def test_adaptive_greeting_prefix_does_not_skip_factual_retrieval(query):
    assert HeuristicComplexityClassifier().classify(query).route == AdaptiveRAGRoute.SINGLE_STEP


def test_temporal_retrieval_bounds_candidate_consumption():
    retriever = GuardedRetriever(Document("evidence", metadata={"timestamp": "2026-01-01T00:00:00Z"}, doc_id="doc"))
    temporal = TemporalRetriever(retriever, candidate_top_k=2)
    assert len(temporal.retrieve("evidence", top_k=1)) == 1
    assert retriever.calls == [("evidence", 2)]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "0.5"])
def test_temporal_retrieval_rejects_invalid_scores_before_ranking(score):
    retriever = GuardedRetriever(Document("evidence", metadata={"timestamp": "2026-01-01T00:00:00Z"}, score=score))
    with pytest.raises((TypeError, ValueError), match="score"):
        TemporalRetriever(retriever, candidate_top_k=1).retrieve("evidence", top_k=1)


def test_mmr_bounds_lazy_candidate_retrieval_before_embedding():
    retriever = GuardedRetriever(Document("evidence", doc_id="doc"))
    mmr = MMRRetriever(HashingEmbedding(), base_retriever=retriever, fetch_k=2)
    assert len(mmr.retrieve("evidence", top_k=1)) == 1
    assert retriever.calls == [("evidence", 2)]


def test_contextual_compression_bounds_generation_when_every_candidate_is_dropped():
    retriever = GuardedRetriever(Document("evidence", doc_id="doc"))

    class CountingLLM:
        calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            return "NO_OUTPUT"

    llm = CountingLLM()
    compressor = ContextualCompressionRetriever(retriever, llm)
    assert compressor.retrieve("evidence", top_k=2) == []
    assert llm.calls == 4
