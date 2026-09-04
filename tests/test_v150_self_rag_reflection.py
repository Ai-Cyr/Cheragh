"""Check paper calculations and model-probability trust boundaries."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from cheragh import Document, StaticLLMClient
from cheragh.self_rag import (
    ReflectionScore,
    ReflectionTokenDistribution as Distribution,
    ReflectionTokenGroup as Group,
    ReflectionTokenRetrievalGate,
    ReflectionTokenScorer,
    ScriptedEvidenceCritic,
    SelfRAGEngine,
)


def distributions():
    return (
        Distribution(Group.RELEVANCE, {"[Relevant]": 0.08, "[Irrelevant]": 0.02}, model_id="self-rag-test"),
        Distribution(Group.SUPPORT, {
            "[Fully supported]": 0.04, "[Partially supported]": 0.02, "[No support / Contradictory]": 0.04,
        }, model_id="self-rag-test"),
        Distribution(Group.UTILITY, {
            "[Utility:1]": 0.02, "[Utility:2]": 0.04, "[Utility:3]": 0.02,
            "[Utility:4]": 0.06, "[Utility:5]": 0.06,
        }, model_id="self-rag-test"),
    )


def test_paper_scores_normalize_each_token_group_independently():
    result = ReflectionTokenScorer().score(*distributions())

    assert result.relevance == pytest.approx(0.8)
    assert result.support == pytest.approx(0.5)
    assert result.utility == pytest.approx(0.25)
    assert result.total == pytest.approx(1.425)
    assert result.sequence_probability is None
    assert result.model_ids == ("self-rag-test",)


def test_custom_weights_and_optional_geometric_mean_sequence_probability():
    result = ReflectionTokenScorer(2.0, 3.0, 0.4).score(
        *distributions(), mean_sequence_logprob=math.log(0.5),
    )

    assert result.total == pytest.approx(3.7)
    assert result.sequence_probability == pytest.approx(0.5)


@pytest.mark.parametrize("level, expected", [(1, -1.0), (2, -0.5), (3, 0.0), (4, 0.5), (5, 1.0)])
def test_utility_uses_paper_signed_weights(level, expected):
    relevance, support, _ = distributions()
    utility = Distribution(Group.UTILITY, {
        f"[Utility:{index}]": float(index == level) for index in range(1, 6)
    })
    result = ReflectionTokenScorer(0, 0, 1).score(relevance, support, utility)
    assert result.utility == expected
    assert result.total == expected


def test_support_credits_partial_evidence_half_as_much_as_full():
    relevance, _, utility = distributions()
    partial = Distribution(Group.SUPPORT, {
        "[Fully supported]": 0, "[Partially supported]": 1, "[No support / Contradictory]": 0,
    })
    assert ReflectionTokenScorer().score(relevance, partial, utility).support == 0.5


def test_logprob_scoring_is_stable_far_below_underflow_range():
    source = distributions()
    logged = [Distribution.from_logprobs(
        item.group,
        {token: math.log(probability) - 10_000 for token, probability in item.probabilities.items()},
        model_id=item.model_id,
    ) for item in source]

    result = ReflectionTokenScorer().score(*logged)
    assert result.total == pytest.approx(ReflectionTokenScorer().score(*source).total)
    assert logged[0].to_dict()["input_kind"] == "log_probabilities"


def test_subnormal_probability_ratios_remain_finite():
    tiny = math.ulp(0.0)
    distribution = Distribution(Group.RELEVANCE, {"[Relevant]": tiny, "[Irrelevant]": tiny})
    assert dict(distribution.probabilities) == {"[Relevant]": 0.5, "[Irrelevant]": 0.5}


@pytest.mark.parametrize("group, values", [
    (Group.RELEVANCE, {"[Relevant]": 1.0}),
    (Group.RELEVANCE, {"[Relevant]": 0.5, "[Irrelevant]": 0.5, "unknown": 0}),
    (Group.RETRIEVAL, {"[Retrieval]": 0.5, "[No Retrieval]": 0.5}),
    (Group.SUPPORT, {"[Fully supported]": 0.5, "[Partially supported]": 0.5}),
    (Group.UTILITY, {f"[Utility:{index}]": 0.2 for index in range(1, 5)}),
])
def test_incomplete_or_unknown_groups_are_rejected_in_both_input_modes(group, values):
    with pytest.raises(ValueError, match="missing="):
        Distribution(group, values)
    with pytest.raises(ValueError, match="missing="):
        Distribution.from_logprobs(group, values)


@pytest.mark.parametrize("value,error", [
    (True, TypeError), ("0.5", TypeError), (None, TypeError),
    (math.nan, ValueError), (math.inf, ValueError), (-math.inf, ValueError),
    (-0.1, ValueError), (1.1, ValueError),
])
def test_probability_values_are_strict(value, error):
    with pytest.raises(error):
        Distribution(Group.RELEVANCE, {"[Relevant]": value, "[Irrelevant]": 0})


@pytest.mark.parametrize("value,error", [
    (True, TypeError), ("-1", TypeError),
    (math.nan, ValueError), (math.inf, ValueError), (-math.inf, ValueError), (0.1, ValueError),
])
def test_logprob_values_are_strict(value, error):
    with pytest.raises(error):
        Distribution.from_logprobs(Group.RELEVANCE, {"[Relevant]": value, "[Irrelevant]": -1})


def test_zero_mass_and_impossible_raw_probability_mass_fail():
    with pytest.raises(ValueError, match="positive total"):
        Distribution(Group.RELEVANCE, {"[Relevant]": 0, "[Irrelevant]": 0})
    with pytest.raises(ValueError, match="must not exceed"):
        Distribution(Group.RELEVANCE, {"[Relevant]": 0.8, "[Irrelevant]": 0.8})


def test_wrong_mapping_key_group_and_model_types_fail():
    with pytest.raises(TypeError):
        Distribution(Group.RELEVANCE, [("[Relevant]", 1)])
    with pytest.raises(TypeError):
        Distribution(Group.RELEVANCE, {1: 1})
    with pytest.raises(TypeError):
        Distribution(3, {})
    with pytest.raises(ValueError):
        Distribution("unknown", {})
    with pytest.raises(TypeError):
        Distribution(Group.RELEVANCE, {"[Relevant]": 1, "[Irrelevant]": 0}, model_id=5)
    with pytest.raises(ValueError):
        Distribution(Group.RELEVANCE, {"[Relevant]": 1, "[Irrelevant]": 0}, model_id="  ")


def test_distribution_and_serialized_diagnostics_are_isolated():
    values = {"[Relevant]": 0.2, "[Irrelevant]": 0.8}
    distribution = Distribution(Group.RELEVANCE, values, model_id="checkpoint")
    values["[Relevant]"] = 1
    diagnostics = distribution.to_dict()
    diagnostics["probabilities"]["[Relevant]"] = 0

    assert distribution.probabilities["[Relevant]"] == pytest.approx(0.2)
    with pytest.raises(TypeError):
        distribution.probabilities["[Relevant]"] = 0
    with pytest.raises(FrozenInstanceError):
        distribution.group = Group.SUPPORT


def test_logprob_inputs_are_detached_even_for_conditional_gate_calculations():
    logprobs = {"[Retrieval]": -1000.0, "[No Retrieval]": -1001.0, "[Continue to Use Evidence]": 0.0}
    distribution = Distribution.from_logprobs(Group.RETRIEVAL, logprobs)
    logprobs["[Retrieval]"] = -2000

    decision = ReflectionTokenRetrievalGate(lambda query: distribution).decide("question")
    assert decision.should_retrieve
    assert decision.confidence == pytest.approx(1 / (1 + math.exp(-1)))


@pytest.mark.parametrize("threshold, expected", [(0.0, True), (0.5, True), (0.75, False), (1.0, False)])
def test_initial_retrieval_gate_uses_conditional_probability_and_strict_threshold(threshold, expected):
    calls = []

    def provider(query):
        calls.append(query)
        return Distribution(Group.RETRIEVAL, {
            "[Retrieval]": 0.3, "[No Retrieval]": 0.1, "[Continue to Use Evidence]": 0.6,
        }, model_id="checkpoint@revision")

    decision = ReflectionTokenRetrievalGate(provider, threshold=threshold).decide("  question   text ")
    assert calls == ["question text"]
    assert decision.should_retrieve is expected
    assert decision.confidence == pytest.approx(0.75 if expected else 0.25)
    assert "checkpoint@revision" in decision.reason
    assert "initial_yes_no" in decision.reason


def test_two_token_initial_group_is_explicit_and_no_evidence_transition_is_not_invented():
    initial = Distribution(Group.INITIAL_RETRIEVAL, {"[Retrieval]": 0, "[No Retrieval]": 1})
    assert not ReflectionTokenRetrievalGate(lambda query: initial).decide("question").should_retrieve

    only_continue = Distribution(Group.RETRIEVAL, {
        "[Retrieval]": 0, "[No Retrieval]": 0, "[Continue to Use Evidence]": 1,
    })
    with pytest.raises(ValueError, match="zero total"):
        ReflectionTokenRetrievalGate(lambda query: only_continue).decide("question")


@pytest.mark.parametrize("value,error", [(True, TypeError), ("0.5", TypeError),
                                         (math.nan, ValueError), (-0.1, ValueError), (1.1, ValueError)])
def test_gate_threshold_is_strict(value, error):
    with pytest.raises(error):
        ReflectionTokenRetrievalGate(lambda query: None, threshold=value)


def test_gate_validates_provider_result_and_query_without_fallback():
    with pytest.raises(TypeError, match="callable"):
        ReflectionTokenRetrievalGate(None)
    with pytest.raises(TypeError, match="return"):
        ReflectionTokenRetrievalGate(lambda query: {"[Retrieval]": 1}).decide("question")
    with pytest.raises(ValueError, match="token group"):
        ReflectionTokenRetrievalGate(lambda query: distributions()[0]).decide("question")

    def unexpected_call(query):
        raise AssertionError("invalid query reached provider")

    gate = ReflectionTokenRetrievalGate(unexpected_call)
    with pytest.raises(TypeError):
        gate.decide(None)
    with pytest.raises(ValueError):
        gate.decide(" \n ")


def test_provider_failure_propagates_without_a_lexical_fallback():
    def failing_provider(query):
        raise RuntimeError("missing model probabilities")

    with pytest.raises(RuntimeError, match="missing model probabilities"):
        ReflectionTokenRetrievalGate(failing_provider).decide("question")


def test_scorer_requires_correct_typed_groups_and_valid_sequence_probability():
    scorer = ReflectionTokenScorer()
    relevance, support, utility = distributions()
    with pytest.raises(TypeError):
        scorer.score({}, support, utility)
    with pytest.raises(ValueError, match="requires"):
        scorer.score(support, relevance, utility)
    for value, error in [(True, TypeError), ("-1", TypeError), (math.nan, ValueError),
                         (math.inf, ValueError), (-math.inf, ValueError), (0.1, ValueError)]:
        with pytest.raises(error):
            scorer.score(relevance, support, utility, mean_sequence_logprob=value)


@pytest.mark.parametrize("value,error", [(True, TypeError), ("1", TypeError),
                                         (-1, ValueError), (math.nan, ValueError), (math.inf, ValueError)])
def test_scorer_weights_are_strict(value, error):
    with pytest.raises(error):
        ReflectionTokenScorer(utility_weight=value)


def test_weighted_score_overflow_fails_instead_of_returning_infinity():
    with pytest.raises(ValueError, match="finite"):
        ReflectionTokenScorer(1.79e308, 1.79e308, 1.79e308).score(*distributions())


def test_score_result_snapshots_and_validation():
    result = ReflectionTokenScorer().score(*distributions())
    payload = result.to_dict()
    payload["model_ids"].append("other")
    assert result.model_ids == ("self-rag-test",)
    with pytest.raises(FrozenInstanceError):
        result.total = 0
    with pytest.raises(ValueError):
        ReflectionScore(0.5, 0.5, 0, math.nan)
    with pytest.raises(ValueError):
        ReflectionScore(1.1, 0.5, 0, 1)
    with pytest.raises(ValueError):
        ReflectionScore(0.5, 0.5, 2, 1)
    with pytest.raises(ValueError):
        ReflectionScore(0.5, 0.5, 0, 1, sequence_probability=2)
    with pytest.raises(TypeError):
        ReflectionScore(0.5, 0.5, 0, 1, model_ids="checkpoint")


def test_probability_gate_integrates_with_existing_engine_and_calls_no_retriever_when_skipped():
    class RecordingRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k=5):
            self.calls.append((query, top_k))
            return [Document("Source vérifiée", doc_id="source")]

    retriever = RecordingRetriever()
    probability = 0.1

    def provider(query):
        return Distribution(Group.INITIAL_RETRIEVAL, {
            "[Retrieval]": probability, "[No Retrieval]": 1 - probability,
        })

    engine = SelfRAGEngine(
        retriever, StaticLLMClient("Réponse"), retrieval_gate=ReflectionTokenRetrievalGate(provider),
        evidence_critic=ScriptedEvidenceCritic(), max_refinements=0,
    )
    first = engine.ask("Question")
    assert first.status == "completed_without_retrieval"
    assert retriever.calls == []

    probability = 0.9
    second = engine.ask("Question", top_k=2)
    assert second.trace.retrieval.should_retrieve
    assert retriever.calls == [("Question", 2)]
