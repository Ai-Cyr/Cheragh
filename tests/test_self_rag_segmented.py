"""Independent search fixtures: global path choice, evidence states, budgets."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from cheragh import Document
from cheragh.self_rag import (
    ReflectionSegment,
    ReflectionTokenDistribution as Distribution,
    ReflectionTokenGroup as Group,
    ReflectionTokenScorer,
    RetrievalAction,
    SegmentedSelfRAGEngine,
)


def retrieval(action):
    return Distribution(Group.RETRIEVAL, {token.value: float(token == action) for token in RetrievalAction})


def relevance(probability):
    return Distribution(Group.RELEVANCE, {"[Relevant]": probability, "[Irrelevant]": 1 - probability}, model_id="fixture")


def segment(text, probability=1., *, next_action=None, relevance_present=True, tokens=3, support=None, utility=None):
    return ReflectionSegment(text, text, tokens, math.log(.5),
                             "retrieval" if next_action is not None else "eos",
                             relevance(probability) if relevance_present else None,
                             support, utility, retrieval(next_action) if next_action is not None else None)


class Decoder:
    def __init__(self, generate, first=RetrievalAction.RETRIEVE):
        self.generate = generate
        self.first = first
        self.prompts = []
        self.gate_prompts = []

    def retrieval_distribution(self, prompt):
        self.gate_prompts.append(prompt)
        return retrieval(self.first)

    def decode_segment(self, prompt, *, max_new_tokens):
        self.prompts.append(prompt)
        return self.generate(prompt, max_new_tokens)


def test_beam_finds_globally_better_passage_path_that_greedy_loses():
    calls = []

    def retrieve(query, top_k):
        calls.append(query)
        if "answer-A" in query:
            return [Document("tail-A", doc_id="tail-a")]
        if "answer-B" in query:
            return [Document("tail-B", doc_id="tail-b")]
        return [Document("passage-A", doc_id="a", score=100), Document("passage-B", doc_id="b", score=0)]

    def generate(prompt, _):
        if "tail-A" in prompt:
            return segment("finish-A", .1)
        if "tail-B" in prompt:
            return segment("finish-B", .95)
        if "passage-A" in prompt:
            return segment("answer-A", .9, next_action=RetrievalAction.RETRIEVE)
        return segment("answer-B", .8, next_action=RetrievalAction.RETRIEVE)

    options = dict(scorer=ReflectionTokenScorer(1, 0, 0), use_sequence_score=False, max_segments=2)
    retriever = SimpleNamespace(retrieve=retrieve)
    wide = SegmentedSelfRAGEngine(retriever, Decoder(generate), beam_width=2, **options).ask("question")
    greedy = SegmentedSelfRAGEngine(retriever, Decoder(generate), beam_width=1, **options).ask("question")
    assert [part.text for part in wide.segments] == ["answer-B", "finish-B"]
    assert [part.text for part in greedy.segments] == ["answer-A", "finish-A"]
    # Oracle: enumerate the two possible complete path products, independent
    # of beam bookkeeping and of retriever ranking.
    assert math.exp(wide.log_score) == pytest.approx(max(.9 * .1, .8 * .95))
    assert [doc.doc_id for doc in wide.documents] == ["b", "tail-b"]
    assert wide.answer == "answer-B [source: b] finish-B [source: tail-b]"
    assert "question\nanswer-B" in calls
    assert wide.trace.beam_sizes == [2, 0]
    json.dumps(wide.to_dict(), allow_nan=False)


def test_continue_reuses_branch_evidence_and_no_retrieval_has_no_citation():
    retriever_calls = []

    def retrieve(query, top_k):
        retriever_calls.append(query)
        return [Document("evidence-body", doc_id="evidence")]

    def generate(prompt, _):
        if prompt.endswith(RetrievalAction.NO_RETRIEVAL.value):
            return segment("subjective ending", relevance_present=False)
        if prompt.endswith(RetrievalAction.CONTINUE.value):
            assert prompt.count("<paragraph>") == 1
            assert "evidence-body" in prompt and "first sentence" in prompt
            return segment("second sentence", relevance_present=False, next_action=RetrievalAction.NO_RETRIEVAL)
        return segment("first sentence", .9, next_action=RetrievalAction.CONTINUE)

    model = Decoder(generate)
    result = SegmentedSelfRAGEngine(SimpleNamespace(retrieve=retrieve), model).ask("task")
    assert retriever_calls == ["task"]
    assert len(model.gate_prompts) == 1, "boundary probabilities should be reused at their original position"
    assert [item.action for item in result.segments] == [RetrievalAction.RETRIEVE, RetrievalAction.CONTINUE,
                                                       RetrievalAction.NO_RETRIEVAL]
    assert [item.document.doc_id if item.document else None for item in result.segments] == ["evidence", "evidence", None]
    assert result.answer == "first sentence [source: evidence] second sentence [source: evidence] subjective ending"
    assert result.segments[1].score.relevance == pytest.approx(.9)
    assert result.segments[2].score.relevance == 0
    assert result.status == "eos"


def test_early_eos_does_not_stop_other_hypotheses():
    def generate(prompt, _):
        if prompt.endswith(RetrievalAction.CONTINUE.value):
            return segment("better completed path", 1.)
        if "early" in prompt:
            return segment("early ending", .1)
        return segment("keep going", .9, next_action=RetrievalAction.CONTINUE)

    retriever = SimpleNamespace(retrieve=lambda *_, **__: [Document("early", doc_id="a"), Document("late", doc_id="b")])
    result = SegmentedSelfRAGEngine(retriever, Decoder(generate), scorer=ReflectionTokenScorer(1, 0, 0),
                                   use_sequence_score=False).ask("task")
    assert "better completed path" in result.answer
    assert "early ending" not in result.answer


def test_initial_no_retrieval_does_not_require_a_retriever_or_fabricate_evidence_scores():
    model = Decoder(lambda *args: segment("a creative answer", relevance_present=False), first=RetrievalAction.NO_RETRIEVAL)
    result = SegmentedSelfRAGEngine(None, model).ask("write a story")
    assert result.answer == "a creative answer"
    assert result.documents == []
    assert result.segments[0].score.relevance == result.segments[0].score.support == 0
    assert result.segments[0].missing_reflections == ("utility",)


@pytest.mark.parametrize("threshold, expected_retrieval", [(0.75, False), (0.749, True)])
def test_threshold_equality_and_initial_continue_exclusion(threshold, expected_retrieval):
    model = Decoder(lambda *args: segment("answer"))
    model.retrieval_distribution = lambda _: Distribution(Group.RETRIEVAL, {
        "[Retrieval]": .3, "[No Retrieval]": .1, "[Continue to Use Evidence]": .6,
    })
    calls = []
    retriever = SimpleNamespace(retrieve=lambda *_, **__: calls.append(1) or [Document("evidence", doc_id="d")])
    result = SegmentedSelfRAGEngine(retriever, model, retrieval_threshold=threshold).ask("task")
    assert bool(calls) is expected_retrieval
    assert bool(result.documents) is expected_retrieval


def test_token_and_retriever_overreturn_limits_bound_model_work():
    yielded = []

    def retrieve(*_, **__):
        for index in range(1000):
            yielded.append(index)
            yield Document("evidence " + str(index), doc_id=str(index))

    allowances = []

    def generate(prompt, max_new_tokens):
        allowances.append(max_new_tokens)
        return segment("answer", tokens=max_new_tokens)

    result = SegmentedSelfRAGEngine(SimpleNamespace(retrieve=retrieve), Decoder(generate), top_k=3,
                                   max_new_tokens=4, max_generated_tokens=6).ask("task")
    assert yielded == [0, 1, 2]
    assert allowances == [4, 2]
    assert result.trace.generated_tokens == 6
    assert result.trace.model_calls == 3
    assert result.status == "generated_token_limit"


def test_global_call_budget_returns_generated_prefix_without_reexpanding_it():
    model = Decoder(lambda *args: segment("prefix", .1, next_action=RetrievalAction.CONTINUE))
    retriever = SimpleNamespace(retrieve=lambda *_, **__: [Document("evidence", doc_id="d")])
    result = SegmentedSelfRAGEngine(retriever, model, max_model_calls=2,
                                   scorer=ReflectionTokenScorer(1, 0, 0), use_sequence_score=False).ask("task")
    assert result.trace.model_calls == 2
    assert result.status == "model_call_limit"
    assert result.answer == "prefix [source: d]"


def test_nonpositive_scores_are_pruned_without_multiplying_negative_utilities():
    low = Distribution(Group.UTILITY, {f"[Utility:{index}]": float(index == 1) for index in range(1, 6)})
    model = Decoder(lambda *args: segment("unsupported", 0, utility=low))
    retriever = SimpleNamespace(retrieve=lambda *_, **__: [Document("evidence", doc_id="d")])
    result = SegmentedSelfRAGEngine(retriever, model, use_sequence_score=False).ask("task")
    assert result.answer == ""
    assert result.status == "no_positive_candidates"
    assert result.trace.pruned_nonpositive == 1


def test_forged_citations_are_replaced_with_actual_conditioning_source():
    model = Decoder(lambda *args: segment("claim [source: invented]"))
    source = Document("evidence")
    result = SegmentedSelfRAGEngine(SimpleNamespace(retrieve=lambda *_, **__: [source]), model).ask("task")
    assert "invented" not in result.answer
    assert result.documents[0].doc_id.startswith("selfrag-")
    assert source.doc_id is None


@pytest.mark.parametrize("prediction, error", [
    ("text-only self assessment", TypeError),
    (segment("answer", relevance_present=False), ValueError),
    (segment("answer", tokens=30), ValueError),
])
def test_invalid_model_outputs_fail_without_lexical_fallback(prediction, error):
    retriever = SimpleNamespace(retrieve=lambda *_, **__: [Document("evidence", doc_id="d")])
    with pytest.raises(error):
        SegmentedSelfRAGEngine(retriever, Decoder(lambda *args: prediction), max_new_tokens=10).ask("task")


def test_conflicting_passage_ids_do_not_create_ambiguous_citations():
    retriever = SimpleNamespace(retrieve=lambda *_, **__: [Document("a", doc_id="same"), Document("b", doc_id="same")])
    with pytest.raises(ValueError, match="different passage"):
        SegmentedSelfRAGEngine(retriever, Decoder(lambda *args: segment("answer"))).ask("task")


def test_budget_keeps_a_better_unfinished_prefix_even_when_another_beam_completed():
    def generate(prompt, _):
        if "continuing" in prompt:
            return segment("best prefix", .9, next_action=RetrievalAction.CONTINUE)
        return segment("weak completed answer", .1)

    retriever = SimpleNamespace(retrieve=lambda *_, **__: [Document("continuing", doc_id="a"),
                                                          Document("finished", doc_id="b")])
    result = SegmentedSelfRAGEngine(retriever, Decoder(generate), max_model_calls=3,
                                   scorer=ReflectionTokenScorer(1, 0, 0), use_sequence_score=False).ask("task")
    assert result.answer == "best prefix [source: a]"
    assert result.status == "model_call_limit"
    assert math.exp(result.log_score) == pytest.approx(.9)


def test_budget_does_not_resurrect_a_fully_expanded_pruned_parent():
    def generate(prompt, _):
        if prompt.endswith(RetrievalAction.CONTINUE.value):
            return segment("rejected continuation", 0.)
        return segment("prefix A" if "passage A" in prompt else "prefix B",
                       .9 if "passage A" in prompt else .8, next_action=RetrievalAction.CONTINUE)

    retriever = SimpleNamespace(retrieve=lambda *_, **__: [Document("passage A", doc_id="a"),
                                                          Document("passage B", doc_id="b")])
    result = SegmentedSelfRAGEngine(retriever, Decoder(generate), max_model_calls=4,
                                   scorer=ReflectionTokenScorer(1, 0, 0), use_sequence_score=False).ask("task")
    assert result.answer == "prefix B [source: b]"
    assert result.trace.pruned_nonpositive == 1
    assert result.status == "model_call_limit"
