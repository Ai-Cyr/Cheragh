"""TimeR4 graph facts, grounded rewrite, temporal fusion and ACL regression."""
from datetime import datetime, timezone
import json

import numpy as np
import pytest

from cheragh import Document
from cheragh.base import CallableLLMClient, EmbeddingModel, StaticLLMClient
from cheragh.security.access_control import AccessPolicy, Principal
from cheragh.temporal.time_r4 import (
    TemporalConstraint, TemporalFact, TemporalInterval, TimeR4Retriever, build_temporal_training_example,
)


def fact(name, when, *, end=None, tenant="a"):
    interval = TemporalInterval(when, end) if end else TemporalInterval.at(when)
    return TemporalFact(name, name, "visits", "Iraq", interval,
                        Document(f"Source reports {name}'s visit.", {"tenant_id": tenant, "nested": [1]}, f"source-{name}"))


class SemanticEncoder(EmbeddingModel):
    """Deterministic role-specific semantic scores, not the production backend."""

    def __init__(self, *, anchor=False):
        self.anchor = anchor
        self.queries = []
        self.batches = []

    def embed_documents(self, texts):
        self.batches.append(list(texts))
        return np.array([[1, 0] if (not self.anchor or text.startswith("Ministry")) else [.1, 1] for text in texts])

    def embed_query(self, text):
        self.queries.append(text)
        return np.array([1, 0])


def rewrite(*, relation="after", anchor="Ministry", start=None, end=None):
    return json.dumps({"query": "Who visited Iraq?", "constraints": [
        {"relation": relation, "anchor_fact_id": anchor, "start": start, "end": end},
    ]})


def retriever(facts=None, llm=None, **kwargs):
    if facts is None:
        facts = [fact("Early", "2016-01-04"), fact("Ministry", "2016-01-05"),
                 fact("Jack", "2016-01-06"), fact("Late", "2016-01-09")]
    return TimeR4Retriever(facts, SemanticEncoder(anchor=True), SemanticEncoder(),
                           llm or StaticLLMClient(rewrite()), **kwargs)


def test_four_stages_resolve_implicit_time_and_rerank_nearest_eligible_fact():
    engine = retriever(anchor_top_k=1, embedding_batch_size=2)
    result = engine.retrieve_with_trace("Who visited Iraq after the Ministry?", top_k=2)
    assert [doc.doc_id for doc in result.documents] == ["Jack", "Late"]
    assert [stage["stage"] for stage in result.trace] == ["retrieve_facts", "rewrite", "retrieve_temporal", "rerank"]
    assert result.trace[0]["fact_ids"] == ["Ministry"]
    assert "2016-01-05" in engine.temporal_embedding_model.queries[0]
    assert engine.fact_embedding_model.queries != engine.temporal_embedding_model.queries
    assert all(len(batch) <= 2 for batch in engine.fact_embedding_model.batches)
    assert result.documents[0].metadata["source_doc_id"] == "source-Jack"
    assert result.documents[0].metadata["temporal_fact"]["predicate"] == "visits"
    assert result.trace[-1]["temporally_rejected"] == 2
    assert result.documents[0].score > result.documents[1].score
    result.documents[0].metadata["nested"].append(99)
    assert engine.retrieve("After Ministry?", 1)[0].metadata["nested"] == [1]


@pytest.mark.parametrize(("relation", "expected"), [("before", ["Early"]), ("after", ["Jack", "Late"]), ("during", ["Ministry"])])
def test_strict_temporal_relations_discard_wrong_interval(relation, expected):
    result = retriever(llm=StaticLLMClient(rewrite(relation=relation))).retrieve("relative temporal question")
    assert [doc.doc_id for doc in result] == expected


def test_caller_constraints_are_intersected_and_cannot_be_relaxed_by_rewrite():
    engine = retriever()
    during = TemporalConstraint("during", TemporalInterval.at("2016-01-09"))
    assert [doc.doc_id for doc in engine.retrieve("After Ministry?", constraints=[during])] == ["Late"]
    never = TemporalConstraint("before", TemporalInterval.at("2016-01-01"))
    assert engine.retrieve("After Ministry?", constraints=[never]) == []


def test_explicit_calendar_granularity_and_open_bounds():
    february = TemporalInterval.at("2024-02")
    assert february.start == datetime(2024, 2, 1, tzinfo=timezone.utc)
    assert february.end.day == 29
    assert TemporalConstraint("during", TemporalInterval("2024", None)).matches(TemporalInterval("2025", None))
    assert not TemporalConstraint("before", TemporalInterval.at("2026")).matches(TemporalInterval("2025", None))
    assert not TemporalConstraint("during", february).matches(TemporalInterval(None, "2024-02-10"))
    engine = retriever(llm=StaticLLMClient(rewrite(relation="during", anchor=None, start="2016", end="2016")))
    assert len(engine.retrieve("Who visited Iraq in 2016?")) == 4


@pytest.mark.parametrize("start,end", [("2023-02-29", "2023-03-01"), ("2024-13", None), ("2025", "2024"), (None, None), ("2024-01-01T00:00:00", None)])
def test_intervals_reject_invalid_dates_and_naive_timestamps(start, end):
    with pytest.raises((TypeError, ValueError)):
        TemporalInterval(start, end)


@pytest.mark.parametrize("response", [
    "not JSON",
    rewrite(anchor="unknown"),
    rewrite(anchor="Ministry", start="1900"),
    rewrite(anchor=None, start="1900", end="1900"),
    '{"query":"Who visited?","constraints":[]}',
    '{"query":"Who visited?","query":"different","constraints":[]}',
])
def test_rewrite_failure_is_closed_and_never_falls_back_to_unfiltered_results(response):
    engine = retriever(llm=StaticLLMClient(response))
    with pytest.raises(ValueError):
        engine.retrieve("After Ministry?")
    engine.rewrite_failure = "empty"
    result = engine.retrieve_with_trace("After Ministry?")
    assert result.documents == ()
    assert result.trace[-1]["status"] == "failed"
    assert engine.temporal_embedding_model.queries == []


def test_acl_filters_before_rewrite_and_trace_even_when_hidden_anchor_ranks_first():
    received = []

    def generate(prompt, **kwargs):
        received.append(prompt)
        assert "SECRET" not in prompt
        return rewrite(anchor="Ministry")

    facts = [fact("Ministry SECRET", "2016-01-01", tenant="b"), fact("Ministry", "2016-01-05"), fact("Jack", "2016-01-06")]
    engine = retriever(facts, CallableLLMClient(generate), principal=Principal("alice", tenant_ids={"a"}), anchor_top_k=1)
    result = engine.retrieve_with_trace("After Ministry?")
    assert [doc.doc_id for doc in result.documents] == ["Jack"]
    assert "SECRET" not in json.dumps(result.trace)
    assert len(received) == 1


def test_custom_deny_all_policy_stops_before_embedding_query_and_llm():
    class DenyAll(AccessPolicy):
        def filter_documents(self, documents, principal=None):
            return []

    engine = retriever(policy=DenyAll(), llm=StaticLLMClient("must not be called"))
    assert engine.retrieve("After Ministry?") == []
    assert engine.fact_embedding_model.queries == []


def test_custom_policy_evaluates_original_source_ids():
    class SourcePolicy(AccessPolicy):
        def filter_documents(self, documents, principal=None):
            return [doc for doc in documents if doc.doc_id in {"source-Ministry", "source-Jack"}]

    engine = retriever(policy=SourcePolicy())
    assert [doc.doc_id for doc in engine.retrieve("After Ministry?")] == ["Jack"]


def test_conflicting_source_ids_cannot_expand_authorization():
    first = fact("Ministry", "2016-01-05", tenant="a")
    other = TemporalFact("secret", "Secret", "visits", "Iraq", TemporalInterval.at("2016-01-06"),
                         Document(first.source.content, {"tenant_id": "b"}, first.source.doc_id))
    with pytest.raises(ValueError, match="identical source"):
        retriever([first, other])


def test_caller_constraints_still_guide_tks_when_rewrite_adds_none():
    engine = retriever(llm=StaticLLMClient(json.dumps({"query": "Who visited Iraq?", "constraints": []})))
    result = engine.retrieve("Who visited Iraq?", constraints=[TemporalConstraint("during", TemporalInterval.at("2016-01-09"))])
    assert [doc.doc_id for doc in result] == ["Late"]
    assert "2016-01-09" in engine.temporal_embedding_model.queries[0]


def test_constraints_iterators_cannot_be_consumed_and_silently_discarded():
    engine = retriever(llm=StaticLLMClient(json.dumps({"query": "Who visited Iraq?", "constraints": []})))
    constraints = iter([TemporalConstraint("during", TemporalInterval.at("2016-01-09"))])
    with pytest.raises(TypeError, match="sequence"):
        engine.retrieve("Who visited Iraq?", constraints=constraints)
    assert engine.fact_embedding_model.queries == []


def test_caller_constraint_sequence_is_frozen_before_rewrite():
    constraints = [TemporalConstraint("during", TemporalInterval.at("2016-01-09"))]

    def generate(prompt, **kwargs):
        constraints.clear()
        return json.dumps({"query": "Who visited Iraq?", "constraints": []})

    result = retriever(llm=CallableLLMClient(generate)).retrieve("Who visited Iraq?", constraints=constraints)
    assert [doc.doc_id for doc in result] == ["Late"]


def test_rewrite_budget_counts_complete_prompt_and_bounds_entire_json_response():
    calls = []

    def generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return rewrite()

    engine = retriever(llm=CallableLLMClient(generate))
    engine.retrieve("After Ministry?")
    actual_prompt_length = len(calls[0][0])
    engine.max_rewrite_chars = actual_prompt_length - 1
    with pytest.raises(ValueError, match="max_rewrite_chars"):
        engine.retrieve("After Ministry?")
    assert len(calls) == 1
    assert calls[0][1]["max_tokens"] == engine.max_rewrite_tokens
    oversized = " " * 24_001 + rewrite()
    with pytest.raises(ValueError, match="max_rewrite_chars"):
        retriever(llm=StaticLLMClient(oversized)).retrieve("After Ministry?")


def test_candidate_and_prompt_bounds_are_explicit():
    engine = retriever(candidate_top_k=1)
    with pytest.raises(ValueError, match="candidate_top_k"):
        engine.retrieve("After Ministry?", top_k=2)
    result = engine.retrieve_with_trace("After Ministry?", top_k=1)
    assert result.trace[2]["candidate_limit_reached"]
    assert result.documents == ()  # first semantic candidate violates time; no broadening
    engine = retriever(max_rewrite_chars=10)
    with pytest.raises(ValueError, match="max_rewrite_chars"):
        engine.retrieve("After Ministry?")


def test_time_content_and_joint_negatives_are_valid_training_records():
    positive = fact("Jack", "2016-01-06")
    constraint = TemporalConstraint("after", TemporalInterval.at("2016-01-05"))
    example = build_temporal_training_example(
        "Who visits Iraq after 2016-01-05?", positive, constraint=constraint,
        incorrect_interval=TemporalInterval.at("2016-01-04"), incorrect_subject="Jack",
        incorrect_predicate="leaves", incorrect_object="Paris",
    )
    assert len(example.positive_documents) == 1 and len(example.negative_documents) == 3
    assert [doc.metadata["temporal_negative_kind"] for doc in example.negative_documents] == ["time", "content", "both"]
    assert all(doc.metadata["synthetic_training_fact"] for doc in example.negative_documents)
    assert example.negative_documents[0].metadata["temporal_fact"]["predicate"] == "visits"
    with pytest.raises(ValueError, match="time negative"):
        build_temporal_training_example("query", positive, constraint=constraint,
                                        incorrect_interval=TemporalInterval.at("2016-01-07"),
                                        incorrect_subject="X", incorrect_predicate="Y", incorrect_object="Z")


def test_real_torch_trainer_accepts_temporal_negatives():
    torch = pytest.importorskip("torch")
    from cheragh.training.torch_trainer import TorchRetrievalTrainer
    example = build_temporal_training_example(
        "Who visits Iraq after 2016-01-05?", fact("Jack", "2016-01-06"),
        constraint=TemporalConstraint("after", TemporalInterval.at("2016-01-05")),
        incorrect_interval=TemporalInterval.at("2016-01-04"), incorrect_subject="Jack",
        incorrect_predicate="leaves", incorrect_object="Paris",
    )
    texts = [example.query, *[doc.content for doc in example.positive_documents + example.negative_documents]]
    ids = {text: i for i, text in enumerate(texts)}

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Embedding(len(ids), 4)

        def forward(self, values):
            return self.table(torch.tensor([ids[text] for text in values]))

    torch.manual_seed(1)
    encoder = Encoder()
    trainer = TorchRetrievalTrainer(encoder, encoder, torch.optim.SGD(encoder.parameters(), lr=.1))
    report = trainer.fit([example], epochs=4)
    assert report["steps"] == 4
    assert report["epoch_losses"][-1] < report["epoch_losses"][0]
