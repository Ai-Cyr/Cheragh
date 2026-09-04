"""RAFT paper recipe: grounded supervision, oracle dropout, and SFT export."""

from __future__ import annotations

import random

import pytest

from cheragh import Document
from cheragh.training import (
    RAFTDatasetBuilder,
    RAFTGeneratedAnswer,
    RAFTTrainingRecord,
    RetrievalTrainingExample,
)


def _example(*, multi_hop: bool = False) -> RetrievalTrainingExample:
    positives = [Document("The Acorn Group has its head office in Delhi.", doc_id="office")]
    if multi_hop:
        positives.append(Document("The Acorn family owns the Acorn Group.", doc_id="family"))
    return RetrievalTrainingExample(
        query="Where is the Acorn Group's head office?",
        positive_documents=tuple(positives),
        negative_documents=(
            Document("The Redwood Group has its head office in Paris.", doc_id="other-office"),
            Document("Oaks are broadleaf trees.", doc_id="trees"),
            Document("The train station opened in 1902.", doc_id="station"),
            Document("The river flows east.", doc_id="river"),
        ),
        answer="Delhi",
        metadata={"split": {"name": "train"}},
    )


def _teacher(query, documents, answer):
    return RAFTGeneratedAnswer(
        answer=answer,
        rationale=(
            "The office location is given explicitly: "
            f"##begin_quote##{documents[0].content}##end_quote##. "
            "Therefore the answer is Delhi."
        ),
    )


def test_dropout_preserves_supervision_without_leaking_oracle_into_prompt():
    example = _example()
    calls = []

    def teacher(query, documents, answer):
        calls.append((query, tuple(document.doc_id for document in documents), answer))
        return _teacher(query, documents, answer)

    included = RAFTDatasetBuilder(oracle_probability=1, answer_generator=teacher).build([example])[0]
    excluded = RAFTDatasetBuilder(oracle_probability=0, answer_generator=teacher).build([example])[0]

    assert included.render_target() == excluded.render_target()
    assert included.answer == excluded.answer == "Delhi"
    assert calls == [(example.query, ("office",), "Delhi")] * 2
    messages = excluded.to_messages()
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert "Delhi" not in messages[0]["content"]
    assert "[source: office]" not in messages[0]["content"]
    assert "##begin_quote##The Acorn Group" in messages[1]["content"]
    assert messages[1]["content"].endswith("##Answer: Delhi")
    assert "[source: office]" in included.to_messages()[0]["content"]
    assert excluded.oracle_documents[0].content == example.positive_documents[0].content


def test_legacy_records_keep_plain_answer_targets_and_original_serialization():
    record = RAFTDatasetBuilder().build([_example()])[0]
    assert record.render_target() == "Delhi"
    assert record.rationale is None
    assert [document.doc_id for document in record.documents] == [
        "office", "other-office", "trees", "station", "river"
    ]
    assert set(record.to_dict()) == {
        "question", "answer", "documents", "oracle_doc_ids", "oracle_included", "metadata"
    }


def test_multi_hop_quotes_are_verified_against_every_original_oracle():
    example = _example(multi_hop=True)

    def teacher(query, documents, answer):
        return RAFTGeneratedAnswer(
            answer=answer,
            rationale=" This also establishes ownership. ".join(
                f"##begin_quote## {document.content} ##end_quote##" for document in documents
            ),
        )

    record = RAFTDatasetBuilder(oracle_probability=0, answer_generator=teacher).build([example])[0]
    assert record.render_target().count("##begin_quote##") == 2
    assert record.oracle_doc_ids == ("office", "family")
    assert not {"office", "family"} & {document.doc_id for document in record.documents}


@pytest.mark.parametrize("quote", ["The office is in London.", "Oaks are broadleaf trees."])
def test_rejects_fabricated_or_distractor_only_quotations(quote):
    def teacher(query, documents, answer):
        return RAFTGeneratedAnswer(answer, f"##begin_quote##{quote}##end_quote##")

    with pytest.raises(ValueError, match="verbatim in an oracle"):
        RAFTDatasetBuilder(oracle_probability=0, answer_generator=teacher).build([_example()])


@pytest.mark.parametrize(
    "rationale",
    [
        "The answer is in the documents.",
        "##begin_quote##unclosed",
        "orphan##end_quote##",
        "##begin_quote## ##end_quote##",
        "##begin_quote##outer##begin_quote##inner##end_quote####end_quote##",
        "##begin_quote##valid##end_quote####end_quote##",
        "",
        None,
    ],
)
def test_rejects_missing_empty_or_malformed_quotations(rationale):
    with pytest.raises(ValueError):
        RAFTGeneratedAnswer("Delhi", rationale)


def test_rejects_teacher_answer_disagreement_and_invalid_return_type():
    def wrong_answer(query, documents, answer):
        return _teacher(query, documents, "London")

    with pytest.raises(ValueError, match="verified answer"):
        RAFTDatasetBuilder(answer_generator=wrong_answer).build([_example()])
    with pytest.raises(TypeError, match="return RAFTGeneratedAnswer"):
        RAFTDatasetBuilder(answer_generator=lambda *args: "Delhi").build([_example()])


def test_teacher_mutation_cannot_rewrite_oracle_or_example():
    example = _example()

    def mutating_teacher(query, documents, answer):
        target = _teacher(query, documents, answer)
        documents[0].content = "Fabricated oracle."
        documents[0].metadata["changed"] = True
        return target

    record = RAFTDatasetBuilder(answer_generator=mutating_teacher).build([example])[0]
    assert record.documents[0].content == example.positive_documents[0].content
    assert record.oracle_documents[0].content == example.positive_documents[0].content
    assert "changed" not in example.positive_documents[0].metadata
    serialized = record.to_dict()
    serialized["oracle_documents"][0]["content"] = "Changed serialized evidence"
    serialized["metadata"]["split"]["name"] = "test"
    assert record.oracle_documents[0].content != "Changed serialized evidence"
    assert record.metadata["split"]["name"] == "train"


def test_seeded_shuffle_changes_oracle_position_without_changing_dropout():
    examples = [_example()] * 30
    global_state = random.getstate()
    shuffled_builder = RAFTDatasetBuilder(oracle_probability=0.7, seed=82, shuffle_documents=True)
    shuffled = shuffled_builder.build(examples)
    repeated = shuffled_builder.build(examples)
    ordered = RAFTDatasetBuilder(oracle_probability=0.7, seed=82).build(examples)

    assert [record.to_dict() for record in shuffled] == [record.to_dict() for record in repeated]
    assert [record.oracle_included for record in shuffled] == [record.oracle_included for record in ordered]
    oracle_positions = {
        next(index for index, doc in enumerate(record.documents) if doc.doc_id == "office")
        for record in shuffled if record.oracle_included
    }
    assert len(oracle_positions) > 1
    assert random.getstate() == global_state


def test_fixed_cardinality_dropout_replaces_all_oracles_with_distractors():
    example = _example(multi_hop=True)
    records = RAFTDatasetBuilder(
        oracle_probability=0.5, seed=2, shuffle_documents=True, context_document_count=3
    ).build([example] * 12)

    assert {record.oracle_included for record in records} == {False, True}
    for record in records:
        assert len(record.documents) == 3
        keys = {document.doc_id for document in record.documents}
        assert len(keys) == 3
        if record.oracle_included:
            assert {"office", "family"}.issubset(keys)
        else:
            assert not {"office", "family"} & keys


def test_fixed_context_rejects_insufficient_distractors_or_too_many_oracles():
    with pytest.raises(ValueError, match="Not enough distractors"):
        RAFTDatasetBuilder(oracle_probability=0, context_document_count=5).build([_example()])
    with pytest.raises(ValueError, match="smaller than the included oracle count"):
        RAFTDatasetBuilder(context_document_count=1).build([_example(multi_hop=True)])


def test_rejects_oracle_content_disguised_as_a_different_distractor_id():
    oracle = Document("Actual oracle evidence", doc_id="original")
    example = RetrievalTrainingExample(
        "question", (oracle,), (Document(" Actual oracle evidence ", doc_id="copy"),), answer="answer"
    )
    with pytest.raises(ValueError, match="duplicate oracle content"):
        RAFTDatasetBuilder(oracle_probability=0).build([example])


def test_manual_record_requires_matching_original_evidence_for_rationale():
    example = _example()
    rationale = _teacher(example.query, example.positive_documents, example.answer).rationale
    with pytest.raises(ValueError, match="requires oracle documents"):
        RAFTTrainingRecord("q", "Delhi", (), ("office",), False, rationale=rationale)
    with pytest.raises(ValueError, match="match oracle_doc_ids"):
        RAFTTrainingRecord(
            "q", "Delhi", (), ("unrelated",), False, rationale=rationale,
            oracle_documents=example.positive_documents,
        )
    with pytest.raises(ValueError, match="match the included context"):
        RAFTTrainingRecord(
            "q", "Delhi", (Document("Different oracle text", doc_id="office"),), ("office",), True,
            rationale=rationale, oracle_documents=example.positive_documents,
        )
    record = RAFTTrainingRecord(
        "q", "Delhi", example.positive_documents, ("office",), True, rationale=rationale
    )
    assert record.oracle_documents[0].content == example.positive_documents[0].content


@pytest.mark.parametrize("kwargs", [{"shuffle_documents": 1}, {"context_document_count": True},
                                  {"context_document_count": 0}, {"answer_generator": 1}])
def test_builder_validates_recipe_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        RAFTDatasetBuilder(**kwargs)
