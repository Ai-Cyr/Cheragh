"""Calibrated neural evaluation, source-preserving strips and CRAG actions."""
from __future__ import annotations

from contextlib import nullcontext
import os
from types import SimpleNamespace

import numpy as np
import pytest

from cheragh.base import BaseRetriever, Document, LLMClient
from cheragh.corrective.engine import CorrectiveRAGEngine, RetrievalAction
from cheragh.corrective.semantic import CrossEncoderRetrievalGrader, LogisticCalibration, SemanticKnowledgeRefiner
from cheragh.engine import RAGEngine


def test_logistic_calibration_is_fitted_and_improves_validation_likelihood():
    logits = [-4, -2, 0, 1, 2, 3, 5, 7]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    calibration = LogisticCalibration.fit(logits, labels)
    probabilities = calibration.probabilities(logits)
    assert calibration.sample_count == 8
    assert probabilities == sorted(probabilities)
    assert probabilities[0] < 0.05
    assert probabilities[-1] > 0.95
    before = np.mean(np.logaddexp(0, np.asarray(logits)) - np.asarray(labels) * np.asarray(logits))
    after = -np.mean(np.asarray(labels) * np.log(probabilities) + (1 - np.asarray(labels)) * np.log1p(-np.asarray(probabilities)))
    assert after < before
    assert calibration.probabilities([-1e200, 1e200]) == [0, 1]


@pytest.mark.parametrize("scores,labels", [([0, 0], [0, 1]), ([0, 1], [1, 1]),
                                         ([0, 1], [0]), ([0, 1], [False, True]),
                                         ([np.nan, 1], [0, 1]), ([0, 1], [1, 0])])
def test_invalid_or_inverted_calibration_is_rejected(scores, labels):
    with pytest.raises(ValueError):
        LogisticCalibration.fit(scores, labels)


class FakeTensor:
    """Tiny deterministic tensor boundary; pretrained CPU smoke is run separately."""
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def to(self, device):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class TestTokenizer:
    __test__ = False
    model_max_length = 12

    def __init__(self):
        self.batches = []

    def __call__(self, query, text, **kwargs):
        assert kwargs.get("truncation") is False
        def ids(question, passage):
            vocabulary = {"cardiac": 8, "uncertain": 4}
            return [0] * (len(question.split()) + 3) + [vocabulary.get(word.strip(".!?"), 1) for word in passage.split()]
        if isinstance(query, str):
            return {"input_ids": ids(query, text)}
        self.batches.extend(zip(query, text))
        encoded = [ids(question, passage) for question, passage in zip(query, text)]
        maximum = max(map(len, encoded))
        return {"input_ids": FakeTensor([row + [0] * (maximum - len(row)) for row in encoded])}


class TestClassifier:
    __test__ = False
    config = SimpleNamespace(max_position_embeddings=12)

    def __init__(self, num_labels=1):
        self.calls = []
        self.num_labels = num_labels

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        self.calls.append(inputs)
        scores = (inputs["input_ids"].values.max(axis=1) - 4) * 2
        logits = scores[:, None] if self.num_labels == 1 else np.column_stack((np.zeros(len(scores)), scores))
        return SimpleNamespace(logits=FakeTensor(logits))


def make_grader(**kwargs):
    model, tokenizer = TestClassifier(kwargs.pop("num_labels", 1)), TestTokenizer()
    grader = CrossEncoderRetrievalGrader(model=model, tokenizer=tokenizer,
                                         calibration=LogisticCalibration(1, 0), **kwargs)
    grader._load = lambda: SimpleNamespace(inference_mode=nullcontext)
    return grader


@pytest.mark.parametrize("num_labels", [1, 2])
def test_classifier_logits_calibrate_into_three_document_set_actions(num_labels):
    grader = make_grader(num_labels=num_labels)
    strong = Document("cardiac", doc_id="good")
    weak = Document("uncertain", doc_id="maybe")
    bad = Document("noise", doc_id="bad")
    assert grader.grade("query", [bad, strong]).action is RetrievalAction.CORRECT
    assert grader.grade("query", [bad, weak]).action is RetrievalAction.AMBIGUOUS
    assert grader.grade("query", [bad]).action is RetrievalAction.INCORRECT
    assert grader.grade("query", []).action is RetrievalAction.INCORRECT
    assert grader.score_documents("query", [weak]) == [0.5]


def test_long_source_tail_is_scored_and_all_model_windows_respect_real_token_limit():
    grader = make_grader(batch_size=2)
    source = Document("noise " * 30 + "cardiac", doc_id="a")
    score = grader.score_documents("query", [source])[0]
    assert score > 0.99
    assert "".join(text for _, text in grader.tokenizer.batches) == source.content
    assert len(grader.model.calls) >= 2
    assert all(call["input_ids"].shape[1] <= 12 for call in grader.model.calls)


def test_model_calls_fail_before_any_silent_truncation_or_uncalibrated_probability():
    grader = make_grader(max_windows_per_document=1)
    with pytest.raises(ValueError, match="window budget"):
        grader.score_documents("query", [Document("noise " * 30)])
    assert grader.model.calls == []
    with pytest.raises(ValueError, match="leave no evaluator window"):
        grader.score_documents("word " * 20, [Document("cardiac")])
    grader.calibration = None
    with pytest.raises(ValueError, match="calibrate"):
        grader.grade("query", [Document("cardiac")])


def test_calibrate_fits_actual_evaluator_logits_then_changes_probability_mapping():
    grader = make_grader()
    grader.calibration = None
    calibration = grader.calibrate([("query", Document("noise")), ("query", Document("cardiac"))], [0, 1])
    assert calibration.sample_count == 2
    assert grader.grade("query", [Document("cardiac")]).action is RetrievalAction.CORRECT


def test_strip_refinement_filters_noise_and_retains_original_order_spans_and_acl():
    source = Document("cardiac first. noise unrelated. cardiac last.", doc_id="a",
                      metadata={"tenant_id": "one", "nested": {"value": "original"}})
    grader = make_grader()
    refiner = SemanticKnowledgeRefiner(grader, sentences_per_strip=1)
    refined = refiner.refine("query", [source])[0]
    assert refined.content == "cardiac first.\n\ncardiac last."
    assert refined.doc_id == "a"
    assert refined.metadata["tenant_id"] == "one"
    provenance = refined.metadata["corrective_provenance"]["refinement"]
    assert [strip["index"] for strip in provenance["retained_strips"]] == [0, 2]
    assert provenance["discarded_strips"][0]["index"] == 1
    assert [source.content[strip["start"]:strip["end"]] for strip in provenance["retained_strips"]] == [
        "cardiac first.", "cardiac last."]
    refined.metadata["nested"]["value"] = "changed"
    assert source.metadata["nested"]["value"] == "original"


def test_refinement_preserves_sentence_order_even_when_later_strip_scores_higher():
    class Grader:
        def score_documents(self, query, documents):
            return [0.7, 0.1, 0.99]
    refined = SemanticKnowledgeRefiner(Grader(), sentences_per_strip=1).refine(
        "query", [Document("First fact. Noise fact. Last fact.", doc_id="a")])
    assert refined[0].content == "First fact.\n\nLast fact."


def test_refinement_budget_fails_explicitly_instead_of_dropping_relevant_strips():
    source = Document("cardiac first. cardiac last.", doc_id="a")
    grader = make_grader()
    with pytest.raises(ValueError, match="max_strips"):
        SemanticKnowledgeRefiner(grader, sentences_per_strip=1, max_strips=1).refine("query", [source])
    assert grader.model.calls == []
    with pytest.raises(ValueError, match="max_refined_tokens"):
        SemanticKnowledgeRefiner(grader, sentences_per_strip=1, max_refined_tokens=5).refine("query", [source])


class FixedRetriever(BaseRetriever):
    def __init__(self, *documents):
        self.documents, self.calls = list(documents), []

    def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.documents[:top_k]


class Reader(LLMClient):
    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.answer


@pytest.mark.parametrize("primary_text,action,expected_ids", [
    ("cardiac first. noise unrelated.", "correct", ["internal"]),
    ("uncertain first. noise unrelated.", "ambiguous", ["external", "internal"]),
    ("noise unrelated.", "incorrect", ["external"]),
])
def test_semantic_grader_and_refiner_drive_all_three_corrective_paths(primary_text, action, expected_ids):
    primary = FixedRetriever(Document(primary_text, doc_id="internal"))
    external = FixedRetriever(Document("cardiac external. noise unrelated.", doc_id="external"))
    grader = make_grader()
    reader = Reader("Supported " + " ".join(f"[source: {source}]" for source in expected_ids))
    engine = CorrectiveRAGEngine(retriever=primary, llm_client=reader, retrieval_grader=grader,
                                knowledge_refiner=SemanticKnowledgeRefiner(grader, sentences_per_strip=1),
                                external_retriever=external, preserve_correction_sources=True, max_retries=0)
    response = engine.ask("query", top_k=1)
    assert response.metadata["retrieval_action"] == action
    assert [doc.doc_id for doc in response.retrieved_documents] == expected_ids
    assert "noise unrelated" not in reader.prompts[0]
    assert len(external.calls) == (0 if action == "correct" else 1)
    assert response.citation_validation.ok
    assert all(doc.metadata["corrective_provenance"]["refinement"]["strategy"] == "SemanticKnowledgeRefiner"
               for doc in response.retrieved_documents)


def test_corrective_engine_preserves_base_context_packer():
    from cheragh.context_packing import LongContextPacker

    class RecordingPacker(LongContextPacker):
        def __init__(self):
            super().__init__(token_budget=100)
            self.calls = 0

        def pack(self, documents):
            self.calls += 1
            return super().pack(documents)

    packer = RecordingPacker()
    base = RAGEngine(retriever=FixedRetriever(Document("cardiac source", doc_id="a")),
                     llm_client=Reader("Supported [source: a]"), context_packer=packer)
    CorrectiveRAGEngine(base_engine=base, retrieval_grader=make_grader(), max_retries=0).ask("query")
    assert packer.calls == 1


@pytest.mark.skipif(not os.environ.get("CHERAGH_CRAG_MODEL_PATH"), reason="set a local pretrained cross-encoder path")
def test_pretrained_cross_encoder_cpu_calibration_and_refinement():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    grader = CrossEncoderRetrievalGrader(os.environ["CHERAGH_CRAG_MODEL_PATH"], local_files_only=True)
    pairs = [
        ("What is the capital of France?", Document("Paris is the capital of France.")),
        ("What is the capital of France?", Document("Cats are mammals and enjoy sleeping.")),
        ("Who wrote Hamlet?", Document("William Shakespeare wrote the tragedy Hamlet.")),
        ("Who wrote Hamlet?", Document("Bananas are yellow when ripe.")),
        ("How do plants get energy?", Document("Plants use photosynthesis to convert sunlight into chemical energy.")),
        ("How do plants get energy?", Document("The train departed from London at noon.")),
    ]
    grader.calibrate(pairs, [1, 0, 1, 0, 1, 0])
    # Mechanism smoke, not a dataset benchmark: the calibration pairs and the
    # test question/document are distinct, but this small set is not sufficient
    # to establish confidence calibration in any production domain.
    query = "What is the capital of Japan?"
    relevant = Document("Tokyo is the capital of Japan.", doc_id="relevant")
    noise = Document("A bicycle has two wheels and handlebars.", doc_id="noise")
    scores = grader.score_documents(query, [relevant, noise])
    assert scores[0] > 0.9 and scores[1] < 0.1
    assert grader.grade(query, [relevant, noise]).action is RetrievalAction.CORRECT
    mixed = Document(relevant.content + " " + noise.content, doc_id="mixed")
    refined = SemanticKnowledgeRefiner(grader, sentences_per_strip=1).refine(query, [mixed])
    assert refined[0].content == relevant.content
    assert refined[0].doc_id == "mixed"
