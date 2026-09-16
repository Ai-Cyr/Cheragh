"""Outcome-derived labels and real CPU classifier training, without downloads."""
from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace

import pytest

from cheragh.adaptive import AdaptiveRAGEngine, AdaptiveRAGRoute as Route
from cheragh.adaptive_learning import (
    AdaptiveSilverDatasetBuilder,
    AdaptiveStrategyOutcome,
    AdaptiveTrainingExample,
    AdaptiveTrainingQuestion,
    TransformersComplexityClassifier,
)
from cheragh.schema import RAGResponse


ROUTES = (Route.NO_RETRIEVAL, Route.SINGLE_STEP, Route.ITERATIVE)
HAS_MODELS = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


def outcomes(correct):
    return [AdaptiveStrategyOutcome(route, "gold" if flag else "wrong", flag, float(index))
            for index, (route, flag) in enumerate(zip(ROUTES, correct))]


@pytest.mark.parametrize("correct, expected", [
    ((True, True, True), Route.NO_RETRIEVAL),
    ((False, True, True), Route.SINGLE_STEP),
    ((False, False, True), Route.ITERATIVE),
    ((False, True, False), Route.SINGLE_STEP),
])
def test_silver_label_chooses_simplest_correct_strategy(correct, expected):
    question = AdaptiveTrainingQuestion("question", ("gold",), dataset_route=Route.ITERATIVE)
    result = AdaptiveSilverDatasetBuilder.from_outcomes(question, outcomes(correct))
    assert result.route == expected
    assert result.label == {Route.NO_RETRIEVAL: "A", Route.SINGLE_STEP: "B", Route.ITERATIVE: "C"}[expected]
    assert result.label_source == "successful_outcome"
    assert [item.correct for item in result.outcomes] == list(correct)


def test_all_failure_is_unlabeled_unless_dataset_prior_was_explicitly_supplied():
    assert AdaptiveSilverDatasetBuilder.from_outcomes(AdaptiveTrainingQuestion("q", ("gold",)), outcomes([False] * 3)) is None
    for route in (Route.SINGLE_STEP, Route.ITERATIVE):
        result = AdaptiveSilverDatasetBuilder.from_outcomes(
            AdaptiveTrainingQuestion("q", ("gold",), dataset_route=route), outcomes([False] * 3),
        )
        assert result.route == route
        assert result.label_source == "dataset_prior"


def test_builder_executes_and_evaluates_all_actual_outputs_and_custom_costs():
    calls, evaluated = [], []

    def engine(route, answer, cost):
        def ask(query):
            calls.append((route, query))
            return SimpleNamespace(response=SimpleNamespace(answer=answer), tokens=cost)
        return SimpleNamespace(ask=ask)

    strategies = {ROUTES[0]: engine(ROUTES[0], "incorrect", 1), ROUTES[1]: engine(ROUTES[1], "Paris", 20),
                  ROUTES[2]: engine(ROUTES[2], "PARIS", 10)}

    def evaluator(answer, references):
        evaluated.append(answer)
        return answer.lower() == references[0].lower()

    builder = AdaptiveSilverDatasetBuilder(strategies, evaluator=evaluator,
                                          cost_fn=lambda route, output, elapsed: output.tokens)
    result = builder.collect(AdaptiveTrainingQuestion("capital?", ("Paris",), example_id="q-42"))
    assert calls == [(route, "capital?") for route in ROUTES]
    assert evaluated == ["incorrect", "Paris", "PARIS"]
    assert result.route == Route.ITERATIVE, "measured cost, not a hardcoded complexity preference, breaks successful ties"
    assert [item.cost for item in result.outcomes] == [1, 20, 10]
    assert all(item.elapsed_seconds >= 0 for item in result.outcomes)
    assert json.loads(json.dumps(result.to_dict()))["example_id"] == "q-42"


def test_cost_ties_use_complexity_order_independently_of_outcome_order():
    observations = [AdaptiveStrategyOutcome(route, "answer", True, 1.) for route in reversed(ROUTES)]
    result = AdaptiveSilverDatasetBuilder.from_outcomes(AdaptiveTrainingQuestion("q", ("answer",)), observations)
    assert result.route == Route.NO_RETRIEVAL


def test_incomplete_duplicate_or_contradictory_training_provenance_is_rejected():
    question = AdaptiveTrainingQuestion("question", ("gold",))
    for invalid in (outcomes([True] * 3)[:2], outcomes([True] * 3)[:2] + outcomes([True] * 3)[:1]):
        with pytest.raises(ValueError, match="each route"):
            AdaptiveSilverDatasetBuilder.from_outcomes(question, invalid)
    with pytest.raises(ValueError, match="least-cost"):
        AdaptiveTrainingExample(question, Route.ITERATIVE, tuple(outcomes([True] * 3)))
    with pytest.raises(ValueError, match="dataset prior"):
        AdaptiveTrainingExample(question, Route.SINGLE_STEP, tuple(outcomes([False] * 3)), "dataset_prior")


@pytest.mark.parametrize("cost", [True, float("inf"), float("nan"), -1])
def test_invalid_outcome_costs_cannot_influence_training_labels(cost):
    with pytest.raises((TypeError, ValueError)):
        AdaptiveStrategyOutcome(Route.SINGLE_STEP, "answer", True, cost)


def tiny_classifier(mode):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

    tokens = ["[PAD]", "[EOS]", "[UNK]", "[CLS]", "[SEP]", "A", "B", "C", "alpha", "beta", "gamma", "question"]
    vocab = {token: index for index, token in enumerate(tokens)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if mode == "seq2seq":
        backend.post_processor = processors.TemplateProcessing(single="$A [EOS]", special_tokens=[("[EOS]", 1)])
        model = T5ForConditionalGeneration(T5Config(vocab_size=len(tokens), d_model=16, d_kv=8, d_ff=32,
                                                   num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0.,
                                                   pad_token_id=0, eos_token_id=1, decoder_start_token_id=0))
    else:
        backend.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
                                                               special_tokens=[("[CLS]", 3), ("[SEP]", 4)])
        model = BertForSequenceClassification(BertConfig(vocab_size=len(tokens), hidden_size=16, intermediate_size=32,
                                                         num_hidden_layers=1, num_attention_heads=2, num_labels=3,
                                                         max_position_embeddings=32, hidden_dropout_prob=0.,
                                                         attention_probs_dropout_prob=0., classifier_dropout=0., pad_token_id=0))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]",
                                       unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")
    return TransformersComplexityClassifier(model, tokenizer, mode=mode, max_input_tokens=24, model_id="tiny-test")


def training_examples():
    result = []
    for name, route in zip(("alpha", "beta", "gamma"), ROUTES):
        question = AdaptiveTrainingQuestion(name + " question", ("gold",))
        result.append(AdaptiveSilverDatasetBuilder.from_outcomes(question, outcomes([candidate == route for candidate in ROUTES])))
    return result


@pytest.mark.skipif(not HAS_MODELS, reason="torch and transformers are optional")
def test_seq2seq_label_supervision_keeps_real_eos_when_it_is_also_padding():
    classifier = tiny_classifier("seq2seq")
    classifier.tokenizer.pad_token = classifier.tokenizer.eos_token
    classifier.model.config.pad_token_id = classifier.tokenizer.eos_token_id
    labels = []

    def capture(module, args, kwargs):
        labels.append(kwargs["labels"].detach().clone())

    handle = classifier.model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        classifier._loss(training_examples())
    finally:
        handle.remove()
    assert labels[0][:, -1].tolist() == [classifier.tokenizer.eos_token_id] * 3


@pytest.mark.skipif(not HAS_MODELS, reason="torch and transformers are optional")
@pytest.mark.parametrize("mode", ["seq2seq", "sequence_classification"])
def test_real_cpu_classifier_learns_and_preserves_predictions_after_save_reload(mode, tmp_path):
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(23 if mode == "seq2seq" else 0)
        classifier = tiny_classifier(mode)
        data = training_examples()
        before = {name: parameter.detach().clone() for name, parameter in classifier.model.named_parameters()}
        epochs = 100 if mode == "seq2seq" else 150
        report = classifier.fit(data, epochs=epochs, batch_size=3,
                                learning_rate=.02 if mode == "seq2seq" else .003, seed=7)
        assert report.steps == epochs
        assert report.final_loss < report.initial_loss * .25
        assert report.training_accuracy == 1.
        assert any(not torch.equal(before[name], parameter) for name, parameter in classifier.model.named_parameters())
        assert [classifier.classify(item.query).route for item in data] == list(ROUTES)
        expected = classifier.predict_proba([item.query for item in data])

        destination = tmp_path / mode
        classifier.save_pretrained(destination)
        restored = TransformersComplexityClassifier.from_pretrained(
            destination, model_kwargs={"local_files_only": True}, tokenizer_kwargs={"local_files_only": True},
        )
        assert restored.training_steps == epochs
        assert restored.max_input_tokens == 24
        for left, right in zip(expected, restored.predict_proba([item.query for item in data])):
            assert left == pytest.approx(right, abs=1e-7)
        assert (destination / "model.safetensors").exists()

        # Exercise the public AdaptiveRAGEngine contract with the actual
        # learned classifier, not a scripted stand-in classifier.
        calls = []
        llm = SimpleNamespace(generate=lambda prompt, **kwargs: "direct")

        def answer(route):
            def ask(query, **kwargs):
                calls.append(route)
                return RAGResponse(query=query, answer=route.value, sources=[], retrieved_documents=[], prompt="")
            return SimpleNamespace(ask=ask)

        engine = AdaptiveRAGEngine(answer(Route.SINGLE_STEP), iterative_engine=answer(Route.ITERATIVE),
                                   llm_client=llm, classifier=restored)
        assert engine.ask("alpha question").answer == "direct"
        assert engine.ask("beta question").answer == "single_step"
        assert engine.ask("gamma question").answer == "iterative"
        assert calls == [Route.SINGLE_STEP, Route.ITERATIVE]
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.skipif(not HAS_MODELS, reason="torch and transformers are optional")
def test_t5_route_probabilities_equal_first_decoder_step_logits_conditioned_on_abc():
    import torch

    torch.manual_seed(5)
    classifier = tiny_classifier("seq2seq")
    encoded = classifier.tokenizer(["alpha question"], return_tensors="pt")
    with torch.inference_mode():
        scores = classifier.model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
                                  decoder_input_ids=torch.tensor([[0]])).logits[0, 0]
        expected = torch.softmax(scores[[5, 6, 7]], dim=-1).tolist()
    actual = classifier.predict_proba(["alpha question"])[0]
    assert list(actual.values()) == pytest.approx(expected)
    assert classifier.classify("alpha question").confidence == pytest.approx(max(expected))


@pytest.mark.skipif(not HAS_MODELS, reason="torch and transformers are optional")
def test_training_step_bound_and_nonfinite_inference_fail_explicitly():
    import torch

    classifier = tiny_classifier("sequence_classification")
    report = classifier.fit(training_examples(), epochs=50, batch_size=1, max_steps=2)
    assert report.steps == classifier.training_steps == 2
    assert len(report.epoch_losses) == 1
    with torch.no_grad():
        classifier.model.classifier.bias[0] = float("nan")
    with pytest.raises(ValueError, match="finite logits"):
        classifier.classify("question")
