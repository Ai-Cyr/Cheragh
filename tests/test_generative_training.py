"""Real CPU language-model SFT and independently checked supervision masks."""
from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace

import pytest

from cheragh.base import Document
from cheragh.training.data import RAFTTrainingRecord, RetrievalTrainingExample
from cheragh.training.generative import (
    GenerativeTrainingExample as Example,
    RankRAGDatasetBuilder as Builder,
    RankRAGEngine,
    RankRAGModel,
    TransformersGenerativeTrainer,
)


HAS_MODELS = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
requires_models = pytest.mark.skipif(not HAS_MODELS, reason="torch and transformers are optional")


def raft_record(included=True):
    oracle = Document("Paris is the capital of France.", doc_id="oracle")
    distractor = Document("Saturn has rings.", doc_id="noise")
    return RAFTTrainingRecord("Capital of France?", "Paris", (oracle, distractor) if included else (distractor,),
                              ("oracle",), included, rationale="Evidence: ##begin_quote##Paris is the capital of France.##end_quote##",
                              oracle_documents=(oracle,))


def test_raft_dropout_changes_only_inputs_and_supervision_oracle_never_leaks():
    retained, dropped = Example.from_raft(raft_record()), Example.from_raft(raft_record(False))
    assert "Paris is the capital of France." in retained.prompt
    assert "Paris" not in dropped.prompt
    assert retained.target == dropped.target
    assert "##begin_quote##Paris is the capital of France.##end_quote##" in dropped.target
    assert dropped.target.endswith("##Answer: Paris")
    assert dropped.metadata["oracle_included"] is False
    assert dropped.metadata["context_doc_ids"] == ["noise"]


def test_raft_revalidates_mutable_documents_and_rejects_duplicate_content_distractor():
    record = raft_record()
    record.documents[0].content = "tampered evidence"
    with pytest.raises(ValueError, match="oracle content"):
        Example.from_raft(record)
    record = raft_record(False)
    record.documents[0].content = "Paris   is the capital of France."
    with pytest.raises(ValueError, match="duplicate oracle content"):
        Example.from_raft(record)
    record = raft_record(False)
    record.oracle_documents[0].content = "evidence removed"
    with pytest.raises(ValueError, match="verbatim"):
        Example.from_raft(record)


def test_raft_answer_only_and_no_distractor_ablation_require_explicit_opt_out():
    oracle = Document("gold", doc_id="oracle")
    record = RAFTTrainingRecord("q", "answer", (oracle,), ("oracle",), True)
    with pytest.raises(ValueError, match="quoted rationale"):
        Example.from_raft(record)
    with pytest.raises(ValueError, match="distractor"):
        Example.from_raft(record, require_rationale=False)
    assert Example.from_raft(record, require_rationale=False, require_distractors=False).target == "answer"


def test_rankrag_prepares_binary_ranking_listwise_and_two_qa_tasks_from_annotations():
    positive = Document("answer", doc_id="good")
    negative = Document("unrelated", doc_id="bad")
    data = Builder.from_retrieval_example(RetrievalTrainingExample("q", (positive,), (negative,), "answer"), seed=4)
    assert [item.task for item in data] == ["context_ranking", "context_ranking", "retrieval_ranking", "context_qa", "retrieval_qa"]
    assert [item.target for item in data[:2]] == ["True", "False"]
    index = data[2].metadata["doc_ids"].index("good") + 1
    assert data[2].target == str(index)
    assert data[3].target == data[4].target == "answer"
    assert "unrelated" not in data[3].prompt and "unrelated" in data[4].prompt
    positive.content = "mutated"
    assert "mutated" not in data[0].prompt


def test_rankrag_relevance_is_explicit_and_doc_id_conflicts_are_rejected():
    doc = Document("x", doc_id="same")
    with pytest.raises(TypeError, match="annotation"):
        Builder.relevance("q", doc, 1)
    with pytest.raises(ValueError, match="unique"):
        Builder.qa("q", [doc, Document("other", doc_id="same")], "answer")
    with pytest.raises(ValueError, match="present"):
        Builder.passage_ranking("q", [doc], ["missing"])
    assert Builder.passage_ranking("q", [doc], []).target == "None"
    with pytest.raises(ValueError, match="identical content"):
        Builder.from_retrieval_example(RetrievalTrainingExample("q", (doc,), (Document("x", doc_id="other"),)))


def test_weighted_blend_is_reproducible_and_zero_weight_excludes_a_task():
    instruction = Example("q", "a")
    rank = Builder.relevance("q", Document("evidence"), True)
    datasets = {"instruction": [instruction], "ranking": [rank]}
    assert Builder.blend(datasets, {"instruction": 0, "ranking": 4}, sample_count=5) == [rank] * 5
    a = Builder.blend(datasets, {"instruction": 2, "ranking": 1}, sample_count=30, seed=5)
    b = Builder.blend(datasets, {"instruction": 2, "ranking": 1}, sample_count=30, seed=5)
    assert a == b and {item.task for item in a} == {"instruction", "context_ranking"}
    with pytest.raises(ValueError, match="positive"):
        Builder.blend(datasets, {"instruction": 0, "ranking": 0}, sample_count=5)


def tiny_generator(mode, examples, *, eos_as_pad=False):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

    tokens = ["[PAD]", "[EOS]", "[UNK]", "[BOS]", "True", "False"]
    tokens += sorted(set(" ".join([part for item in examples for part in (item.prompt, item.target)]).split()) - set(tokens))
    vocab = {token: index for index, token in enumerate(tokens)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if mode == "seq2seq":
        backend.post_processor = processors.TemplateProcessing(single="$A [EOS]", special_tokens=[("[EOS]", 1)])
        model = T5ForConditionalGeneration(T5Config(vocab_size=len(tokens), d_model=24, d_kv=12, d_ff=48,
                                                   num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0.,
                                                   pad_token_id=1 if eos_as_pad else 0, eos_token_id=1, decoder_start_token_id=0))
    else:
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokens), n_embd=24, n_layer=1, n_head=2,
                                          n_positions=256, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
                                          pad_token_id=1 if eos_as_pad else 0, eos_token_id=1, bos_token_id=3))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[EOS]" if eos_as_pad else "[PAD]",
                                       eos_token="[EOS]", unk_token="[UNK]", bos_token="[BOS]")
    return TransformersGenerativeTrainer(model, tokenizer, max_input_tokens=192, max_target_tokens=64)


@requires_models
@pytest.mark.parametrize("mode", ["causal", "seq2seq"])
def test_supervision_masks_prompts_and_padding_but_preserves_real_eos_when_pad_equals_eos(mode):
    import torch

    examples = [Example("question alpha", "red"), Example("question beta extra", "blue green")]
    trainer = tiny_generator(mode, examples, eos_as_pad=True)
    batch = trainer.collate(examples)
    eos = trainer.tokenizer.eos_token_id
    expected_tokens = [trainer.tokenizer.encode(item.target, add_special_tokens=False) + [eos] for item in examples]
    assert [row[row != -100].tolist() for row in batch["labels"]] == expected_tokens
    assert batch["input_ids"][0, -1].item() == eos  # padded shorter row
    assert batch["attention_mask"][0, -1].item() == 0
    assert batch["labels"][0, -1].item() == -100
    if mode == "causal":
        assert batch["labels"][0, :3].tolist() == [-100, -100, -100]  # BOS + two prompt tokens
        assert batch["labels"][1, :4].tolist() == [-100] * 4
    with torch.inference_mode():
        output = trainer.model(**batch)
        logits, labels = output.logits, batch["labels"]
        if mode == "causal":
            logits, labels = logits[:, :-1], labels[:, 1:]
        manual_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    loss, count = trainer._loss(examples)
    assert loss.item() == pytest.approx(manual_loss.item())
    assert count == 5  # red+EOS, blue+green+EOS


@requires_models
@pytest.mark.parametrize("mode", ["causal", "seq2seq"])
def test_real_cpu_gradient_learning_generation_and_checkpoint_roundtrip(mode, tmp_path):
    import torch

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        examples = [Example("question alpha", "red"), Example("question beta", "blue")]
        trainer = tiny_generator(mode, examples)
        before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
        report = trainer.fit(examples, epochs=80, batch_size=2, learning_rate=.01, seed=5)
        assert report.steps == 80 and report.supervised_tokens == 320
        assert report.final_loss < report.initial_loss * .1
        assert any(not torch.equal(before[name], parameter) for name, parameter in trainer.model.named_parameters())
        assert [trainer.generate(item.prompt, max_new_tokens=4) for item in examples] == ["red", "blue"]
        expected = trainer.next_token_probabilities([item.prompt for item in examples], "red")
        destination = tmp_path / mode
        trainer.save_pretrained(destination)
        restored = TransformersGenerativeTrainer.from_pretrained(
            destination, model_kwargs={"local_files_only": True}, tokenizer_kwargs={"local_files_only": True},
        )
        assert restored.training_steps == 80
        assert restored.mode == mode and restored.max_input_tokens == 192
        assert restored.next_token_probabilities([item.prompt for item in examples], "red") == pytest.approx(expected, abs=1e-7)
        assert [restored.generate(item.prompt, max_new_tokens=4) for item in examples] == ["red", "blue"]
        assert (destination / "model.safetensors").exists()
        assert json.loads((destination / "cheragh_generative.json").read_text())["add_bos_token"] is True
    finally:
        torch.set_num_threads(old_threads)


@requires_models
def test_raft_record_runs_real_sft_with_quoted_target_and_without_withheld_oracle():
    import torch

    torch.manual_seed(4)
    record = raft_record(False)
    example = Example.from_raft(record)
    trainer = tiny_generator("causal", [example])
    loss_before = trainer._loss([example])[0].item()
    report = trainer.fit([record], max_steps=2, epochs=8, batch_size=1, learning_rate=.01)
    assert report.steps == trainer.training_steps == 2
    assert report.final_loss < loss_before
    batch = trainer.collate([example])
    target = trainer.tokenizer.decode(batch["labels"][0][batch["labels"][0] != -100], skip_special_tokens=True)
    assert "##Reason:" in target and "##Answer:" in target


@requires_models
def test_overlong_target_or_late_input_is_rejected_before_any_parameter_update():
    import torch

    data = [Example("short", "answer"), Example("short short short short", "answer")]
    trainer = tiny_generator("causal", data)
    trainer.max_input_tokens = 3
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    with pytest.raises(ValueError, match="max_input_tokens"):
        trainer.fit(data, batch_size=1)
    assert trainer.training_steps == 0
    assert all(torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))
    trainer.max_input_tokens = 10
    trainer.max_target_tokens = 1
    with pytest.raises(ValueError, match="max_target_tokens"):
        trainer.collate(data[:1])
    trainer.max_target_tokens = 64
    trainer.max_input_tokens = 500
    with pytest.raises(ValueError, match="context window"):
        trainer.collate([Example("short " * 255, "answer")])


@requires_models
@pytest.mark.parametrize("mode", ["causal", "seq2seq"])
def test_rank_score_matches_full_vocabulary_probability_not_binary_renormalization(mode):
    import torch

    trainer = tiny_generator(mode, [Example("question", "True")])
    ids = trainer._prompt_ids("question")
    kwargs = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
    if mode == "seq2seq":
        kwargs["decoder_input_ids"] = torch.tensor([[0]])
    with torch.inference_mode():
        logits = trainer.model(**kwargs).logits[0, -1]
    true_id, false_id = trainer.tokenizer.encode("True", add_special_tokens=False)[0], trainer.tokenizer.encode("False", add_special_tokens=False)[0]
    actual = trainer.next_token_probabilities(["question"], "True")[0]
    assert actual == pytest.approx(torch.softmax(logits.float(), -1)[true_id].item())
    assert actual < torch.softmax(logits.float()[[true_id, false_id]], -1)[0].item()
    with pytest.raises(ValueError, match="one non-unknown"):
        trainer.next_token_probabilities(["question"], "True False")


@requires_models
def test_rankrag_joint_training_then_real_same_model_reranking_and_answer_generation():
    import torch

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(2)
        good, bad = Document("gold evidence", doc_id="good"), Document("irrelevant noise", doc_id="bad")
        query = "which word?"
        data = [Builder.relevance(query, good, True), Builder.relevance(query, bad, False), Builder.qa(query, [good], "gold")]
        trainer = tiny_generator("causal", data)
        report = trainer.fit(data, epochs=120, batch_size=3, learning_rate=.008)
        assert report.final_loss < report.initial_loss * .1
        model = RankRAGModel(trainer)
        ranked = model.rerank(query, [bad, good], top_k=2)
        assert [doc.doc_id for doc in ranked] == ["good", "bad"]
        assert ranked[0].score > .8 and ranked[1].score < .2
        calls = []

        def retrieve(question, top_k):
            calls.append((question, top_k))
            return [bad, good, Document("extra", doc_id="extra")]

        engine = RankRAGEngine(SimpleNamespace(retrieve=retrieve), model, candidate_top_k=2, top_k=1, max_new_tokens=4)
        result = engine.ask(query)
        assert calls == [(query, 2)]
        assert result.answer == "gold"
        assert [source.doc_id for source in result.sources] == ["good"]
        assert result.metadata["candidate_count"] == 2
        assert result.metadata["selected_count"] == 1
        assert "gold evidence" in result.prompt and "irrelevant noise" not in result.prompt
        assert result.citations == []  # conditioned evidence does not prove an inline citation
        assert good.score is None and bad.score is None
        with pytest.raises(ValueError, match="candidate_top_k"):
            engine.ask(query, top_k=3)
    finally:
        torch.set_num_threads(old_threads)
