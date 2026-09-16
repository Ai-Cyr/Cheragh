"""Semantic judge contracts, real tiny models, and opt-in pretrained smoke."""
import inspect
import json
import os
from types import SimpleNamespace

import pytest

from cheragh import Document
from cheragh.base import CallableLLMClient, StaticLLMClient
from cheragh.evaluation.claims import ClaimEvaluator, ClaimStatus
from cheragh.evaluation.semantic import LLMClaimSegmenter, LLMFaithfulnessJudge, NLIFaithfulnessJudge


def _segment_response(**changes):
    item = {"text": "Paris is in France.", "source_text": "Paris is in France [source: atlas].",
            "citations": ["atlas"]}
    item.update(changes)
    return json.dumps({"claims": [item]})


def test_llm_decomposition_keeps_atomic_claims_and_source_attribution():
    answer = "Paris is in France and has 2 million residents [source: atlas]."
    response = {"claims": [
        {"text": "Paris is in France.", "source_text": answer, "citations": ["atlas"]},
        {"text": "Paris has 2 million residents.", "source_text": answer, "citations": ["atlas"]},
    ]}
    segmenter = LLMClaimSegmenter(StaticLLMClient(json.dumps(response)))
    result = segmenter.segment(answer)
    assert len(result) == 2
    assert all(claim.citations == ("atlas",) for claim in result)
    assert result[1].text == "Paris has 2 million residents."
    assert segmenter.segment(" ") == []


@pytest.mark.parametrize("response", [
    _segment_response(citations=[]),
    _segment_response(citations=["invented"]),
    _segment_response(source_text="Paris is in France [source: invented].", citations=["invented"]),
    _segment_response(text="Paris is in France [source: atlas]."),
    '{"claims": [], "claims": []}',
    '{"claims": NaN}',
    '```json\n{"claims": []}\n```',
    '{"claims": "not-an-array"}',
])
def test_llm_segmenter_rejects_unanchored_citations_and_malformed_json(response):
    with pytest.raises(ValueError):
        LLMClaimSegmenter(StaticLLMClient(response)).segment("Paris is in France [source: atlas].")


def _verdict(verdict="supported", supporting=None, contradicting=None):
    return {"verdict": verdict, "supporting_quotes": supporting or [],
            "contradicting_quotes": contradicting or [], "rationale": "Checked against this source."}


def test_llm_judge_checks_tail_and_retains_conflicts_with_source_offsets():
    content = "Alpha exists. " + "irrelevant. " * 8 + "Alpha does not exist."
    seen = []

    def generate(prompt, **kwargs):
        item = json.loads(prompt.split("PAIR_JSON:\n")[1])
        seen.append(item["evidence"])
        assert kwargs["temperature"] == 0.0
        if "Alpha does not exist." in item["evidence"]:
            return json.dumps(_verdict("contradicted", contradicting=["Alpha does not exist."]))
        if "Alpha exists." in item["evidence"]:
            return json.dumps(_verdict(supporting=["Alpha exists."]))
        return json.dumps(_verdict("unsupported"))

    judge = LLMFaithfulnessJudge(CallableLLMClient(generate), max_evidence_chars=50, overlap_chars=25)
    score = judge.score("Alpha exists.", Document(content))
    assert score.entailment == score.contradiction == 1.0
    windows = json.loads(score.rationale)
    assert len(windows) == len(seen) > 1
    covered = set()
    for window in windows:
        covered.update(range(window["start"], window["end"]))
    assert covered == set(range(len(content)))


@pytest.mark.parametrize("response", [
    _verdict(supporting=["Invented supporting evidence."]),
    _verdict(),
    _verdict("unsupported", supporting=["Paris is in France."]),
    _verdict("conflicting", supporting=["Paris is in France."]),
    _verdict("invalid"),
])
def test_llm_judge_requires_grounded_quotes_matching_verdict(response):
    judge = LLMFaithfulnessJudge(StaticLLMClient(json.dumps(response)))
    with pytest.raises(ValueError):
        judge.score("Paris is in France.", Document("Paris is in France."))


def test_semantic_judge_integrates_with_citation_alignment():
    judge = LLMFaithfulnessJudge(StaticLLMClient(json.dumps(
        _verdict("contradicted", contradicting=["Paris is in France."]),
    )))
    result = ClaimEvaluator(scorer=judge).evaluate(
        "Paris is in Germany [source: atlas].", [Document("Paris is in France.", doc_id="atlas")],
    )
    assert result.diagnostics[0].status is ClaimStatus.CONTRADICTED
    assert result.diagnostics[0].citation_alignments[0].contradicted


@pytest.fixture
def nli_runtime(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "cat", "dog", "exists", "good", "bad", "noise", "."]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab), encoding="utf-8")
    if "vocab" in inspect.signature(transformers.BertTokenizer).parameters:
        tokenizer = transformers.BertTokenizer(vocab={token: index for index, token in enumerate(vocab)})
    else:
        tokenizer = transformers.BertTokenizerFast(vocab_file=str(path))
    return torch, transformers, tokenizer


def _nli_config(transformers, tokenizer, **kwargs):
    return transformers.BertConfig(
        vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, max_position_embeddings=64,
        hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        num_labels=3, **kwargs,
    )


def test_nli_windows_cover_late_evidence_preserve_claim_and_bound_batches(nli_runtime):
    torch, _, tokenizer = nli_runtime

    class EvidenceNLI(torch.nn.Module):
        config = SimpleNamespace(id2label={0: "neutral", 1: "contradiction", 2: "entailment"}, num_labels=3)

        def __init__(self):
            super().__init__()
            self.batches = []

        def forward(self, input_ids, attention_mask, token_type_ids):
            self.batches.append((input_ids.clone(), token_type_ids.clone()))
            logits = torch.zeros((len(input_ids), 3))
            for i, row in enumerate(input_ids):
                # Role-sensitive: only premise tokens determine the verdict.
                premise = row[token_type_ids[i] == 0].tolist()
                logits[i, 0] = 2
                if tokenizer.convert_tokens_to_ids("good") in premise:
                    logits[i, 2] = 10
                if tokenizer.convert_tokens_to_ids("bad") in premise:
                    logits[i, 1] = 10
            return SimpleNamespace(logits=logits)

    model = EvidenceNLI()
    judge = NLIFaithfulnessJudge(model=model, tokenizer=tokenizer, max_length=12, overlap_tokens=2, batch_size=2)
    score = judge.score("cat exists", Document("good " + "noise " * 20 + "bad"))
    assert score.entailment > .99 and score.contradiction > .99
    assert len(model.batches) > 1 and all(len(ids) <= 2 for ids, _ in model.batches)
    hypothesis = tokenizer("cat exists", add_special_tokens=False)["input_ids"] + [tokenizer.sep_token_id]
    for input_ids, types in model.batches:
        for row, roles in zip(input_ids, types, strict=True):
            assert row[roles == 1].tolist() == hypothesis
    assert json.loads(score.rationale)["windows"] > 2


def test_nli_loads_actual_local_checkpoint_and_matches_independent_softmax(tmp_path, nli_runtime):
    torch, transformers, tokenizer = nli_runtime
    config = _nli_config(transformers, tokenizer, id2label={0: "entailment", 1: "neutral", 2: "contradiction"})
    model = transformers.BertForSequenceClassification(config).eval()
    path = tmp_path / "nli"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    judge = NLIFaithfulnessJudge(str(path), model_kwargs={"local_files_only": True}, max_length=32)
    score = judge.score("cat exists", Document("dog exists"))
    with torch.inference_mode():
        expected = model(**tokenizer("dog exists", "cat exists", return_tensors="pt")).logits.softmax(dim=-1)[0]
    assert score.entailment == pytest.approx(float(expected[0]), abs=1e-7)
    assert score.contradiction == pytest.approx(float(expected[2]), abs=1e-7)
    assert not judge.model.training


def test_nli_requires_explicit_label_order_for_ambiguous_models(nli_runtime):
    _, transformers, tokenizer = nli_runtime
    model = transformers.BertForSequenceClassification(_nli_config(transformers, tokenizer))
    with pytest.raises(ValueError, match="explicit label_mapping"):
        NLIFaithfulnessJudge(model=model, tokenizer=tokenizer)
    judge = NLIFaithfulnessJudge(model=model, tokenizer=tokenizer,
                               label_mapping={"entailment": 2, "contradiction": 0, "neutral": 1})
    assert judge.label_mapping["entailment"] == 2


def test_nli_rejects_long_claim_without_silent_truncation(nli_runtime):
    _, transformers, tokenizer = nli_runtime
    model = transformers.BertForSequenceClassification(_nli_config(
        transformers, tokenizer, id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
    ))
    judge = NLIFaithfulnessJudge(model=model, tokenizer=tokenizer, max_length=8)
    with pytest.raises(ValueError, match="claim exceeds"):
        judge.score("cat " * 10, Document("cat exists"))
    assert judge.score("cat", Document("")).entailment == 0


@pytest.mark.skipif(not os.environ.get("CHERAGH_REAL_MODELS_CACHE"), reason="opt-in cached pretrained model smoke")
def test_pretrained_nli_supports_paraphrase_and_detects_contradiction():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    judge = NLIFaithfulnessJudge(model_kwargs={"cache_dir": os.environ["CHERAGH_REAL_MODELS_CACHE"],
                                             "local_files_only": True})
    support = judge.score("A man eats something.", Document("A man is eating pizza."))
    contradiction = judge.score("No man is eating anything.", Document("A man is eating pizza."))
    assert support.entailment > .8
    assert contradiction.contradiction > .8
