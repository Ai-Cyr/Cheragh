"""Full-logit conformance and an offline tiny Llama integration (optional deps)."""
from __future__ import annotations

import importlib.util
import math
from types import SimpleNamespace

import pytest

from cheragh.self_rag import ReflectionTokenGroup as Group, TransformersSelfRAGDecoder


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")

TOKENS = [
    "[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]",
    "[Relevant]", "[Irrelevant]", "[Fully supported]", "[Partially supported]",
    "[No support / Contradictory]", *[f"[Utility:{index}]" for index in range(1, 6)],
]


class TinyTokenizer:
    """Atomic reflection vocabulary; ordinary test text is one vocabulary item."""

    eos_token_id = 0
    unk_token_id = 31
    ids = {token: index for index, token in enumerate(TOKENS, start=1)}

    def encode(self, text, add_special_tokens=False):
        if text in self.ids:
            return [self.ids[text]]
        return [14] + [15] * len(text.split()) if add_special_tokens else [15] * len(text.split())

    def decode(self, ids, skip_special_tokens=False, **_):
        reverse = {value: key for key, value in self.ids.items()}
        return "".join(reverse[token] if token in reverse else "fact" for token in ids
                       if not skip_special_tokens or token not in reverse)


def scripted_model(rows, *, cache=True):
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(vocab_size=32, max_position_embeddings=64)
            self.generation_config = SimpleNamespace(eos_token_id=0)
            self.calls = []

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True):
            self.calls.append((input_ids.shape[1], attention_mask.shape[1]))
            index = len(self.calls) - 1
            logits = torch.full((1, input_ids.shape[1], 32), -20.)
            for token_id, probability in rows[min(index, len(rows) - 1)].items():
                logits[0, -1, token_id] = math.log(probability)
            return SimpleNamespace(logits=logits, past_key_values=(index,) if cache and use_cache else None)

    return Model()


@pytest.mark.parametrize("cache", [False, True])
def test_decoder_reads_reflection_groups_at_their_actual_positions(cache):
    ids = TinyTokenizer.ids
    rows = [
        {ids["[Relevant]"]: .8, ids["[Irrelevant]"]: .2},
        {15: 1.},
        {ids["[Fully supported]"]: .6, ids["[Partially supported]"]: .3, ids["[No support / Contradictory]"]: .1},
        {ids[f"[Utility:{index}]"]: .6 if index == 5 else .1 for index in range(1, 6)},
        {ids["[Retrieval]"]: .2, ids["[No Retrieval]"]: .2, ids["[Continue to Use Evidence]"]: .6},
    ]
    model = scripted_model(rows, cache=cache)
    result = TransformersSelfRAGDecoder(model, TinyTokenizer(), model_id="tiny-fixture").decode_segment(
        "prompt", max_new_tokens=10,
    )
    assert result.text == "fact"
    assert result.stop_reason == "retrieval"
    assert result.generated_tokens == 5
    assert result.relevance.probabilities["[Relevant]"] == pytest.approx(.8)
    assert result.support.probabilities["[Partially supported]"] == pytest.approx(.3)
    assert result.utility.probabilities["[Utility:5]"] == pytest.approx(.6)
    assert result.next_retrieval.probabilities["[Continue to Use Evidence]"] == pytest.approx(.6)
    assert result.continuation == "[Relevant]fact[Fully supported][Utility:5]"
    assert math.exp(result.mean_sequence_logprob) == pytest.approx((.8 * 1 * .6 * .6 * .6) ** .2)
    assert model.calls == ([(2, 2), (1, 3), (1, 4), (1, 5), (1, 6)] if cache
                           else [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)])
    assert not model.training


def test_next_token_retrieval_distribution_uses_full_group_without_decoding_text():
    ids = TinyTokenizer.ids
    model = scripted_model([{ids["[Retrieval]"]: .3, ids["[No Retrieval]"]: .1,
                             ids["[Continue to Use Evidence]"]: .6}])
    result = TransformersSelfRAGDecoder(model, TinyTokenizer()).retrieval_distribution("task")
    assert result.group == Group.RETRIEVAL
    assert result.probabilities == pytest.approx({"[Retrieval]": .3, "[No Retrieval]": .1,
                                                "[Continue to Use Evidence]": .6})
    assert len(model.calls) == 1


def test_eos_is_counted_but_not_replayed_and_missing_utility_is_not_fabricated():
    model = scripted_model([{15: 1}, {0: 1}])
    result = TransformersSelfRAGDecoder(model, TinyTokenizer()).decode_segment("task", max_new_tokens=5)
    assert result.stop_reason == "eos"
    assert result.generated_tokens == 2
    assert result.text == result.continuation == "fact"
    assert result.utility is None


def test_context_and_output_limits_do_not_silently_drop_existing_history():
    decoder = TransformersSelfRAGDecoder(scripted_model([{15: 1}]), TinyTokenizer(), max_context_tokens=4)
    result = decoder.decode_segment("task", max_new_tokens=50)
    assert result.generated_tokens == 2
    assert result.stop_reason == "length"
    with pytest.raises(ValueError, match="no generation capacity"):
        decoder.decode_segment("too many input tokens", max_new_tokens=1)


def test_tokenizer_without_learned_atomic_reflection_tokens_is_rejected():
    tokenizer = TinyTokenizer()
    tokenizer.encode = lambda text, **kwargs: [15, 16]
    with pytest.raises(ValueError, match="one learned special token"):
        TransformersSelfRAGDecoder(scripted_model([{15: 1}]), tokenizer)


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers is optional")
def test_actual_tiny_llama_logits_and_cached_decode_match_reference_forward():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(17)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                      max_position_embeddings=64, eos_token_id=0))
    tokenizer = TinyTokenizer()
    adapter = TransformersSelfRAGDecoder(model, tokenizer)
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([[14, 15]])).logits[0, -1]
        expected = torch.softmax(logits[[1, 2, 3]], dim=0).tolist()
    assert list(adapter.retrieval_distribution("task").probabilities.values()) == pytest.approx(expected)

    # A fixed language head keeps decoding in ordinary text, while the real
    # Llama layers exercise Transformers' DynamicCache and attention masks.
    model.lm_head = torch.nn.Linear(16, 32, bias=True)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[15] = 4
    result = adapter.decode_segment("task", max_new_tokens=4)
    assert result.stop_reason == "length"
    assert result.text == "fact" * 4
    assert result.generated_tokens == 4
    assert result.mean_sequence_logprob == pytest.approx(math.log(math.exp(4) / (math.exp(4) + 31)))


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers is optional")
def test_from_pretrained_loads_local_saved_model_and_tokenizer_without_network(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocabulary = {token: index for index, token in enumerate(["</s>", "<s>", "[UNK]", "question", *TOKENS])}
    backend = Tokenizer(models.WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>", bos_token="<s>",
                                       unk_token="[UNK]", additional_special_tokens=TOKENS)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                      max_position_embeddings=64, eos_token_id=0))
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    adapter = TransformersSelfRAGDecoder.from_pretrained(str(tmp_path), model_kwargs={"local_files_only": True},
                                                        tokenizer_kwargs={"local_files_only": True})
    distribution = adapter.retrieval_distribution("question")
    assert distribution.model_id == str(tmp_path)
    assert sum(distribution.probabilities.values()) == pytest.approx(1)
    assert adapter.decode_segment("question", max_new_tokens=2).generated_tokens <= 2
