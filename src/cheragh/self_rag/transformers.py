"""Concrete greedy reflection-token decoder for Hugging Face causal models.

Load a Self-RAG-trained checkpoint, e.g. ``selfrag/selfrag_llama2_7b``. An
ordinary chat model, even one that can print the token spellings, is not a
replacement. Token probabilities come from full model logits at the generated
reflection positions; textual self-assessments are never converted to scores.

Torch and Transformers are optional and imported only when an adapter is made.
No checkpoint is downloaded implicitly. ``from_pretrained`` explicitly loads
the requested checkpoint using Transformers' normal revision/cache controls.
"""
from __future__ import annotations

from typing import Any

from ..base import _validate_top_k
from .reflection import ReflectionTokenDistribution as Distribution, ReflectionTokenGroup as Group, _TOKENS
from .segmented import ReflectionSegment


class TransformersSelfRAGDecoder:
    """Autoregressive decoding with KV cache and exact reflection distributions.

    Segments terminate at the next retrieval token, EOS, or a length boundary.
    Critique tokens remain in ``continuation`` for future model conditioning
    while ``text`` omits special tokens. Missing critique tokens are reported
    as absent; the beam engine decides which groups are required.

    ``max_context_tokens`` never silently truncates passage/history tokens.
    A prompt that already fills the context raises an error. Generation that
    reaches remaining context capacity returns ``stop_reason='length'``.
    """

    def __init__(self, model: Any, tokenizer: Any, *, model_id: str | None = None,
                 max_context_tokens: int | None = None, input_device: Any = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TransformersSelfRAGDecoder requires torch and transformers") from exc
        if not callable(model) or not callable(getattr(model, "eval", None)):
            raise TypeError("model must be a causal language model")
        if not callable(getattr(tokenizer, "encode", None)) or not callable(getattr(tokenizer, "decode", None)):
            raise TypeError("tokenizer must provide encode and decode")
        if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
            raise ValueError("model_id must be a non-empty string or None")
        configured_limit = getattr(getattr(model, "config", None), "max_position_embeddings", 2048)
        self.max_context_tokens = _validate_top_k(
            configured_limit if max_context_tokens is None else max_context_tokens, name="max_context_tokens"
        )
        if isinstance(configured_limit, int) and self.max_context_tokens > configured_limit:
            raise ValueError("max_context_tokens exceeds the model's configured positional capacity")
        self._torch, self.model, self.tokenizer = torch, model, tokenizer
        self.model_id = model_id
        self.input_device = input_device if input_device is not None else next(model.parameters()).device
        self._groups = (Group.RETRIEVAL, Group.RELEVANCE, Group.SUPPORT, Group.UTILITY)
        self._ids: dict[str, int] = {}
        unknown = getattr(tokenizer, "unk_token_id", None)
        for group in self._groups:
            for token in _TOKENS[group]:
                ids = tokenizer.encode(token, add_special_tokens=False)
                if len(ids) != 1 or ids[0] == unknown:
                    raise ValueError(f"checkpoint tokenizer must encode {token!r} as one learned special token")
                self._ids[token] = int(ids[0])
        if len(set(self._ids.values())) != len(self._ids):
            raise ValueError("reflection tokens must have distinct vocabulary IDs")
        vocabulary_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if isinstance(vocabulary_size, int) and max(self._ids.values()) >= vocabulary_size:
            raise ValueError("reflection token IDs exceed model vocabulary; use compatible trained weights")
        self._id_groups = {self._ids[token]: group for group in self._groups for token in _TOKENS[group]}
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if eos is None:
            eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            raise ValueError("model/tokenizer must define EOS")
        self._eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_id: str, *, revision: str | None = None,
                        max_context_tokens: int | None = None,
                        model_kwargs: dict[str, Any] | None = None,
                        tokenizer_kwargs: dict[str, Any] | None = None) -> TransformersSelfRAGDecoder:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch and transformers to load a Self-RAG checkpoint") from exc
        model_options, tokenizer_options = dict(model_kwargs or {}), dict(tokenizer_kwargs or {})
        if model_options.get("trust_remote_code") or tokenizer_options.get("trust_remote_code"):
            raise ValueError("remote model code is not enabled by this adapter")
        for options in (model_options, tokenizer_options):
            options["trust_remote_code"] = False
            if revision is not None:
                options["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_options)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_options)
        return cls(model, tokenizer, model_id=f"{model_id}@{revision}" if revision else model_id,
                   max_context_tokens=max_context_tokens)

    def _input(self, prompt: str):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be non-empty text")
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not ids or len(ids) >= self.max_context_tokens:
            raise ValueError("prompt leaves no generation capacity in the model context")
        tensor = self._torch.tensor([ids], dtype=self._torch.long, device=self.input_device)
        return tensor, self._torch.ones_like(tensor)

    def _distribution(self, logprobs: Any, group: Group) -> Distribution:
        return Distribution.from_logprobs(
            group, {token: float(logprobs[self._ids[token]].item()) for token in _TOKENS[group]},
            model_id=self.model_id,
        )

    def retrieval_distribution(self, prompt: str) -> Distribution:
        ids, attention = self._input(prompt)
        with self._torch.inference_mode():
            output = self.model(input_ids=ids, attention_mask=attention, use_cache=False)
            logprobs = self._torch.log_softmax(output.logits[0, -1].float(), dim=-1)
            return self._distribution(logprobs, Group.RETRIEVAL)

    def decode_segment(self, prompt: str, *, max_new_tokens: int) -> ReflectionSegment:
        maximum = _validate_top_k(max_new_tokens, name="max_new_tokens")
        ids, attention = self._input(prompt)
        maximum = min(maximum, self.max_context_tokens - ids.shape[1])
        generated: list[int] = []
        visible: list[int] = []
        observed: dict[Group, Distribution] = {}
        token_logprobs: list[float] = []
        next_retrieval = None
        stop_reason = "length"
        past = None
        with self._torch.inference_mode():
            for _ in range(maximum):
                output = self.model(input_ids=ids, attention_mask=attention, past_key_values=past, use_cache=True)
                logprobs = self._torch.log_softmax(output.logits[0, -1].float(), dim=-1)
                selected = int(logprobs.argmax().item())
                token_logprobs.append(float(logprobs[selected].item()))
                group = self._id_groups.get(selected)
                if group == Group.RETRIEVAL:
                    next_retrieval = self._distribution(logprobs, Group.RETRIEVAL)
                    stop_reason = "retrieval"
                    break
                if selected in self._eos:
                    stop_reason = "eos"
                    break
                if group is not None:
                    if group not in observed:
                        observed[group] = self._distribution(logprobs, group)
                else:
                    visible.append(selected)
                generated.append(selected)
                past = output.past_key_values
                if past is None:
                    # Some compatible causal models do not implement caching.
                    ids = self._torch.cat((ids, ids.new_tensor([[selected]])), dim=1)
                else:
                    ids = ids.new_tensor([[selected]])
                attention = self._torch.cat((attention, attention.new_ones((1, 1))), dim=1)
        return ReflectionSegment(
            text=self.tokenizer.decode(visible, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            continuation=self.tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            generated_tokens=len(token_logprobs), mean_sequence_logprob=sum(token_logprobs) / len(token_logprobs),
            stop_reason=stop_reason, relevance=observed.get(Group.RELEVANCE), support=observed.get(Group.SUPPORT),
            utility=observed.get(Group.UTILITY), next_retrieval=next_retrieval,
        )
