"""Actual supervised language-model adaptation for RAFT and RankRAG.

RAFT follows https://arxiv.org/abs/2403.10131 section 3: oracle dropout affects
the input, while the quoted, verified target stays unchanged. RankRAG follows
https://arxiv.org/abs/2407.02485 sections 4.2/4.3: one model learns QA and
relevance tasks, scores passages by P(True), then answers with its top passages.

This module supplies the optimization and inference mechanisms, not pretrained
weights, the papers' training mixtures, distributed training, or benchmark
reproduction. Transformers and torch are optional and imported only on use.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice
import json
import math
from pathlib import Path
import random
from typing import Any

from ..base import Document, _snapshot_document, _validate_top_k
from ..reranking import BaseReranker, _copy_with_rerank_score
from ..schema import RAGResponse, Source
from .data import RAFTTrainingRecord, RetrievalTrainingExample, _document_key


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_query(query: str) -> str:
    return _text(query, "query").strip()


def _positive(value: float, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


@dataclass(frozen=True)
class GenerativeTrainingExample:
    """An immutable prompt/completion boundary for assistant-only supervision.

    Targets can contain QA answers, verified RAFT rationales, or supervised
    reading notes followed by an answer. Text is never silently truncated.
    """

    prompt: str
    target: str
    task: str = "instruction"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("prompt", "target", "task"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "metadata", deepcopy(self.metadata))

    @classmethod
    def from_raft(cls, record: RAFTTrainingRecord, *, require_rationale: bool = True,
                  require_distractors: bool = True) -> GenerativeTrainingExample:
        """Validate oracle provenance again at the training boundary.

        Defaults require the paper's quoted target and distracting evidence.
        Disabling either requirement explicitly enables an ablation. Withheld
        oracle evidence is checked, but never appended to the model input.
        """
        if not isinstance(record, RAFTTrainingRecord):
            raise TypeError("record must be a RAFTTrainingRecord")
        if not isinstance(require_rationale, bool) or not isinstance(require_distractors, bool):
            raise TypeError("RAFT requirements must be booleans")
        # Document objects inside a frozen record are mutable. Reconstructing
        # prevents a caller's later edit from bypassing quotation verification.
        checked = RAFTTrainingRecord(record.question, record.answer, record.documents, record.oracle_doc_ids,
                                     record.oracle_included, record.metadata, record.rationale, record.oracle_documents)
        if require_rationale and checked.rationale is None:
            raise ValueError("RAFT training requires a quoted rationale; disable require_rationale for an ablation")
        oracle_ids = set(checked.oracle_doc_ids)
        if not oracle_ids:
            raise ValueError("RAFT training requires identified oracle evidence")
        oracles = checked.oracle_documents or tuple(doc for doc in checked.documents if _document_key(doc) in oracle_ids)
        if not oracles:
            raise ValueError("RAFT training requires supervision oracle documents to validate distractors")
        distractors = [doc for doc in checked.documents if _document_key(doc) not in oracle_ids]
        if require_distractors and not distractors:
            raise ValueError("RAFT training requires distractor documents")
        oracle_contents = {" ".join(doc.content.split()) for doc in oracles}
        if any(" ".join(doc.content.split()) in oracle_contents for doc in distractors):
            raise ValueError("A distractor cannot duplicate oracle content under another ID")
        return cls(checked.render_prompt(), checked.render_target(), "raft", {
            "oracle_included": checked.oracle_included, "oracle_doc_ids": list(checked.oracle_doc_ids),
            "context_doc_ids": [_document_key(doc) for doc in checked.documents],
            "has_quoted_rationale": checked.rationale is not None, "distractor_count": len(distractors),
            "record_metadata": deepcopy(checked.metadata),
        })


def _documents(documents: Sequence[Document]) -> list[Document]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence) or not documents:
        raise ValueError("documents must be a non-empty sequence")
    result = [_snapshot_document(document) for document in documents]
    if len({_document_key(doc) for doc in result}) != len(result):
        raise ValueError("document IDs must be unique")
    return result


class RankRAGDatasetBuilder:
    """Prepare the shared text-to-text tasks from RankRAG section 4.2.

    Relevance labels must come from annotations or a separately evaluated
    silver-labeling process. This builder never invents negatives from absence
    of a lexical answer match. A caller can mix ordinary instruction examples,
    context-rich QA, retrieval QA, binary ranking, and passage-index ranking.
    Prompt wording is a compact adaptation, not a checkpoint-specific template.
    """

    @staticmethod
    def ranking_prompt(query: str, document: Document) -> str:
        query = _validate_query(query)
        return (f"Passage:\n{document.content}\n\nQuestion: {query}\n"
                "Decide whether this passage is relevant to answering the question. "
                "Reply True or False.\nAnswer:")

    @staticmethod
    def qa_prompt(query: str, documents: Sequence[Document]) -> str:
        query = _validate_query(query)
        context = "\n\n".join(f"Passage {index}:\n{doc.content}" for index, doc in enumerate(documents, 1))
        return (f"Use the passages to answer the question. If they do not provide an answer, say so.\n\n"
                f"{context}\n\nQuestion: {query}\nAnswer:")

    @classmethod
    def relevance(cls, query: str, document: Document, relevant: bool) -> GenerativeTrainingExample:
        if not isinstance(relevant, bool):
            raise TypeError("relevant must be a boolean annotation")
        document = _snapshot_document(document)
        return GenerativeTrainingExample(cls.ranking_prompt(query, document), "True" if relevant else "False",
                                         "context_ranking", {"doc_id": _document_key(document), "relevant": relevant})

    @classmethod
    def qa(cls, query: str, documents: Sequence[Document], answer: str) -> GenerativeTrainingExample:
        checked = _documents(documents)
        return GenerativeTrainingExample(cls.qa_prompt(query, checked), answer,
                                         "context_qa" if len(checked) == 1 else "retrieval_qa",
                                         {"doc_ids": [_document_key(doc) for doc in checked]})

    @classmethod
    def passage_ranking(cls, query: str, documents: Sequence[Document],
                        relevant_doc_ids: Sequence[str]) -> GenerativeTrainingExample:
        query, checked = _validate_query(query), _documents(documents)
        if isinstance(relevant_doc_ids, (str, bytes)) or not isinstance(relevant_doc_ids, Sequence):
            raise TypeError("relevant_doc_ids must be a sequence")
        relevant = set(relevant_doc_ids)
        keys = [_document_key(doc) for doc in checked]
        if len(relevant) != len(relevant_doc_ids) or not relevant.issubset(keys):
            raise ValueError("relevant document IDs must be unique and present in the context")
        context = "\n\n".join(f"Passage {index}:\n{doc.content}" for index, doc in enumerate(checked, 1))
        prompt = (f"{context}\n\nQuestion: {query}\nIdentify every passage relevant to answering the question. "
                  "Return their 1-based indexes, separated by commas, or None.\nAnswer:")
        target = ", ".join(str(index) for index, key in enumerate(keys, 1) if key in relevant) or "None"
        return GenerativeTrainingExample(prompt, target, "retrieval_ranking", {"doc_ids": keys})

    @classmethod
    def from_retrieval_example(cls, example: RetrievalTrainingExample, *, seed: int = 0) -> list[GenerativeTrainingExample]:
        if not isinstance(example, RetrievalTrainingExample):
            raise TypeError("example must be a RetrievalTrainingExample")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        checked = RetrievalTrainingExample(example.query, example.positive_documents, example.negative_documents,
                                           example.answer, example.metadata)
        positives, negatives = checked.positive_documents, checked.negative_documents
        if {" ".join(doc.content.split()) for doc in positives} & {" ".join(doc.content.split()) for doc in negatives}:
            raise ValueError("positive and negative passages cannot have identical content")
        result = [cls.relevance(checked.query, doc, label) for docs, label in ((positives, True), (negatives, False))
                  for doc in docs]
        combined = [*positives, *negatives]
        random.Random(seed).shuffle(combined)
        result.append(cls.passage_ranking(checked.query, combined, checked.positive_doc_ids))
        if checked.answer is not None:
            result.append(cls.qa(checked.query, positives, checked.answer))
            if negatives:
                result.append(cls.qa(checked.query, combined, checked.answer))
        return result

    @staticmethod
    def blend(datasets: Mapping[str, Sequence[GenerativeTrainingExample]], weights: Mapping[str, float], *,
              sample_count: int, seed: int = 0) -> list[GenerativeTrainingExample]:
        """Sample an explicitly weighted task/data blend with replacement.

        The paper's dataset proportions require those actual datasets; this
        function intentionally has no purported reproduction default mixture.
        """
        count = _validate_top_k(sample_count, name="sample_count")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not datasets or set(datasets) != set(weights):
            raise ValueError("datasets and weights must have the same non-empty keys")
        keys, values = list(datasets), [_positive(weights[key], "weight", zero=True) for key in datasets]
        if not any(values):
            raise ValueError("at least one blend weight must be positive")
        for key in keys:
            if not datasets[key] or any(not isinstance(item, GenerativeTrainingExample) for item in datasets[key]):
                raise ValueError("each dataset must contain training examples")
        rng = random.Random(seed)
        return [rng.choice(datasets[key]) for key in rng.choices(keys, weights=values, k=count)]


@dataclass(frozen=True)
class GenerativeTrainingReport:
    examples: int
    steps: int
    supervised_tokens: int
    initial_loss: float
    final_loss: float
    epoch_losses: tuple[float, ...]


class TransformersGenerativeTrainer:
    """Single-device, actual AdamW SFT for causal or encoder-decoder models.

    Causal prompts and targets are tokenized separately and concatenated at an
    explicit token boundary; a BOS token is optionally prepended. This avoids
    ambiguous subword merges across that boundary. The exact same prompt
    encoding is used at inference. Targets receive EOS, prompt and padding
    labels are -100, and genuine EOS remains supervised even when PAD == EOS.
    Supply already-formatted prompts to use a particular model's chat format.
    No implicit chat template or evidence/target truncation is applied.

    ``fit`` can train either kind of model. RankRAG's original model is causal;
    the seq2seq path is an explicit extension. Concurrent fit/inference on the
    same model, optimizer resumption, and distributed models are unsupported.
    """

    def __init__(self, model: Any, tokenizer: Any, *, max_input_tokens: int = 2048,
                 max_target_tokens: int = 512, add_bos_token: bool = True) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Generative training requires torch and transformers") from exc
        if not callable(model) or not callable(getattr(model, "parameters", None)) or not callable(tokenizer):
            raise TypeError("model and tokenizer must be Transformers-compatible")
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define padding and EOS tokens")
        if not isinstance(add_bos_token, bool):
            raise TypeError("add_bos_token must be a boolean")
        self.model, self.tokenizer, self._torch = model, tokenizer, torch
        self.mode = "seq2seq" if model.config.is_encoder_decoder else "causal"
        if self.mode == "seq2seq" and getattr(model.config, "decoder_start_token_id", None) is None:
            raise ValueError("seq2seq model must define decoder_start_token_id")
        self.max_input_tokens = _validate_top_k(max_input_tokens, name="max_input_tokens")
        self.max_target_tokens = _validate_top_k(max_target_tokens, name="max_target_tokens")
        self.add_bos_token, self.training_steps = add_bos_token, 0
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _check_context(self, length: int) -> None:
        raw_limits = [getattr(self.model.config, name, None) for name in ("max_position_embeddings", "n_positions")]
        limits = [limit for limit in raw_limits if isinstance(limit, int) and limit > 0]
        if limits and length > min(limits):
            raise ValueError("token sequence exceeds the model context window")

    def _prompt_ids(self, prompt: str) -> list[int]:
        ids = list(self.tokenizer.encode(_text(prompt, "prompt"), add_special_tokens=self.mode == "seq2seq"))
        if self.mode == "causal" and self.add_bos_token and self.tokenizer.bos_token_id is not None:
            if not ids or ids[0] != self.tokenizer.bos_token_id:
                ids.insert(0, self.tokenizer.bos_token_id)
        if not ids or len(ids) > self.max_input_tokens:
            raise ValueError("prompt is empty after tokenization or exceeds max_input_tokens")
        self._check_context(len(ids))
        return ids

    def _target_ids(self, target: str) -> list[int]:
        ids = list(self.tokenizer.encode(_text(target, "target"), add_special_tokens=False))
        if not ids:
            raise ValueError("target is empty after tokenization")
        if ids[-1] != self.tokenizer.eos_token_id:
            ids.append(self.tokenizer.eos_token_id)
        if len(ids) > self.max_target_tokens:
            raise ValueError("target exceeds max_target_tokens; truncation would lose supervision")
        return ids

    def collate(self, examples: Sequence[GenerativeTrainingExample]) -> dict[str, Any]:
        """Return auditable tensors, with padding masked by position, not ID."""
        if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence) or not examples:
            raise ValueError("examples must be a non-empty sequence")
        if any(not isinstance(item, GenerativeTrainingExample) for item in examples):
            raise TypeError("examples must contain GenerativeTrainingExample objects")
        inputs, targets = [], []
        for example in examples:
            prompt, target = self._prompt_ids(example.prompt), self._target_ids(example.target)
            if self.mode == "causal":
                self._check_context(len(prompt) + len(target))
                inputs.append(prompt + target)
                targets.append([-100] * len(prompt) + target)
            else:
                self._check_context(len(target))
                inputs.append(prompt)
                targets.append(target)
        width, target_width = max(map(len, inputs)), max(map(len, targets))
        input_ids = [row + [self.tokenizer.pad_token_id] * (width - len(row)) for row in inputs]
        attention = [[1] * len(row) + [0] * (width - len(row)) for row in inputs]
        labels = [row + [-100] * (target_width - len(row)) for row in targets]
        return {name: self._torch.tensor(rows, dtype=self._torch.long, device=self.device)
                for name, rows in (("input_ids", input_ids), ("attention_mask", attention), ("labels", labels))}

    def _loss(self, examples):
        batch = self.collate(examples)
        output = self.model(**batch)
        loss = output.loss
        if loss.ndim != 0 or not bool(self._torch.isfinite(loss)):
            raise ValueError("model must return a finite scalar training loss")
        labels = batch["labels"][:, 1:] if self.mode == "causal" else batch["labels"]
        return loss, int((labels != -100).sum().item())

    def fit(self, examples: Sequence[GenerativeTrainingExample | RAFTTrainingRecord], *, epochs: int = 1,
            batch_size: int = 4, learning_rate: float = 2e-5, weight_decay: float = 0.,
            max_grad_norm: float = 1., max_steps: int | None = None, seed: int = 0) -> GenerativeTrainingReport:
        """Optimize answer-token likelihood; RAFT records use strict validation.

        To train a RAFT ablation, explicitly construct training examples with
        ``from_raft`` and the desired relaxed requirements first. ``seed``
        controls data order, not the supplied model's initialization/dropout.
        Losses are measured token-weighted means; they are not QA metrics.
        """
        if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence) or not examples:
            raise ValueError("examples must be a non-empty sequence")
        data = tuple(GenerativeTrainingExample.from_raft(item) if isinstance(item, RAFTTrainingRecord) else item
                     for item in examples)
        if any(not isinstance(item, GenerativeTrainingExample) for item in data):
            raise TypeError("examples must contain training examples or RAFT records")
        epochs, batch_size = _validate_top_k(epochs, name="epochs"), _validate_top_k(batch_size, name="batch_size")
        if max_steps is not None:
            max_steps = _validate_top_k(max_steps, name="max_steps")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        rate, decay, clip = _positive(learning_rate, "learning_rate"), _positive(weight_decay, "weight_decay", zero=True), _positive(max_grad_norm, "max_grad_norm")
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("model has no trainable parameters")
        optimizer = self._torch.optim.AdamW(parameters, lr=rate, weight_decay=decay)
        was_training = self.model.training
        order, rng = list(range(len(data))), random.Random(seed)
        steps, token_count, losses = 0, 0, []

        def evaluate():
            self.model.eval()
            total, tokens = 0., 0
            with self._torch.inference_mode():
                for start in range(0, len(data), batch_size):
                    loss, count = self._loss(data[start:start + batch_size])
                    total += float(loss.item()) * count
                    tokens += count
            return total / tokens

        try:
            # This pass validates every row and context length before any update.
            initial = evaluate()
            for _ in range(epochs):
                rng.shuffle(order)
                self.model.train()
                total, tokens = 0., 0
                for start in range(0, len(order), batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    loss, count = self._loss([data[index] for index in order[start:start + batch_size]])
                    loss.backward()
                    self._torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.detach().item()) * count
                    tokens += count
                    steps += 1
                    self.training_steps += 1
                    if max_steps is not None and steps >= max_steps:
                        break
                token_count += tokens
                losses.append(total / tokens)
                if max_steps is not None and steps >= max_steps:
                    break
            return GenerativeTrainingReport(len(data), steps, token_count, initial, evaluate(), tuple(losses))
        finally:
            optimizer.zero_grad(set_to_none=True)
            self.model.train(was_training)

    def next_token_probabilities(self, prompts: Sequence[str], token: str) -> list[float]:
        """P(token) over the full vocabulary, not a True/False renormalization."""
        if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
            raise TypeError("prompts must be a sequence")
        ids = self.tokenizer.encode(_text(token, "token"), add_special_tokens=False)
        if len(ids) != 1 or ids[0] == self.tokenizer.unk_token_id:
            raise ValueError("ranking label must be one non-unknown token")
        rows = [self._prompt_ids(prompt) for prompt in prompts]
        result = []
        was_training = self.model.training
        self.model.eval()
        try:
            with self._torch.inference_mode():
                for row in rows:
                    input_ids = self._torch.tensor([row], device=self.device)
                    kwargs = {"input_ids": input_ids, "attention_mask": self._torch.ones_like(input_ids)}
                    if self.mode == "seq2seq":
                        kwargs["decoder_input_ids"] = self._torch.tensor([[self.model.config.decoder_start_token_id]], device=self.device)
                    logits = self.model(**kwargs).logits[0, -1].float()
                    if not bool(self._torch.isfinite(logits).all()) or ids[0] >= logits.shape[-1]:
                        raise ValueError("model must return finite vocabulary logits")
                    result.append(float(self._torch.softmax(logits, dim=-1)[ids[0]].item()))
        finally:
            self.model.train(was_training)
        return result

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> str:
        count = _validate_top_k(max_new_tokens, name="max_new_tokens")
        if count > self.max_target_tokens:
            raise ValueError("generation exceeds max_target_tokens")
        ids = self._prompt_ids(prompt)
        self._check_context(len(ids) + count if self.mode == "causal" else count + 1)
        was_training = self.model.training
        self.model.eval()
        try:
            with self._torch.inference_mode():
                tensor = self._torch.tensor([ids], device=self.device)
                output = self.model.generate(input_ids=tensor, attention_mask=self._torch.ones_like(tensor),
                                             do_sample=False, max_new_tokens=count, num_beams=1,
                                             num_return_sequences=1, return_dict_in_generate=False,
                                             pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                # Encoder-decoder generate includes its decoder start token.
                # It need not be a tokenizer special token (e.g. a language ID).
                generated = output[0, len(ids):] if self.mode == "causal" else output[0, 1:]
                return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        finally:
            self.model.train(was_training)

    def save_pretrained(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(destination, safe_serialization=True)
        self.tokenizer.save_pretrained(destination)
        metadata = {"format_version": 1, "mode": self.mode, "max_input_tokens": self.max_input_tokens,
                    "max_target_tokens": self.max_target_tokens, "add_bos_token": self.add_bos_token,
                    "training_steps": self.training_steps}
        (destination / "cheragh_generative.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, model_id: str | Path, *, revision: str | None = None,
                        model_kwargs: Mapping[str, Any] | None = None,
                        tokenizer_kwargs: Mapping[str, Any] | None = None, **trainer_kwargs) -> TransformersGenerativeTrainer:
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch and transformers for generative adaptation") from exc
        metadata: dict[str, Any] = {}
        metadata_path = Path(model_id) / "cheragh_generative.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
                raise ValueError("unsupported generative model metadata")
        model_options, tokenizer_options = dict(model_kwargs or {}), dict(tokenizer_kwargs or {})
        for options in (model_options, tokenizer_options):
            if options.get("trust_remote_code"):
                raise ValueError("remote model code is not enabled by this trainer")
            options["trust_remote_code"] = False
            if revision is not None:
                options["revision"] = revision
        config_options = {key: value for key, value in model_options.items()
                          if key in {"revision", "cache_dir", "local_files_only", "token", "trust_remote_code"}}
        config = AutoConfig.from_pretrained(str(model_id), **config_options)
        mode = "seq2seq" if config.is_encoder_decoder else "causal"
        if metadata and metadata.get("mode") != mode:
            raise ValueError("saved generative mode differs from model architecture")
        loader = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
        model, tokenizer = loader.from_pretrained(str(model_id), **model_options), AutoTokenizer.from_pretrained(str(model_id), **tokenizer_options)
        options = {key: metadata[key] for key in ("max_input_tokens", "max_target_tokens", "add_bos_token") if key in metadata}
        options.update(trainer_kwargs)
        instance = cls(model, tokenizer, **options)
        steps = metadata.get("training_steps", 0)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("invalid saved training_steps")
        instance.training_steps = steps
        return instance


class RankRAGModel(BaseReranker):
    """Use the same trained generative weights for ranking and answering."""

    def __init__(self, generator: TransformersGenerativeTrainer):
        if not isinstance(generator, TransformersGenerativeTrainer):
            raise TypeError("generator must be a TransformersGenerativeTrainer")
        self.generator = generator

    def rerank(self, query: str, documents: Sequence[Document], top_k: int = 5) -> list[Document]:
        query, count = _validate_query(query), _validate_top_k(top_k)
        if not documents:
            return []
        checked = _documents(documents)
        scores = self.generator.next_token_probabilities([RankRAGDatasetBuilder.ranking_prompt(query, doc) for doc in checked], "True")
        ordered = sorted(zip(checked, scores), key=lambda pair: pair[1], reverse=True)
        return [_copy_with_rerank_score(doc, score, extra_metadata={"rankrag_probability_true": score}) for doc, score in ordered[:count]]

    def answer(self, query: str, documents: Sequence[Document], *, max_new_tokens: int = 128) -> str:
        return self.generator.generate(RankRAGDatasetBuilder.qa_prompt(query, documents), max_new_tokens=max_new_tokens)


class RankRAGEngine:
    """Bounded retrieve → P(True) rerank → generation using one model.

    Returned sources are the conditioning passages, not fabricated evidence
    citations. The paper does not require inline citations or entailment checks.
    """

    def __init__(self, retriever, model: RankRAGModel, *, candidate_top_k: int = 100, top_k: int = 5,
                 max_new_tokens: int = 128):
        if not callable(getattr(retriever, "retrieve", None)) or not isinstance(model, RankRAGModel):
            raise TypeError("retriever and RankRAGModel are required")
        self.retriever, self.model = retriever, model
        self.candidate_top_k = _validate_top_k(candidate_top_k, name="candidate_top_k")
        self.top_k = _validate_top_k(top_k)
        self.max_new_tokens = _validate_top_k(max_new_tokens, name="max_new_tokens")
        if self.top_k > self.candidate_top_k:
            raise ValueError("top_k must not exceed candidate_top_k")

    def ask(self, query: str, *, top_k: int | None = None) -> RAGResponse:
        query = _validate_query(query)
        count = self.top_k if top_k is None else _validate_top_k(top_k)
        if count > self.candidate_top_k:
            raise ValueError("top_k must not exceed candidate_top_k")
        candidates = list(islice(self.retriever.retrieve(query, top_k=self.candidate_top_k), self.candidate_top_k))
        selected = self.model.rerank(query, candidates, top_k=count)
        prompt = RankRAGDatasetBuilder.qa_prompt(query, selected)
        answer = self.model.generator.generate(prompt, max_new_tokens=self.max_new_tokens)
        return RAGResponse(query, answer, [Source.from_document(doc) for doc in selected], selected, prompt,
                           metadata={"technique": "rankrag", "candidate_count": len(candidates),
                                     "selected_count": len(selected), "ranking_score": "full_vocabulary_probability_true",
                                     "training_steps": self.model.generator.training_steps})


__all__ = ["GenerativeTrainingExample", "GenerativeTrainingReport", "TransformersGenerativeTrainer",
           "RankRAGDatasetBuilder", "RankRAGModel", "RankRAGEngine"]
