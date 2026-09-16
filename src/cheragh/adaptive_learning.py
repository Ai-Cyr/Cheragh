"""Learned Adaptive-RAG routing and outcome-derived training labels.

Implements the training/inference boundary from Adaptive-RAG §3.2:
https://arxiv.org/abs/2403.14403
https://github.com/starsuzi/Adaptive-RAG/tree/main/classifier

The data builder executes all three supplied strategies and evaluates their
actual answers. The optional Transformers classifier trains real parameters
with cross-entropy: seq2seq labels A/B/C (the paper's T5 path), or a three-logit
classification head. This module neither changes the explicit heuristic
classifier nor claims that a randomly initialized model has learned routing.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import random
import re
import string
import time
from typing import Any

from .adaptive import AdaptiveRAGRoute as Route, AdaptiveRoutingDecision, _validate_query
from .base import _validate_top_k


_ROUTES = (Route.NO_RETRIEVAL, Route.SINGLE_STEP, Route.ITERATIVE)
_LABELS = ("A", "B", "C")
_METADATA_FILE = "cheragh_adaptive_classifier.json"


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class AdaptiveTrainingQuestion:
    query: str
    reference_answers: tuple[str, ...]
    example_id: str = ""
    dataset_route: Route | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _validate_query(self.query))
        if isinstance(self.reference_answers, (str, bytes)) or not isinstance(self.reference_answers, Sequence):
            raise TypeError("reference_answers must be a sequence of strings")
        if not self.reference_answers or any(not isinstance(value, str) or not value.strip() for value in self.reference_answers):
            raise ValueError("reference_answers must contain non-empty strings")
        object.__setattr__(self, "reference_answers", tuple(self.reference_answers))
        if not isinstance(self.example_id, str):
            raise TypeError("example_id must be a string")
        if self.dataset_route is not None and self.dataset_route not in (Route.SINGLE_STEP, Route.ITERATIVE):
            raise ValueError("dataset_route must be an explicit single-step or iterative dataset prior")
        if self.dataset_route is not None and not isinstance(self.dataset_route, Route):
            raise TypeError("dataset_route must be an AdaptiveRAGRoute")


@dataclass(frozen=True)
class AdaptiveStrategyOutcome:
    route: Route
    answer: str
    correct: bool
    cost: float
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.route, Route):
            raise TypeError("route must be an AdaptiveRAGRoute")
        if not isinstance(self.answer, str) or not isinstance(self.correct, bool):
            raise TypeError("answer must be text and correct must be bool")
        object.__setattr__(self, "cost", _nonnegative(self.cost, "cost"))
        object.__setattr__(self, "elapsed_seconds", _nonnegative(self.elapsed_seconds, "elapsed_seconds"))

    def to_dict(self) -> dict[str, Any]:
        return {"route": self.route.value, "answer": self.answer, "correct": self.correct,
                "cost": self.cost, "elapsed_seconds": self.elapsed_seconds}


@dataclass(frozen=True)
class AdaptiveTrainingExample:
    question: AdaptiveTrainingQuestion
    route: Route
    outcomes: tuple[AdaptiveStrategyOutcome, ...]
    label_source: str = "successful_outcome"

    def __post_init__(self) -> None:
        if not isinstance(self.question, AdaptiveTrainingQuestion) or not isinstance(self.route, Route):
            raise TypeError("question and route must be validated Adaptive-RAG training objects")
        if not isinstance(self.outcomes, Sequence) or any(not isinstance(value, AdaptiveStrategyOutcome) for value in self.outcomes):
            raise TypeError("outcomes must contain AdaptiveStrategyOutcome objects")
        if len(self.outcomes) != 3 or {item.route for item in self.outcomes} != set(_ROUTES):
            raise ValueError("training examples require exactly one outcome from each of the three routes")
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        successful = [item for item in self.outcomes if item.correct]
        if self.label_source == "successful_outcome":
            expected = min(successful, key=lambda item: (item.cost, _ROUTES.index(item.route))) if successful else None
            if expected is None or expected.route != self.route:
                raise ValueError("the label must select the least-cost correct strategy")
        elif self.label_source == "dataset_prior":
            if successful or self.question.dataset_route != self.route:
                raise ValueError("a dataset prior applies only when all strategies fail")
        else:
            raise ValueError("label_source must be successful_outcome or dataset_prior")

    @property
    def query(self) -> str:
        return self.question.query

    @property
    def label(self) -> str:
        return _LABELS[_ROUTES.index(self.route)]

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "reference_answers": list(self.question.reference_answers),
                "example_id": self.question.example_id, "route": self.route.value, "label": self.label,
                "label_source": self.label_source, "outcomes": [item.to_dict() for item in self.outcomes]}


def normalized_answer_exact_match(answer: str, references: Sequence[str]) -> bool:
    """SQuAD-style normalized exact match; replace for task-specific correctness."""

    def normalized(text: str) -> str:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())

    candidate = normalized(answer)
    return any(candidate == normalized(reference) for reference in references)


class AdaptiveSilverDatasetBuilder:
    """Collect actual outcomes, then choose the cheapest correct route.

    Every strategy is called once for each question, including after an early
    successful answer. Default costs 0/1/2 encode the paper's priority for
    simpler strategies; ``cost_fn(route, result, elapsed_seconds)`` can instead
    measure actual tokens, retrieval steps, latency, or another declared cost.
    Cost ties select the simpler route. A dataset prior is used only when all
    strategies fail; otherwise unanswered questions are omitted (``None``).
    """

    def __init__(self, strategies: Mapping[Route, Any], *,
                 evaluator: Callable[[str, Sequence[str]], bool] = normalized_answer_exact_match,
                 cost_fn: Callable[[Route, Any, float], float] | None = None) -> None:
        if not isinstance(strategies, Mapping) or set(strategies) != set(_ROUTES):
            raise ValueError("strategies must contain all three AdaptiveRAGRoute entries")
        if any(not isinstance(route, Route) for route in strategies):
            raise TypeError("strategy keys must be AdaptiveRAGRoute values")
        self.strategies = {}
        for route in _ROUTES:
            strategy = strategies[route]
            method = getattr(strategy, "ask", strategy)
            if not callable(method):
                raise TypeError("each strategy must be callable or provide ask(query)")
            self.strategies[route] = method
        if not callable(evaluator) or (cost_fn is not None and not callable(cost_fn)):
            raise TypeError("evaluator and cost_fn must be callable")
        self.evaluator, self.cost_fn = evaluator, cost_fn

    def collect(self, question: AdaptiveTrainingQuestion) -> AdaptiveTrainingExample | None:
        if not isinstance(question, AdaptiveTrainingQuestion):
            raise TypeError("question must be an AdaptiveTrainingQuestion")
        outcomes = []
        for route in _ROUTES:
            started = time.monotonic()
            raw = self.strategies[route](question.query)
            elapsed = time.monotonic() - started
            result = getattr(raw, "response", raw)
            answer = result if isinstance(result, str) else getattr(result, "answer", None)
            if not isinstance(answer, str):
                raise TypeError("strategies must return text, an answer object, or a wrapper with response.answer")
            correct = self.evaluator(answer, question.reference_answers)
            if not isinstance(correct, bool):
                raise TypeError("correctness evaluator must return bool")
            cost = self.cost_fn(route, raw, elapsed) if self.cost_fn is not None else float(_ROUTES.index(route))
            outcomes.append(AdaptiveStrategyOutcome(route, answer, correct, cost, elapsed))
        return self.from_outcomes(question, outcomes)

    @staticmethod
    def from_outcomes(question: AdaptiveTrainingQuestion,
                      outcomes: Sequence[AdaptiveStrategyOutcome]) -> AdaptiveTrainingExample | None:
        if not isinstance(question, AdaptiveTrainingQuestion):
            raise TypeError("question must be an AdaptiveTrainingQuestion")
        if not isinstance(outcomes, Sequence) or any(not isinstance(item, AdaptiveStrategyOutcome) for item in outcomes):
            raise TypeError("outcomes must be a sequence of AdaptiveStrategyOutcome")
        if len(outcomes) != 3 or {item.route for item in outcomes} != set(_ROUTES):
            raise ValueError("exactly one outcome is required from each route")
        correct = [item for item in outcomes if item.correct]
        if correct:
            selected = min(correct, key=lambda item: (item.cost, _ROUTES.index(item.route)))
            return AdaptiveTrainingExample(question, selected.route, tuple(outcomes))
        if question.dataset_route is not None:
            return AdaptiveTrainingExample(question, question.dataset_route, tuple(outcomes), "dataset_prior")
        return None


@dataclass(frozen=True)
class AdaptiveClassifierTrainingReport:
    examples: int
    steps: int
    initial_loss: float
    final_loss: float
    epoch_losses: tuple[float, ...]
    training_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {"examples": self.examples, "steps": self.steps, "initial_loss": self.initial_loss,
                "final_loss": self.final_loss, "epoch_losses": list(self.epoch_losses),
                "training_accuracy": self.training_accuracy}


class TransformersComplexityClassifier:
    """Learned classifier compatible with ``QueryComplexityClassifier``.

    Seq2seq models use the normalized A/B/C logits of the first decoder step,
    exactly the inference boundary in the authors' classifier. Sequence
    classification models use indices 0/1/2 for no/single/iterative retrieval.
    Probabilities are uncalibrated conditional model probabilities, not an
    estimated chance that downstream QA will be correct.

    Models and tokenizers must be supplied or explicitly loaded. No checkpoint
    is downloaded on import. Long questions are truncated to ``max_input_tokens``
    during both training and inference, as in the reference implementation.
    """

    def __init__(self, model: Any, tokenizer: Any, *, mode: str | None = None,
                 max_input_tokens: int = 384, input_prefix: str = "", model_id: str | None = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("The learned classifier requires torch and transformers") from exc
        if not callable(model) or not callable(getattr(model, "parameters", None)) or not callable(tokenizer):
            raise TypeError("model and tokenizer must be Transformers-compatible objects")
        if not isinstance(input_prefix, str):
            raise TypeError("input_prefix must be a string")
        if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
            raise ValueError("model_id must be a non-empty string or None")
        self.mode = mode or ("seq2seq" if getattr(model.config, "is_encoder_decoder", False) else "sequence_classification")
        if self.mode not in {"seq2seq", "sequence_classification"}:
            raise ValueError("mode must be seq2seq or sequence_classification")
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError("tokenizer must define a padding token")
        self.max_input_tokens = _validate_top_k(max_input_tokens, name="max_input_tokens")
        self.input_prefix, self.model_id = input_prefix, model_id
        self.model, self.tokenizer, self._torch = model, tokenizer, torch
        self.training_steps = 0
        self._label_ids = []
        self._decoder_start = None
        if self.mode == "seq2seq":
            for label in _LABELS:
                ids = tokenizer.encode(label, add_special_tokens=False)
                if len(ids) != 1 or ids[0] == getattr(tokenizer, "unk_token_id", None):
                    raise ValueError("seq2seq labels A/B/C must each map to one vocabulary token")
                self._label_ids.append(int(ids[0]))
            if len(set(self._label_ids)) != 3:
                raise ValueError("A/B/C must have distinct vocabulary IDs")
            self._decoder_start = getattr(model.config, "decoder_start_token_id", None)
            if self._decoder_start is None:
                raise ValueError("seq2seq model must define decoder_start_token_id")
            if max(self._label_ids) >= model.config.vocab_size:
                raise ValueError("label IDs exceed the model vocabulary")
        elif getattr(model.config, "num_labels", None) != 3:
            raise ValueError("sequence classifier must have exactly three labels")
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _encode(self, queries: Sequence[str]):
        encoded = self.tokenizer([self.input_prefix + _validate_query(query) for query in queries],
                                 padding=True, truncation=True, max_length=self.max_input_tokens, return_tensors="pt")
        return {name: value.to(self.device) for name, value in encoded.items() if name in ("input_ids", "attention_mask")}

    def _logits(self, encoded):
        if self.mode == "seq2seq":
            starts = self._torch.full((encoded["input_ids"].shape[0], 1), self._decoder_start,
                                      device=self.device, dtype=self._torch.long)
            return self.model(**encoded, decoder_input_ids=starts).logits[:, 0, self._label_ids]
        return self.model(**encoded).logits

    def predict_proba(self, queries: Sequence[str]) -> list[dict[Route, float]]:
        if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
            raise TypeError("queries must be a sequence of strings")
        if not queries:
            return []
        encoded = self._encode(queries)
        was_training = self.model.training
        self.model.eval()
        try:
            with self._torch.inference_mode():
                logits = self._logits(encoded)
                if logits.shape != (len(queries), 3) or not bool(self._torch.isfinite(logits).all()):
                    raise ValueError("classifier must return finite logits of shape (batch, 3)")
                rows = self._torch.softmax(logits.float(), dim=-1).cpu().tolist()
                return [dict(zip(_ROUTES, row)) for row in rows]
        finally:
            self.model.train(was_training)

    def classify(self, query: str) -> AdaptiveRoutingDecision:
        probabilities = self.predict_proba([query])[0]
        route = max(probabilities, key=lambda item: probabilities[item])
        return AdaptiveRoutingDecision(route, probabilities[route],
                                       f"transformers_{self.mode}; model_id={self.model_id or 'provided_model'}")

    def _loss(self, examples: Sequence[AdaptiveTrainingExample]):
        encoded = self._encode([item.query for item in examples])
        if self.mode == "seq2seq":
            target_encoding = self.tokenizer(text_target=[item.label for item in examples], padding=True,
                                             return_attention_mask=True, return_tensors="pt")
            targets = target_encoding["input_ids"].to(self.device)
            # Mask padding positions, not a token value: EOS can legitimately
            # share PAD's ID and must still teach the decoder when to stop.
            targets = targets.masked_fill(target_encoding["attention_mask"].to(self.device) == 0, -100)
            loss = self.model(**encoded, labels=targets).loss
        else:
            labels = self._torch.tensor([_ROUTES.index(item.route) for item in examples],
                                        dtype=self._torch.long, device=self.device)
            loss = self._torch.nn.functional.cross_entropy(self._logits(encoded), labels)
        if loss.ndim != 0 or not bool(self._torch.isfinite(loss)):
            raise ValueError("training loss must be a finite scalar")
        return loss

    def fit(self, examples: Sequence[AdaptiveTrainingExample], *, epochs: int = 3, batch_size: int = 8,
            learning_rate: float = 3e-5, weight_decay: float = 0., max_grad_norm: float = 1.,
            seed: int = 0, max_steps: int | None = None) -> AdaptiveClassifierTrainingReport:
        """Perform actual AdamW updates and report measured losses/accuracy.

        ``seed`` controls sample order only; initialize/seed a supplied model
        explicitly to reproduce weights and dropout. This single-device loop
        supports CPU and GPU models, not distributed training or serving while
        fitting. Train/evaluation splits must be established by the caller.
        """
        if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence) or not examples:
            raise ValueError("examples must be a non-empty sequence")
        if any(not isinstance(item, AdaptiveTrainingExample) for item in examples):
            raise TypeError("examples must contain AdaptiveTrainingExample objects")
        epochs, batch_size = _validate_top_k(epochs, name="epochs"), _validate_top_k(batch_size, name="batch_size")
        if max_steps is not None:
            max_steps = _validate_top_k(max_steps, name="max_steps")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        rate, decay, clip = (_nonnegative(learning_rate, "learning_rate"), _nonnegative(weight_decay, "weight_decay"),
                             _nonnegative(max_grad_norm, "max_grad_norm"))
        if rate == 0 or clip == 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("model has no trainable parameters")
        optimizer = self._torch.optim.AdamW(parameters, lr=rate, weight_decay=decay)
        data, order = tuple(examples), list(range(len(examples)))
        rng = random.Random(seed)
        steps, losses = 0, []
        was_training = self.model.training

        def evaluation_loss():
            self.model.eval()
            with self._torch.inference_mode():
                total = sum(float(self._loss(data[start:start + batch_size]).item()) * len(data[start:start + batch_size])
                            for start in range(0, len(data), batch_size))
            return total / len(data)

        try:
            initial = evaluation_loss()
            for _ in range(epochs):
                rng.shuffle(order)
                self.model.train()
                total, seen = 0., 0
                for start in range(0, len(order), batch_size):
                    batch = [data[index] for index in order[start:start + batch_size]]
                    optimizer.zero_grad(set_to_none=True)
                    loss = self._loss(batch)
                    loss.backward()
                    self._torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.detach().item()) * len(batch)
                    seen += len(batch)
                    steps += 1
                    self.training_steps += 1
                    if max_steps is not None and steps >= max_steps:
                        break
                losses.append(total / seen)
                if max_steps is not None and steps >= max_steps:
                    break
            final = evaluation_loss()
            predictions = []
            for start in range(0, len(data), batch_size):
                predictions.extend(self.predict_proba([item.query for item in data[start:start + batch_size]]))
            accuracy = sum(max(row, key=lambda route: row[route]) == item.route
                           for row, item in zip(predictions, data)) / len(data)
            return AdaptiveClassifierTrainingReport(len(data), steps, initial, final, tuple(losses), accuracy)
        finally:
            optimizer.zero_grad(set_to_none=True)
            self.model.train(was_training)

    def save_pretrained(self, directory: str | Path) -> None:
        """Persist safe model weights, tokenizer, label mapping, and wrapper settings."""
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(destination, safe_serialization=True)
        self.tokenizer.save_pretrained(destination)
        payload = {"format_version": 1, "mode": self.mode, "max_input_tokens": self.max_input_tokens,
                   "input_prefix": self.input_prefix, "model_id": self.model_id, "training_steps": self.training_steps,
                   "label_routes": dict(zip(_LABELS, [route.value for route in _ROUTES]))}
        (destination / _METADATA_FILE).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, model_id: str | Path, *, mode: str | None = None, revision: str | None = None,
                        max_input_tokens: int | None = None, input_prefix: str | None = None,
                        model_kwargs: Mapping[str, Any] | None = None,
                        tokenizer_kwargs: Mapping[str, Any] | None = None) -> TransformersComplexityClassifier:
        try:
            from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch and transformers to load a learned Adaptive-RAG classifier") from exc
        metadata: dict[str, Any] = {}
        metadata_path = Path(model_id) / _METADATA_FILE
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
                raise ValueError("unsupported Adaptive-RAG classifier metadata")
            if metadata.get("label_routes") != dict(zip(_LABELS, [route.value for route in _ROUTES])):
                raise ValueError("saved classifier label mapping is incompatible")
            if mode is not None and mode != metadata.get("mode"):
                raise ValueError("requested mode differs from saved classifier mode")
        model_options, tokenizer_options = dict(model_kwargs or {}), dict(tokenizer_kwargs or {})
        for options in (model_options, tokenizer_options):
            if options.get("trust_remote_code"):
                raise ValueError("remote model code is not enabled by this classifier")
            options["trust_remote_code"] = False
            if revision is not None:
                options["revision"] = revision
        selected_mode = mode or metadata.get("mode")
        if selected_mode is None:
            config_options = {key: value for key, value in model_options.items()
                              if key in {"revision", "cache_dir", "local_files_only", "token", "trust_remote_code"}}
            config = AutoConfig.from_pretrained(str(model_id), **config_options)
            selected_mode = "seq2seq" if config.is_encoder_decoder else "sequence_classification"
        if selected_mode not in {"seq2seq", "sequence_classification"}:
            raise ValueError("invalid classifier mode")
        loader = AutoModelForSeq2SeqLM if selected_mode == "seq2seq" else AutoModelForSequenceClassification
        if selected_mode == "sequence_classification":
            model_options.setdefault("num_labels", 3)
        tokenizer = AutoTokenizer.from_pretrained(str(model_id), **tokenizer_options)
        model = loader.from_pretrained(str(model_id), **model_options)
        instance = cls(model, tokenizer, mode=selected_mode,
                       max_input_tokens=max_input_tokens if max_input_tokens is not None else metadata.get("max_input_tokens", 384),
                       input_prefix=input_prefix if input_prefix is not None else metadata.get("input_prefix", ""),
                       model_id=metadata.get("model_id") or (f"{model_id}@{revision}" if revision else str(model_id)))
        steps = metadata.get("training_steps", 0)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("invalid saved training_steps")
        instance.training_steps = steps
        return instance


__all__ = [
    "AdaptiveTrainingQuestion", "AdaptiveStrategyOutcome", "AdaptiveTrainingExample",
    "AdaptiveSilverDatasetBuilder", "AdaptiveClassifierTrainingReport", "TransformersComplexityClassifier",
    "normalized_answer_exact_match",
]
