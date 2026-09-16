"""Learned relevance evaluation and ordered knowledge refinement for CRAG.

CRAG (https://arxiv.org/abs/2401.15884) uses a fine-tuned T5-large evaluator.
This module supplies a concrete pretrained cross-encoder alternative, not those
T5 weights: question/passage pairs are jointly encoded by a sequence classifier.
Raw logits must be calibrated on representative held-out relevance labels before
they are used as confidence probabilities. Merely applying sigmoid to a ranking
logit is deliberately not presented as empirical calibration.

The same evaluator scores documents and decomposed knowledge strips. A document
longer than the model window is evaluated through every bounded text window and
assigned the maximum window score; no tail is silently truncated. This extension
and the choice of checkpoint differ from the paper's exact experimental setup.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import math
import numbers
import re
from typing import Any

import numpy as np

from ..base import Document, _snapshot_document, _validate_non_negative_int, _validate_top_k
from .engine import (
    RetrievalAction, RetrievalGrade, _corrective_provenance, _probability,
    _validate_user_query, _validated_snapshots,
)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class LogisticCalibration:
    """Portable Platt-style calibration: sigmoid(scale * logit + bias).

    Provide fitted parameters from your validation data, or use ``fit``. A
    positive scale preserves the relevance rank ordering. Calibration on a toy
    corpus verifies mechanics only, not confidence validity on production data.
    """

    scale: float
    bias: float
    sample_count: int = 0

    def __post_init__(self) -> None:
        if _finite(self.scale, "scale") <= 0:
            raise ValueError("calibration scale must be positive")
        _finite(self.bias, "bias")
        _validate_non_negative_int(self.sample_count, name="sample_count")

    def probabilities(self, logits: Sequence[float]) -> list[float]:
        values = np.asarray([_finite(value, "logit") for value in logits], dtype=float)
        with np.errstate(over="ignore"):
            adjusted = self.scale * values + self.bias
            probabilities = np.exp(-np.logaddexp(0.0, -adjusted))
        if not np.isfinite(probabilities).all():
            raise ValueError("calibration produced non-finite probabilities")
        return probabilities.tolist()

    @classmethod
    def fit(cls, logits: Sequence[float], labels: Sequence[int], *, l2: float = 1e-3,
            max_iterations: int = 100) -> LogisticCalibration:
        """Fit regularized binary logistic calibration with damped Newton steps.

        Labels must contain both 0 (irrelevant) and 1 (relevant). Fit on a
        separate validation set, not the examples used to report model quality.
        An inverse relationship is rejected instead of flipping relevance.
        """
        values = np.asarray([_finite(value, "logit") for value in logits], dtype=float)
        if len(values) != len(labels) or len(values) < 2:
            raise ValueError("calibration requires matching logits and at least two labels")
        if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) or value not in (0, 1) for value in labels):
            raise ValueError("calibration labels must be binary integers")
        targets = np.asarray(labels, dtype=float)
        if set(targets.tolist()) != {0.0, 1.0}:
            raise ValueError("calibration requires both relevant and irrelevant labels")
        regularization = _finite(l2, "l2")
        if regularization <= 0:
            raise ValueError("l2 must be positive")
        iterations = _validate_top_k(max_iterations, name="max_iterations")
        mean, deviation = float(values.mean()), float(values.std())
        if not math.isfinite(deviation) or deviation <= 1e-12:
            raise ValueError("calibration logits must have non-zero finite variance")
        features = np.column_stack(((values - mean) / deviation, np.ones(len(values))))
        weights = np.array([1.0, 0.0])

        def objective(candidate: np.ndarray) -> float:
            scores = features @ candidate
            return float(np.mean(np.logaddexp(0, scores) - targets * scores)
                         + regularization * (candidate @ candidate) / 2)

        for _ in range(iterations):
            scores = features @ weights
            probabilities = np.exp(-np.logaddexp(0, -scores))
            gradient = features.T @ (probabilities - targets) / len(values) + regularization * weights
            hessian = (features.T * (probabilities * (1 - probabilities))) @ features / len(values)
            hessian += regularization * np.eye(2)
            direction = np.linalg.solve(hessian, gradient)
            step = 1.0
            previous = objective(weights)
            while step > 1e-8 and objective(weights - step * direction) > previous:
                step /= 2
            weights -= step * direction
            if float(np.linalg.norm(step * direction)) < 1e-8:
                break
        scale = float(weights[0] / deviation)
        if scale <= 0:
            raise ValueError("calibration labels invert model relevance; check the checkpoint or positive label")
        return cls(scale, float(weights[1] - scale * mean), sample_count=len(values))


class CrossEncoderRetrievalGrader:
    """Transformers cross-encoder with fitted confidence and CRAG decisions.

    The default checkpoint is a small English MS MARCO ranking model. It is
    loaded lazily; install ``cheragh[learned-retrieval]`` for torch/Transformers.
    A different classification checkpoint or an already-loaded model/tokenizer
    can be supplied. One-logit relevance and binary classification are supported.

    Fit ``calibrate`` on labeled question/document pairs or provide an existing
    ``LogisticCalibration``. Upper/lower thresholds are inclusive: any document
    at or above upper => Correct; all at or below lower => Incorrect; otherwise
    Ambiguous. Empty retrieval is Incorrect without loading a model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-TinyBERT-L2-v2", *,
                 calibration: LogisticCalibration | None = None, correct_threshold: float = 0.7,
                 incorrect_threshold: float = 0.3, model: Any = None, tokenizer: Any = None,
                 max_input_tokens: int = 512, max_windows_per_document: int = 64,
                 max_pairs: int = 4096, batch_size: int = 16, positive_label: int = 1,
                 device: str = "cpu", local_files_only: bool = False, cache_dir: str | None = None,
                 revision: str | None = None):
        if (model is None) != (tokenizer is None):
            raise ValueError("provide both model and tokenizer, or neither")
        if calibration is not None and not isinstance(calibration, LogisticCalibration):
            raise TypeError("calibration must be LogisticCalibration")
        self.correct_threshold = _probability(correct_threshold, name="correct_threshold")
        self.incorrect_threshold = _probability(incorrect_threshold, name="incorrect_threshold")
        if self.incorrect_threshold >= self.correct_threshold:
            raise ValueError("incorrect_threshold must be below correct_threshold")
        self.max_input_tokens = _validate_top_k(max_input_tokens, name="max_input_tokens")
        self.max_windows_per_document = _validate_top_k(max_windows_per_document, name="max_windows_per_document")
        self.max_pairs = _validate_top_k(max_pairs, name="max_pairs")
        self.batch_size = _validate_top_k(batch_size, name="batch_size")
        if isinstance(positive_label, bool) or not isinstance(positive_label, int) or positive_label not in (0, 1):
            raise ValueError("positive_label must be 0 or 1")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be non-empty")
        self.model_name, self.model, self.tokenizer = model_name, model, tokenizer
        self.calibration, self.device, self.positive_label = calibration, device, positive_label
        self.load_kwargs = {"local_files_only": local_files_only, "cache_dir": cache_dir, "revision": revision}

    def _load(self) -> Any:
        try:
            import torch
            if self.model is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.load_kwargs)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **self.load_kwargs)
        except ImportError as exc:
            raise ImportError("CrossEncoderRetrievalGrader requires cheragh[learned-retrieval]") from exc
        self.model.to(self.device)
        self.model.eval()
        return torch

    def raw_scores(self, pairs: Sequence[tuple[str, Document]]) -> list[float]:
        """Return max-window logits for calibration, without requiring a fit."""
        if len(pairs) > self.max_pairs:
            raise ValueError("CRAG evaluator exceeds max_pairs")
        if not pairs:
            return []
        normalized = [(_validate_user_query(query), _validated_snapshots([doc], name="documents")[0])
                      for query, doc in pairs]
        torch = self._load()
        limits = [self.max_input_tokens]
        for limit in (getattr(self.tokenizer, "model_max_length", None),
                      getattr(getattr(self.model, "config", None), "max_position_embeddings", None)):
            if isinstance(limit, int) and limit > 0:
                limits.append(limit)
        limit = min(limits)
        windows: list[tuple[int, str, str]] = []

        def size(query: str, passage: str) -> int:
            encoded = self.tokenizer(query, passage, truncation=False, verbose=False)
            return len(encoded["input_ids"])

        for index, (query, document) in enumerate(normalized):
            if size(query, "") >= limit:
                raise ValueError("CRAG query and special tokens leave no evaluator window for evidence")
            start, count = 0, 0
            while start < len(document.content):
                if count >= self.max_windows_per_document or len(windows) >= self.max_pairs:
                    raise ValueError("CRAG evaluator window budget exceeded; no source tail was discarded")
                lo, hi, end = start + 1, len(document.content), start
                while lo <= hi:
                    middle = (lo + hi) // 2
                    if size(query, document.content[start:middle]) <= limit:
                        end, lo = middle, middle + 1
                    else:
                        hi = middle - 1
                if end == start:
                    raise ValueError("CRAG evaluator window cannot fit one source character")
                if end < len(document.content) and not document.content[end].isspace():
                    boundary = max(document.content.rfind(" ", start, end), document.content.rfind("\n", start, end))
                    if boundary >= start:
                        end = boundary + 1
                windows.append((index, query, document.content[start:end]))
                start, count = end, count + 1
        results = [-math.inf] * len(normalized)
        with torch.inference_mode():
            for offset in range(0, len(windows), self.batch_size):
                batch = windows[offset:offset + self.batch_size]
                inputs = self.tokenizer([item[1] for item in batch], [item[2] for item in batch],
                                        padding=True, truncation=False, return_tensors="pt", verbose=False)
                if inputs["input_ids"].shape[1] > limit:
                    raise ValueError("CRAG tokenizer exceeded the evaluator input budget")
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                logits = self.model(**inputs).logits.detach().float().cpu().numpy()
                if logits.shape == (len(batch), 1):
                    scores = logits[:, 0]
                elif logits.shape == (len(batch), 2):
                    scores = logits[:, self.positive_label] - logits[:, 1 - self.positive_label]
                else:
                    raise ValueError("CRAG cross-encoder must emit one relevance logit or two class logits")
                if not np.isfinite(scores).all():
                    raise ValueError("CRAG cross-encoder logits must be finite")
                for (index, _, _), score in zip(batch, scores):
                    results[index] = max(results[index], float(score))
        return results

    def calibrate(self, pairs: Sequence[tuple[str, Document]], labels: Sequence[int], *,
                  l2: float = 1e-3) -> LogisticCalibration:
        """Fit and install calibration using held-out labeled relevance pairs."""
        if len(pairs) != len(labels):
            raise ValueError("calibration pairs and labels must have equal length")
        calibration = LogisticCalibration.fit(self.raw_scores(pairs), labels, l2=l2)
        self.calibration = calibration
        return calibration

    def score_documents(self, query: str, documents: Iterable[Document]) -> list[float]:
        query = _validate_user_query(query)
        snapshots = _validated_snapshots(documents, name="documents")
        if not snapshots:
            return []
        if self.calibration is None:
            raise ValueError("fit calibrate() on labeled validation pairs or supply LogisticCalibration before grading")
        return self.calibration.probabilities(self.raw_scores([(query, doc) for doc in snapshots]))

    def grade(self, query: str, documents: Iterable[Document]) -> RetrievalGrade:
        scores = self.score_documents(query, documents)
        score = max(scores, default=0.0)
        action = (RetrievalAction.CORRECT if score >= self.correct_threshold else
                  RetrievalAction.INCORRECT if not scores or score <= self.incorrect_threshold else RetrievalAction.AMBIGUOUS)
        return RetrievalGrade(score, action is RetrievalAction.CORRECT, "calibrated_cross_encoder_max_window",
                              len(scores), action)


class SemanticKnowledgeRefiner:
    """Decompose into sentence strips, grade each, and concatenate in source order.

    The original source ID and metadata are preserved. Refinement metadata maps
    every retained strip back to its exact character range in the input source.
    ``max_refined_tokens`` bounds recomposed evidence, excluding reader prompt
    overhead; overflowing evidence raises rather than silently dropping strips.
    """

    def __init__(self, grader: CrossEncoderRetrievalGrader, *, min_relevance: float = 0.5,
                 sentences_per_strip: int = 2, max_strips: int = 512, max_refined_tokens: int = 8192,
                 token_counter: Callable[[str], int] | None = None):
        if not callable(getattr(grader, "score_documents", None)):
            raise TypeError("grader must expose score_documents(query, documents)")
        self.grader = grader
        self.min_relevance = _probability(min_relevance, name="min_relevance")
        self.sentences_per_strip = _validate_top_k(sentences_per_strip, name="sentences_per_strip")
        self.max_strips = _validate_top_k(max_strips, name="max_strips")
        self.max_refined_tokens = _validate_top_k(max_refined_tokens, name="max_refined_tokens")
        if token_counter is not None and not callable(token_counter):
            raise TypeError("token_counter must be callable")
        self.token_counter = token_counter or (lambda text: len(text.encode("utf-8")))

    def decompose(self, document: Document) -> list[Document]:
        source = _validated_snapshots([document], name="documents")[0]
        spans: list[tuple[int, int]] = []
        start = 0
        for boundary in re.finditer(r"(?<=[.!?])\s+|\n+", source.content):
            if source.content[start:boundary.start()].strip():
                spans.append((start, boundary.end()))
            start = boundary.end()
        if source.content[start:].strip():
            spans.append((start, len(source.content)))
        strips: list[Document] = []
        for offset in range(0, len(spans), self.sentences_per_strip):
            section = spans[offset:offset + self.sentences_per_strip]
            start, end = section[0][0], section[-1][1]
            while start < end and source.content[start].isspace():
                start += 1
            while end > start and source.content[end - 1].isspace():
                end -= 1
            strip = _snapshot_document(source)
            strip.content = source.content[start:end]
            strip.metadata["crag_strip"] = {"index": len(strips), "start": start, "end": end}
            strips.append(strip)
        return strips

    def refine(self, query: str, documents: Sequence[Document]) -> list[Document]:
        query = _validate_user_query(query)
        sources = _validated_snapshots(documents, name="documents")
        groups = [self.decompose(source) for source in sources]
        strips = [strip for group in groups for strip in group]
        if len(strips) > self.max_strips:
            raise ValueError("CRAG knowledge refinement exceeds max_strips")
        # Injected graders may annotate/mutate their inputs. Retained evidence
        # and character spans must still refer to the untouched source text.
        scores = list(self.grader.score_documents(query, [_snapshot_document(strip) for strip in strips]))
        if len(scores) != len(strips):
            raise ValueError("CRAG evaluator must return one probability per knowledge strip")
        scores = [_probability(score, name="strip relevance") for score in scores]
        results: list[Document] = []
        position = 0
        for source, group in zip(sources, groups):
            probabilities = scores[position:position + len(group)]
            position += len(group)
            retained = [(strip, score) for strip, score in zip(group, probabilities) if score >= self.min_relevance]
            if not retained:
                continue
            refined = _snapshot_document(source)
            refined.content = "\n\n".join(strip.content for strip, _ in retained)
            provenance = _corrective_provenance(refined)
            provenance["refinement"] = {
                "strategy": "SemanticKnowledgeRefiner", "original_characters": len(source.content),
                "retained_characters": len(refined.content), "total_strips": len(group),
                "retained_strips": [{**strip.metadata["crag_strip"], "relevance": score} for strip, score in retained],
                "discarded_strips": [{**strip.metadata["crag_strip"], "relevance": score}
                                     for strip, score in zip(group, probabilities) if score < self.min_relevance],
            }
            results.append(refined)
        text = "\n\n".join(document.content for document in results)
        tokens = _validate_non_negative_int(self.token_counter(text), name="token_counter result")
        if text and tokens == 0:
            raise ValueError("token_counter must be positive for non-empty text")
        if tokens > self.max_refined_tokens:
            raise ValueError("CRAG refined knowledge exceeds max_refined_tokens; no strips were silently dropped")
        return results
