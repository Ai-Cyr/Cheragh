"""Self-RAG reflection-token scoring without a model or training dependency.

These adapters implement the normalized relevance/support/utility calculations
used by the Self-RAG authors. Providers must supply real token probabilities
from a compatible model, including every token in a requested group. A text
LLM response or lexical overlap cannot supply those probabilities.

The query-only retrieval gate implements the initial Yes/(Yes+No) threshold
policy. It does not implement the later Continue-to-Use-Evidence transition,
segment generation, or beam search. The existing SelfRAGEngine therefore
remains an inference approximation when this gate is injected.

References:
    https://arxiv.org/abs/2310.11511
    https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_short_form.py
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real
from types import MappingProxyType
from typing import Any

from .engine import RetrievalDecision


class ReflectionTokenGroup(str, Enum):
    """Complete token groups using the original Self-RAG token spellings.

    INITIAL_RETRIEVAL explicitly describes a provider's two-token initial
    distribution. RETRIEVAL additionally requires Continue to Use Evidence.
    Missing tokens never implicitly receive a zero probability.
    """

    INITIAL_RETRIEVAL = "initial_retrieval"
    RETRIEVAL = "retrieval"
    RELEVANCE = "relevance"
    SUPPORT = "support"
    UTILITY = "utility"


_TOKENS: Mapping[ReflectionTokenGroup, tuple[str, ...]] = MappingProxyType({
    ReflectionTokenGroup.INITIAL_RETRIEVAL: ("[Retrieval]", "[No Retrieval]"),
    ReflectionTokenGroup.RETRIEVAL: (
        "[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]",
    ),
    ReflectionTokenGroup.RELEVANCE: ("[Relevant]", "[Irrelevant]"),
    ReflectionTokenGroup.SUPPORT: (
        "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
    ),
    ReflectionTokenGroup.UTILITY: tuple(f"[Utility:{index}]" for index in range(1, 6)),
})


def _real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, excluding booleans")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _group(value: Any) -> ReflectionTokenGroup:
    if not isinstance(value, (str, ReflectionTokenGroup)):
        raise TypeError("group must be a ReflectionTokenGroup or its string value")
    return ReflectionTokenGroup(value)


def _complete_values(
    group: ReflectionTokenGroup,
    values: Mapping[str, float],
    *,
    log_probabilities: bool,
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError("token probabilities must be a mapping")
    if any(not isinstance(token, str) for token in values):
        raise TypeError("reflection token names must be strings")
    expected = set(_TOKENS[group])
    actual = set(values)
    if actual != expected:
        raise ValueError(
            f"Incomplete or unknown {group.value} reflection token group: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    validated: dict[str, float] = {}
    for token in _TOKENS[group]:
        value = _real(values[token], name=token)
        if log_probabilities:
            if value > 0:
                raise ValueError("log-probabilities must be <= 0")
        elif not 0 <= value <= 1:
            raise ValueError("probabilities must be between 0 and 1")
        validated[token] = value
    if not log_probabilities:
        total = math.fsum(validated.values())
        if total <= 0:
            raise ValueError("reflection token probabilities must have positive total mass")
        # A group is a subset of the model vocabulary, so its mass may be < 1.
        if total > 1.0 + 1e-9:
            raise ValueError("reflection token probability mass must not exceed 1")
    return validated


def _normalize(values: Mapping[str, float], *, log_probabilities: bool) -> dict[str, float]:
    if log_probabilities:
        maximum = max(values.values())
        masses = {token: math.exp(value - maximum) for token, value in values.items()}
    else:
        # Scaling first also preserves ratios for subnormal probability inputs.
        maximum = max(values.values())
        if maximum <= 0:
            raise ValueError("selected reflection tokens have zero total probability")
        masses = {token: value / maximum for token, value in values.items()}
    denominator = math.fsum(masses.values())
    return {token: value / denominator for token, value in masses.items()}


@dataclass(frozen=True)
class ReflectionTokenDistribution:
    """Immutable, complete, within-group normalized token probabilities.

    ``probabilities`` must contain exactly the token names for ``group``. Raw
    vocabulary probabilities may sum to less than one and are normalized over
    that group. Use ``from_logprobs`` for finite, non-positive log-probabilities;
    exact zeros can be supplied through this probability constructor instead.
    ``model_id`` records the checkpoint or provider revision when supplied.
    """

    group: ReflectionTokenGroup
    probabilities: Mapping[str, float]
    model_id: str | None = None
    _log_probabilities: Mapping[str, float] | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        group = _group(self.group)
        if self.model_id is not None:
            if not isinstance(self.model_id, str):
                raise TypeError("model_id must be a string or None")
            if not self.model_id.strip():
                raise ValueError("model_id must not be blank")
        values = _complete_values(group, self.probabilities, log_probabilities=False)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "probabilities", MappingProxyType(_normalize(values, log_probabilities=False)))

    @classmethod
    def from_logprobs(
        cls,
        group: ReflectionTokenGroup,
        log_probabilities: Mapping[str, float],
        *,
        model_id: str | None = None,
    ) -> ReflectionTokenDistribution:
        """Normalize with a stable log-softmax, retaining conditional ratios."""

        selected_group = _group(group)
        values = _complete_values(selected_group, log_probabilities, log_probabilities=True)
        result = cls(selected_group, _normalize(values, log_probabilities=True), model_id=model_id)
        # Preserve the detached log inputs for conditional Yes/(Yes+No)
        # normalization when Continue has overwhelming probability mass.
        object.__setattr__(result, "_log_probabilities", MappingProxyType(values))
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize detached diagnostics without exposing mutable internals."""

        return {
            "group": self.group.value,
            "probabilities": dict(self.probabilities),
            "model_id": self.model_id,
            "input_kind": "log_probabilities" if self._log_probabilities is not None else "probabilities",
        }


@dataclass(frozen=True)
class ReflectionScore:
    """Decomposed score of one model-generated passage/segment candidate."""

    relevance: float
    support: float
    utility: float
    total: float
    sequence_probability: float | None = None
    model_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("relevance", "support", "utility", "total"):
            object.__setattr__(self, name, _real(getattr(self, name), name=name))
        if not 0 <= self.relevance <= 1 or not 0 <= self.support <= 1:
            raise ValueError("relevance and support must be between 0 and 1")
        if not -1 <= self.utility <= 1:
            raise ValueError("utility must be between -1 and 1")
        if self.sequence_probability is not None:
            probability = _real(self.sequence_probability, name="sequence_probability")
            if not 0 <= probability <= 1:
                raise ValueError("sequence_probability must be between 0 and 1")
            object.__setattr__(self, "sequence_probability", probability)
        if isinstance(self.model_ids, (str, bytes)) or not isinstance(self.model_ids, Sequence):
            raise TypeError("model_ids must be a sequence of non-empty strings")
        if any(not isinstance(model_id, str) or not model_id.strip() for model_id in self.model_ids):
            raise ValueError("model_ids must contain non-empty strings")
        object.__setattr__(self, "model_ids", tuple(self.model_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevance": self.relevance,
            "support": self.support,
            "utility": self.utility,
            "sequence_probability": self.sequence_probability,
            "total": self.total,
            "model_ids": list(self.model_ids),
        }


@dataclass(frozen=True)
class ReflectionTokenScorer:
    """Self-RAG's normalized soft scoring; no probabilities are fabricated.

    The three groups must describe the same candidate at their respective
    reflection-token positions. The provider owns this alignment. If supplied,
    ``mean_sequence_logprob`` contributes ``exp(mean_sequence_logprob)`` with
    unit weight, matching the optional sequence score in the authors' code.
    """

    relevance_weight: float = 1.0
    support_weight: float = 1.0
    utility_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in ("relevance_weight", "support_weight", "utility_weight"):
            value = _real(getattr(self, name), name=name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

    def score(
        self,
        relevance: ReflectionTokenDistribution,
        support: ReflectionTokenDistribution,
        utility: ReflectionTokenDistribution,
        *,
        mean_sequence_logprob: float | None = None,
    ) -> ReflectionScore:
        for name, distribution, expected in (
            ("relevance", relevance, ReflectionTokenGroup.RELEVANCE),
            ("support", support, ReflectionTokenGroup.SUPPORT),
            ("utility", utility, ReflectionTokenGroup.UTILITY),
        ):
            if not isinstance(distribution, ReflectionTokenDistribution):
                raise TypeError(f"{name} must be a ReflectionTokenDistribution")
            if distribution.group != expected:
                raise ValueError(f"{name} requires the {expected.value} token group")
        sequence_probability = None
        if mean_sequence_logprob is not None:
            logprob = _real(mean_sequence_logprob, name="mean_sequence_logprob")
            if logprob > 0:
                raise ValueError("mean_sequence_logprob must be <= 0")
            sequence_probability = math.exp(logprob)
        relevance_score = relevance.probabilities["[Relevant]"]
        support_score = (
            support.probabilities["[Fully supported]"]
            + 0.5 * support.probabilities["[Partially supported]"]
        )
        utility_score = math.fsum(
            weight * utility.probabilities[f"[Utility:{index}]"]
            for index, weight in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0), start=1)
        )
        try:
            total = math.fsum((
                self.relevance_weight * relevance_score,
                self.support_weight * support_score,
                self.utility_weight * utility_score,
                sequence_probability if sequence_probability is not None else 0.0,
            ))
        except OverflowError as exc:
            raise ValueError("weighted reflection score must be finite") from exc
        model_ids = tuple(dict.fromkeys(
            distribution.model_id for distribution in (relevance, support, utility)
            if distribution.model_id is not None
        ))
        return ReflectionScore(
            relevance_score, support_score, utility_score, total,
            sequence_probability=sequence_probability, model_ids=model_ids,
        )


class ReflectionTokenRetrievalGate:
    """Initial retrieval gate for explicit, model-provided probabilities.

    ``provider(query)`` returns either the INITIAL_RETRIEVAL two-token group or
    the complete RETRIEVAL three-token group. The threshold applies to
    P(Retrieval)/(P(Retrieval)+P(No Retrieval)), not a ratio of log-probabilities.
    Equality does not trigger retrieval, following the authors' strict ``>``.

    Continue-to-Use-Evidence is excluded from this initial decision because
    the existing query-only gate has no evidence state. If both initial token
    masses are zero, this adapter fails explicitly. It cannot decide whether
    to reuse evidence during segment decoding.
    """

    def __init__(
        self,
        provider: Callable[[str], ReflectionTokenDistribution],
        *,
        threshold: float = 0.5,
    ) -> None:
        if not callable(provider):
            raise TypeError("provider must be callable")
        threshold = _real(threshold, name="threshold")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.provider = provider
        self.threshold = threshold

    def decide(self, query: str) -> RetrievalDecision:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        query = " ".join(query.split())
        if not query:
            raise ValueError("query must not be blank")
        distribution = self.provider(query)
        if not isinstance(distribution, ReflectionTokenDistribution):
            raise TypeError("provider must return a ReflectionTokenDistribution")
        if distribution.group not in (
            ReflectionTokenGroup.INITIAL_RETRIEVAL, ReflectionTokenGroup.RETRIEVAL,
        ):
            raise ValueError("retrieval gate requires an initial_retrieval or retrieval token group")
        tokens = _TOKENS[ReflectionTokenGroup.INITIAL_RETRIEVAL]
        logs = distribution._log_probabilities
        if logs is not None:
            probabilities = _normalize({token: logs[token] for token in tokens}, log_probabilities=True)
        else:
            probabilities = _normalize(
                {token: distribution.probabilities[token] for token in tokens}, log_probabilities=False,
            )
        probability = probabilities["[Retrieval]"]
        should_retrieve = probability > self.threshold
        return RetrievalDecision(
            should_retrieve=should_retrieve,
            confidence=probability if should_retrieve else 1.0 - probability,
            reason=(
                "reflection_token_initial_yes_no; "
                f"p_retrieval={probability:.17g}; threshold={self.threshold:.17g}; "
                f"model_id={distribution.model_id or 'unspecified'}"
            ),
        )


__all__ = [
    "ReflectionScore",
    "ReflectionTokenDistribution",
    "ReflectionTokenGroup",
    "ReflectionTokenRetrievalGate",
    "ReflectionTokenScorer",
]
