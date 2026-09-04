"""Immutable generation-time token probabilities, without provider state."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class GeneratedToken:
    """A decoded token/span with its natural-log probability.

    Spans may group multiple byte tokens forming one Unicode character. Their
    log probability is the minimum constituent value, preserving FLARE's
    any-low-token trigger. Whitespace is retained for exact text alignment.
    """

    text: str
    logprob: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("Token text must be a non-empty string")
        if isinstance(self.logprob, bool) or not isinstance(self.logprob, Real):
            raise TypeError("logprob must be a real number")
        value = float(self.logprob)
        if not math.isfinite(value) or value > 0:
            raise ValueError("logprob must be finite and <= 0")
        object.__setattr__(self, "logprob", value)

    @property
    def probability(self) -> float:
        return math.exp(self.logprob)


@dataclass(frozen=True)
class ConfidenceDraft:
    """Text and probabilities from the SAME model response, aligned exactly."""

    text: str
    tokens: tuple[GeneratedToken, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Draft text must be a string")
        tokens = tuple(self.tokens)
        if any(not isinstance(token, GeneratedToken) for token in tokens):
            raise TypeError("Draft tokens must be GeneratedToken instances")
        if "".join(token.text for token in tokens) != self.text:
            raise ValueError("Token probabilities must align exactly with generated text")
        object.__setattr__(self, "tokens", tokens)


def chat_completion_draft(response: Any) -> ConfidenceDraft:
    """Decode Chat Completions logprobs; reject absent or misaligned values.

    Reference: https://developers.openai.com/api/reference/resources/chat
    Uses UTF-8 bytes where provided because token strings can split characters.
    """
    choices = getattr(response, "choices", None)
    if not choices or len(choices) != 1:
        raise ValueError("Confidence generation requires exactly one completion")
    choice = choices[0]
    text = getattr(choice.message, "content", None)
    entries = getattr(getattr(choice, "logprobs", None), "content", None)
    if not isinstance(text, str) or entries is None:
        raise ValueError("Model did not return text token log probabilities")
    tokens: list[GeneratedToken] = []
    pending = bytearray()
    pending_logprob = 0.0
    for entry in entries:
        raw = getattr(entry, "bytes", None)
        value = GeneratedToken("validation", entry.logprob).logprob
        if raw is None:
            if pending:
                raise ValueError("Incomplete UTF-8 token sequence")
            tokens.append(GeneratedToken(entry.token, value))
            continue
        if not isinstance(raw, (tuple, list)) or not raw or any(
            isinstance(byte, bool) or not isinstance(byte, int) or not 0 <= byte <= 255
            for byte in raw
        ):
            raise ValueError("Invalid token UTF-8 bytes")
        pending.extend(raw)
        pending_logprob = min(pending_logprob, value)
        try:
            decoded = pending.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.reason == "unexpected end of data":
                continue
            raise ValueError("Invalid token UTF-8 bytes") from exc
        tokens.append(GeneratedToken(decoded, pending_logprob))
        pending.clear()
        pending_logprob = 0.0
    if pending:
        raise ValueError("Incomplete UTF-8 token sequence")
    return ConfidenceDraft(text, tuple(tokens))
