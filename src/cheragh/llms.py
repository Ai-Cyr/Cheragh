"""LLM provider integrations.

Provider SDKs are imported lazily so that ``cheragh`` remains lightweight
unless an integration is explicitly used.
"""
from __future__ import annotations

import json
import math
from numbers import Real
from typing import Any, Iterator, Optional
from urllib import request

from .base import LLMClient, OpenAILLMClient, _iter_chat_completion_text
from .generation import ConfidenceDraft, chat_completion_draft


def _validate_timeout_seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("timeout must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("timeout must be finite and > 0")
    return normalized


class OpenAIChatClient(OpenAILLMClient):
    """Alias around :class:`cheragh.base.OpenAILLMClient` for clearer naming."""

    def generate_with_confidence(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> ConfidenceDraft:
        """Return generation-time logprobs for FLARE with a supporting model.

        Unsupported models or absent logprobs fail explicitly. No mutable
        last-response cache is used, so concurrent requests cannot mix drafts.
        """
        if any(key in kwargs for key in ("stream", "n", "logprobs")):
            raise ValueError("Confidence generation controls stream, n and logprobs")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            logprobs=True,
            n=1,
            **kwargs,
        )
        return chat_completion_draft(response)


class AzureOpenAIChatClient(LLMClient):
    """Azure OpenAI chat-completions client."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-02-01",
        client: Any | None = None,
        **client_kwargs: Any,
    ):
        if client is None:
            try:
                from openai import AzureOpenAI
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "AzureOpenAIChatClient requires the optional dependency 'openai'. "
                    "Install with: pip install cheragh[openai]"
                ) from exc
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                **client_kwargs,
            )
        self.client = client
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def stream(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        yield from _iter_chat_completion_text(stream)


class AnthropicClient(LLMClient):
    """Anthropic Messages API client."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        api_key: Optional[str] = None,
        client: Any | None = None,
        **client_kwargs: Any,
    ):
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("AnthropicClient requires: pip install cheragh[anthropic]") from exc
            client = (
                Anthropic(api_key=api_key, **client_kwargs)
                if api_key
                else Anthropic(**client_kwargs)
            )
        self.client = client
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024, **kwargs: Any) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        parts = []
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


class LiteLLMClient(LLMClient):
    """Client using LiteLLM's provider-agnostic completion API."""

    def __init__(self, model: str, **default_kwargs: Any):
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("LiteLLMClient requires: pip install cheragh[litellm]") from exc
        self.litellm = litellm
        self.model = model
        self.default_kwargs = default_kwargs

    def generate(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> str:
        params = {**self.default_kwargs, **kwargs}
        response = self.litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **params,
        )
        return response.choices[0].message.content or ""

    def stream(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> Iterator[str]:
        params = {**self.default_kwargs, **kwargs}
        stream = self.litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True,
            **params,
        )
        yield from _iter_chat_completion_text(stream)


class OllamaClient(LLMClient):
    """Small stdlib-only client for a local Ollama server."""

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = _validate_timeout_seconds(timeout_seconds)

    def generate(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> str:
        request_kwargs = dict(kwargs)
        timeout = _validate_timeout_seconds(request_kwargs.pop("timeout", self.timeout_seconds))
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
            **request_kwargs,
        }
        req = request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - user-provided local URL
            data = json.loads(response.read().decode("utf-8"))
        return str(data.get("response", ""))

    def stream(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> Iterator[str]:
        request_kwargs = dict(kwargs)
        timeout = _validate_timeout_seconds(request_kwargs.pop("timeout", self.timeout_seconds))
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
            **request_kwargs,
        }
        req = request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - user-provided local URL
            for raw_line in response:
                if not raw_line.strip():
                    continue
                data = json.loads(raw_line.decode("utf-8"))
                chunk = data.get("response")
                if chunk:
                    yield str(chunk)
