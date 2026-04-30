"""LLM provider integrations.

Provider SDKs are imported lazily so that ``cheragh`` remains lightweight
unless an integration is explicitly used.
"""
from __future__ import annotations

import json
from typing import Any, Iterator, Optional
from urllib import request

from .base import LLMClient, OpenAILLMClient


class OpenAIChatClient(OpenAILLMClient):
    """Alias around :class:`cheragh.base.OpenAILLMClient` for clearer naming."""


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
        for event in stream:  # pragma: no cover - provider integration
            chunk = event.choices[0].delta.content
            if chunk:
                yield chunk


class AnthropicClient(LLMClient):
    """Anthropic Messages API client."""

    def __init__(self, model: str = "claude-3-5-sonnet-latest", api_key: Optional[str] = None, client: Any | None = None):
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("AnthropicClient requires: pip install cheragh[anthropic]") from exc
            client = Anthropic(api_key=api_key) if api_key else Anthropic()
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
        for event in self.litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True,
            **params,
        ):  # pragma: no cover - provider integration
            chunk = event.choices[0].delta.content
            if chunk:
                yield chunk


class OllamaClient(LLMClient):
    """Small stdlib-only client for a local Ollama server."""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
            **kwargs,
        }
        req = request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=kwargs.pop("timeout", 120)) as response:  # noqa: S310 - user-provided local URL
            data = json.loads(response.read().decode("utf-8"))
        return str(data.get("response", ""))

    def stream(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> Iterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
            **kwargs,
        }
        req = request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=kwargs.pop("timeout", 120)) as response:  # noqa: S310 - user-provided local URL
            for raw_line in response:
                if not raw_line.strip():
                    continue
                data = json.loads(raw_line.decode("utf-8"))
                chunk = data.get("response")
                if chunk:
                    yield str(chunk)
