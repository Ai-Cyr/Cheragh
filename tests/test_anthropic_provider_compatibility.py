"""Anthropic defaults and real SDK request serialization, without API calls."""

import json
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cheragh.engine import _llm_from_config
from cheragh.llms import AnthropicClient


def test_anthropic_default_uses_active_immutable_model():
    assert AnthropicClient(client=object()).model == "claude-sonnet-4-6"


@pytest.mark.parametrize("model", [None, "application-selected-model"])
def test_anthropic_config_preserves_default_override_and_client_limits(model):
    created = []

    def create(**kwargs):
        created.append(kwargs)
        return AnthropicClient(client=object(), **kwargs)

    config = {"provider": "anthropic", "timeout_seconds": 7.5, "max_retries": 0}
    if model is not None:
        config["model"] = model
    with patch("cheragh.llms.AnthropicClient", side_effect=create):
        client = _llm_from_config(config)
    assert client.model == (model or "claude-sonnet-4-6")
    assert created[0]["timeout"] == 7.5
    assert created[0]["max_retries"] == 0


def test_anthropic_explicit_model_and_text_blocks_remain_supported():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(thinking="private"),
                                        SimpleNamespace(text="first"), SimpleNamespace(text=" second")])

    client = AnthropicClient(model="application-model", client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert client.generate("question", temperature=0.25, max_tokens=321, system="instructions") == "first second"
    assert calls == [{"model": "application-model", "messages": [{"role": "user", "content": "question"}],
                      "max_tokens": 321, "temperature": 0.25, "system": "instructions"}]


@pytest.mark.parametrize("extra_body", [None, {"metadata": {"user_id": "offline"}}, {"temperature": 0.5}])
def test_anthropic_sdk_without_temperature_preserves_extra_body(extra_body):
    calls = []

    def create(*, model, max_tokens, messages, extra_body=None):
        calls.append({"model": model, "max_tokens": max_tokens, "messages": messages, "extra_body": extra_body})
        return SimpleNamespace(content=[SimpleNamespace(text="answer")])

    original = None if extra_body is None else dict(extra_body)
    client = AnthropicClient(client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert client.generate("question", temperature=0.25, extra_body=extra_body) == "answer"
    assert calls == [{"model": "claude-sonnet-4-6", "max_tokens": 1024,
                      "messages": [{"role": "user", "content": "question"}],
                      "extra_body": {"temperature": 0.25, **(original or {})}}]
    assert extra_body == original


@pytest.mark.parametrize("inject_sdk", [False, True])
@pytest.mark.parametrize("extra_body", [None, {"temperature": 0.5, "metadata": {"user_id": "offline"}}])
def test_anthropic_real_sdk_serializes_compatible_messages_parameters(inject_sdk, extra_body):
    anthropic = pytest.importorskip("anthropic")
    # SDK 1.x migrated its HTTP client, transport and response types to httpx2.
    # Keep exercising the actual transport used by each supported SDK major.
    major = int(anthropic.__version__.split(".", maxsplit=1)[0])
    http = import_module("httpx2" if major >= 1 else "httpx")
    requests = []

    def respond(request):
        assert request.url.host == "anthropic.invalid"
        assert request.url.path == "/v1/messages"
        requests.append(json.loads(request.content))
        return http.Response(200, json={
            "id": "msg_offline", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "grounded answer"}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 3, "output_tokens": 2},
        })

    with http.Client(transport=http.MockTransport(respond)) as http_client:
        options = {"api_key": "offline-test-key", "base_url": "https://anthropic.invalid",
                   "http_client": http_client, "max_retries": 0}
        client = (AnthropicClient(client=anthropic.Anthropic(**options)) if inject_sdk
                  else AnthropicClient(**options))
        with client.client:
            answer = client.generate(
                "question", temperature=0.25, max_tokens=321, system="instructions", extra_body=extra_body,
            )
    assert answer == "grounded answer"
    assert requests == [{"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "question"}],
                         "max_tokens": 321, "temperature": 0.25, "system": "instructions", **(extra_body or {})}]
