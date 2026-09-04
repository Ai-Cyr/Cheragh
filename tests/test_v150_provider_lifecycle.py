"""Provider transport ownership must survive early response termination."""
from types import SimpleNamespace as NS

import pytest

from cheragh import RAGEngine
from cheragh.base import OpenAILLMClient
from cheragh.llms import AzureOpenAIChatClient, LiteLLMClient, OpenAIChatClient


class ProviderStream:
    def __init__(self, events, *, fail=False):
        self.events = iter(events)
        self.fail = fail
        self.close_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.events)
        except StopIteration:
            if self.fail:
                raise RuntimeError("provider failed")
            raise

    def close(self):
        self.close_count += 1


def event(content):
    return NS(choices=[NS(delta=NS(content=content))])


def adapter(client_type, provider_stream):
    client = client_type.__new__(client_type)
    client.model = "model"
    client.client = NS(chat=NS(completions=NS(create=lambda **_: provider_stream)))
    if client_type is LiteLLMClient:
        client.litellm = NS(completion=lambda **_: provider_stream)
        client.default_kwargs = {}
    return client


@pytest.mark.parametrize("client_type", [OpenAILLMClient, OpenAIChatClient, AzureOpenAIChatClient, LiteLLMClient])
def test_completion_stream_accepts_usage_event_and_closes_on_exhaustion(client_type):
    transport = ProviderStream([event(None), event("answer"), NS(choices=[], usage=NS(total_tokens=3))])
    assert list(adapter(client_type, transport).stream("prompt", stream_options={"include_usage": True})) == ["answer"]
    assert transport.close_count == 1


@pytest.mark.parametrize("client_type", [OpenAILLMClient, AzureOpenAIChatClient, LiteLLMClient])
def test_completion_stream_closes_transport_when_consumer_stops(client_type):
    transport = ProviderStream([event("first"), event("second")])
    stream = adapter(client_type, transport).stream("prompt")
    assert next(stream) == "first"
    stream.close()
    stream.close()
    assert transport.close_count == 1


@pytest.mark.parametrize("client_type", [OpenAILLMClient, AzureOpenAIChatClient, LiteLLMClient])
def test_completion_stream_closes_transport_when_provider_raises(client_type):
    transport = ProviderStream([event("first")], fail=True)
    stream = adapter(client_type, transport).stream("prompt")
    assert next(stream) == "first"
    with pytest.raises(RuntimeError, match="provider failed"):
        next(stream)
    assert transport.close_count == 1


@pytest.mark.parametrize("mode", ["complete", "cancel", "invalid_chunk", "provider_error"])
def test_engine_explicitly_closes_retained_provider_iterator(mode):
    transport = ProviderStream(["first", 123 if mode == "invalid_chunk" else "second"], fail=mode == "provider_error")
    llm = NS(generate=lambda _: "unused", stream=lambda *_, **__: transport)
    engine = RAGEngine(NS(retrieve=lambda *_, **__: []), llm, trace_enabled=False)
    stream = engine.stream_with_response("question")
    assert next(stream) == "first"
    if mode == "cancel":
        stream.close()
        assert stream.response is None
    elif mode == "invalid_chunk":
        with pytest.raises(TypeError, match="str chunks"):
            list(stream)
    elif mode == "provider_error":
        with pytest.raises(RuntimeError, match="provider failed"):
            list(stream)
    else:
        assert list(stream) == ["second"]
        assert stream.response.answer == "firstsecond"
    assert transport.close_count == 1


def test_close_failure_preserves_original_provider_error_without_logging_secrets(caplog):
    transport = ProviderStream([event("first")], fail=True)

    def broken_close():
        raise RuntimeError("SECRET transport credentials")

    transport.close = broken_close
    stream = adapter(OpenAILLMClient, transport).stream("prompt")
    assert next(stream) == "first"
    with pytest.raises(RuntimeError, match="provider failed"):
        next(stream)
    assert "provider_stream_close_failed" in caplog.text
    assert "SECRET" not in caplog.text
