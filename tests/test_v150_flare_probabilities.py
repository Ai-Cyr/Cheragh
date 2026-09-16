import math
from types import SimpleNamespace as NS

import pytest

from cheragh import Document, StaticLLMClient
from cheragh.flare import FLAREPipeline
from cheragh.generation import ConfidenceDraft, GeneratedToken, chat_completion_draft
from cheragh.llms import OpenAIChatClient


def completion(text, entries):
    return NS(choices=[NS(message=NS(content=text), logprobs=NS(content=entries))])


def token(text, probability):
    return GeneratedToken(text, math.log(probability))


class Recorder:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, top_k=5):
        self.queries.append(query)
        return [Document("Le délai est de 14 jours.", doc_id="policy")]


def test_generation_probabilities_mask_hallucinated_tokens_from_same_response():
    retriever = Recorder()
    draft = ConfidenceDraft("Le délai est de 30 jours.", (
        token("Le délai est de ", .95), token("30", .1), token(" jours.", .95),
    ))
    calls = []

    def generate(prompt):
        calls.append(prompt)
        return draft

    pipeline = FLAREPipeline(retriever, StaticLLMClient("14 jours. [source: policy]"),
                             max_iterations=1, draft_generator=generate)
    answer = pipeline.ask("Quel délai ?")
    assert retriever.queries == ["Le délai est de jours."]
    assert len(calls) == 1
    assert answer.citations == ["policy"]
    assert answer.metadata["iterations"][0]["uncertainty_rationale"] == "generation_token_logprobs"


def test_high_confidence_never_calls_retriever_or_second_generation():
    retriever = Recorder()
    pipeline = FLAREPipeline(retriever, StaticLLMClient("wrong"), max_iterations=1,
                             draft_generator=lambda _: ConfidenceDraft("Certain.", (token("Certain.", .99),)))
    assert pipeline.run("Question")["answer"] == "Certain."
    assert retriever.queries == []


def test_mask_threshold_is_independent_of_retrieval_trigger():
    retriever = Recorder()
    draft = ConfidenceDraft("Alpha beta gamma", (token("Alpha ", .9), token("beta ", .6), token("gamma", .2)))
    pipeline = FLAREPipeline(retriever, StaticLLMClient("answer"), max_iterations=1,
                             draft_generator=lambda _: draft, confidence_threshold=.3, masking_threshold=.7)
    pipeline.run("Question")
    assert retriever.queries == ["Alpha"]


def test_fully_masked_draft_uses_original_query():
    retriever = Recorder()
    pipeline = FLAREPipeline(retriever, StaticLLMClient("answer"), max_iterations=1,
                             draft_generator=lambda _: ConfidenceDraft("Invented!", (token("Invented!", .01),)))
    pipeline.run("Original question")
    assert retriever.queries == ["Original question"]


def test_whitespace_and_unicode_byte_tokens_are_aligned():
    entries = [NS(token=" ", bytes=[32], logprob=-.1),
               NS(token="replacement", bytes=[195], logprob=-.2),
               NS(token="replacement", bytes=[169], logprob=-2.0)]
    draft = chat_completion_draft(completion(" é", entries))
    assert draft.tokens == (GeneratedToken(" ", -.1), GeneratedToken("é", -2.0))
    assert draft.text == " é"


@pytest.mark.parametrize("entries", [None, [], [NS(token="wrong", bytes=None, logprob=-1.)]])
def test_missing_or_misaligned_logprobs_fail_instead_of_fake_confidence(entries):
    with pytest.raises(ValueError):
        chat_completion_draft(completion("real", entries))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, .1, True])
def test_invalid_logprob_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        GeneratedToken("x", value)


@pytest.mark.parametrize("raw", [[195], [255], [True], [-1], []])
def test_invalid_utf8_rejected(raw):
    with pytest.raises(ValueError):
        chat_completion_draft(completion("x", [NS(token="x", bytes=raw, logprob=-.1)]))


def test_openai_adapter_requests_real_logprobs_without_extra_generation():
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return completion("A", [NS(token="A", bytes=[65], logprob=-.4)])

    client = OpenAIChatClient.__new__(OpenAIChatClient)
    client.client = NS(chat=NS(completions=NS(create=create)))
    client.model = "configured-model"
    draft = client.generate_with_confidence("prompt", max_completion_tokens=32)
    assert draft == ConfidenceDraft("A", (GeneratedToken("A", -.4),))
    assert len(requests) == 1
    assert requests[0]["logprobs"] is True
    assert requests[0]["n"] == 1
    assert requests[0]["max_completion_tokens"] == 32
    with pytest.raises(ValueError):
        client.generate_with_confidence("prompt", logprobs=False)


def test_draft_generator_failure_does_not_fallback_to_text_generation():
    llm = StaticLLMClient("invented")
    pipeline = FLAREPipeline(Recorder(), llm, draft_generator=lambda _: None)
    with pytest.raises(TypeError, match="ConfidenceDraft"):
        pipeline.run("Question")
    assert llm.prompts == []
