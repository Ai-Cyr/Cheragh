"""The offline default must quote retrieved evidence, not prompt instructions."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cheragh import Document, RAGEngine
from cheragh.base import ExtractiveLLMClient
from cheragh.pipeline import AdvancedRAGPipeline, DEFAULT_ANSWER_PROMPT_FR


@pytest.mark.parametrize("streaming", [False, True])
def test_default_engine_quotes_actual_source_and_keeps_citations_valid(streaming):
    document = Document("The production release identifier is amber-lighthouse.", doc_id="release-42")
    engine = RAGEngine(SimpleNamespace(retrieve=lambda *_, **__: [document]), require_citations=True)
    if streaming:
        stream = engine.stream_with_response("What is the production release identifier?")
        answer = "".join(stream)
        response = stream.response
        assert response is not None
        assert response.answer == answer
    else:
        response = engine.ask("What is the production release identifier?")
    assert "amber-lighthouse" in response.answer
    assert response.answer.endswith("[source: release-42]")
    assert "Tu es un assistant" not in response.answer
    assert response.citations == ["release-42"]
    assert "unknown_citations" not in response.warnings
    assert "missing_citations" not in response.warnings


def test_empty_standard_context_does_not_echo_system_instructions_or_question():
    prompt = DEFAULT_ANSWER_PROMPT_FR.format(context="", query="User question must not become an answer")
    assert ExtractiveLLMClient().generate(prompt) == "Aucun contexte exploitable fourni."


def test_long_source_retains_complete_citation_and_omits_location_and_other_sources():
    documents = [
        Document("amber-lighthouse " * 200, doc_id="release-42", metadata={"page": 7}),
        Document("unrelated second source", doc_id="other"),
    ]
    prompt = DEFAULT_ANSWER_PROMPT_FR.format(
        context=AdvancedRAGPipeline._format_context(documents), query="Which release?"
    )
    answer = ExtractiveLLMClient().generate(prompt)
    assert len(answer) <= 1200
    assert answer.startswith("amber-lighthouse")
    assert answer.endswith("[source: release-42]")
    assert "location:" not in answer
    assert "unrelated second source" not in answer


def test_context_only_custom_prompt_remains_supported():
    answer = ExtractiveLLMClient().generate("[source: release-42]\nThe identifier is amber-lighthouse.")
    assert answer == "The identifier is amber-lighthouse. [source: release-42]"


def test_uncited_excerpts_for_hierarchical_summary_omit_prompt_instructions():
    prompt = "Summarize the following evidence.\n\nExtraits:\namber-lighthouse is the release.\n\nRésumé:"
    assert ExtractiveLLMClient().generate(prompt) == "amber-lighthouse is the release."


def test_plain_text_fallback_remains_compatible():
    assert ExtractiveLLMClient().generate("The release is amber-lighthouse.") == "The release is amber-lighthouse."
