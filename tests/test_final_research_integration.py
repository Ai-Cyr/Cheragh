"""Public integration boundaries not covered by individual algorithm suites."""
import hashlib
import re

import pytest

from cheragh import Document, FLAREPipeline, TransformersPropositionizer


class Retriever:
    def retrieve(self, query, top_k=5):
        return [Document("Paris is the capital of France.")]


def test_flare_assigns_stable_source_ids_without_mutating_sources():
    class LLM:
        def generate(self, prompt):
            if "Extraits pertinents" in prompt:
                source = re.search(r"\[source: ([^]]+)\]", prompt).group(1)
                return f"Paris is the capital of France. [source: {source}]"
            return "Paris is the capital of France."

    result = FLAREPipeline(Retriever(), LLM(), max_iterations=1).ask("France?")
    expected = "flare::" + hashlib.sha256(b"Paris is the capital of France.").hexdigest()
    assert result.retrieved_documents[0].doc_id == expected
    assert result.citations[0] == expected
    assert not result.citation_validation.unknown_citations


def test_flare_only_exact_sentinel_stops_generation():
    class LLM:
        def __init__(self):
            self.answers = iter(["The marker [DONE] has a documented meaning.", "En conclusion is a French phrase.", "There is still more to explain.", "[DONE]"])

        def generate(self, prompt):
            return next(self.answers)

    result = FLAREPipeline(Retriever(), LLM(), max_iterations=4, min_draft_length=1000).ask("Explain these expressions")
    assert "still more" in result.answer
    assert len(result.metadata["iterations"]) == 3


@pytest.fixture
def propositionizer_parts():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    class Tokenizer:
        raw = '["Paris is in France.", "Paris is in France."]'
        size = 4

        def __call__(self, text, **kwargs):
            self.text, self.kwargs = text, kwargs
            return {"input_ids": torch.ones((1, self.size), dtype=torch.long), "attention_mask": torch.ones((1, self.size), dtype=torch.long)}

        def decode(self, output, **kwargs):
            return self.raw

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = []

        def generate(self, **kwargs):
            assert not torch.is_grad_enabled()
            self.calls.append(kwargs)
            return torch.ones((1, 2), dtype=torch.long)

    return Model(), Tokenizer()


def test_propositionizer_uses_authors_format_and_strict_json(propositionizer_parts):
    model, tokenizer = propositionizer_parts
    adapter = TransformersPropositionizer(model=model, tokenizer=tokenizer, max_input_tokens=8, max_new_tokens=19)
    result = adapter.extract(Document("Paris is in France.", metadata={"title": "France", "section": "Cities"}))
    assert tokenizer.text == "Title: France. Section: Cities. Content: Paris is in France."
    assert tokenizer.kwargs["truncation"] is False
    assert result == ["Paris is in France."]
    assert model.calls[0]["max_new_tokens"] == 19
    assert model.calls[0]["do_sample"] is False
    for invalid in ('{"proposition": "fact"}', '["fact", null]', '[""]', 'not JSON'):
        tokenizer.raw = invalid
        with pytest.raises(ValueError):
            adapter.extract(Document("Paris is in France."))


def test_propositionizer_refuses_truncation_before_generation(propositionizer_parts):
    model, tokenizer = propositionizer_parts
    tokenizer.size = 9
    adapter = TransformersPropositionizer(model=model, tokenizer=tokenizer, max_input_tokens=8)
    with pytest.raises(ValueError, match="chunk the document"):
        adapter.extract(Document("Long source"))
    assert model.calls == []
