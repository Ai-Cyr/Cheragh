import json

import pytest

from cheragh.base import Document
from cheragh.chain_of_note import ChainOfNoteRAGEngine, ChainOfNoteRetriever
from cheragh.query_decomposition import QueryDecompositionRetriever
from cheragh.step_back import StepBackRAGEngine, StepBackRetriever


class Reader:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return next(self.outputs)


class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append(query)
        return self.documents.get(query, [])[:top_k]


def note(kind, quote="", **kwargs):
    return json.dumps(dict(kind=kind, summary="Concise assessment.", evidence_quotes=[quote] if quote else [],
                           parametric_knowledge="", limitations="", **kwargs))


def test_stepback_keeps_general_evidence_with_a_small_top_k():
    base = Retriever({"specific": [Document("case one", doc_id="a"), Document("case two", doc_id="b")],
                      "general": [Document("principle", doc_id="p")]})
    result = StepBackRetriever(base, Reader(["general"])).retrieve("specific", top_k=2)
    assert [doc.doc_id for doc in result] == ["a", "p"]


def test_stepback_generates_principles_before_applying_them_to_original_question():
    base = Retriever({"specific": [Document("specific fact", doc_id="a")],
                      "general": [Document("general principle", doc_id="p")]})
    llm = Reader(["general", "A principle with {braces}. [source: p]", "Conclusion [source: a] [source: p]"])
    result = StepBackRAGEngine(base, llm).ask("specific")
    assert len(llm.prompts) == 3
    assert "general principle" in llm.prompts[1]
    assert "specific fact" not in llm.prompts[1]
    assert "A principle with {braces}" in llm.prompts[2]
    assert "specific fact" in llm.prompts[2]
    assert {source.doc_id for source in result.sources} == {"a", "p"}
    assert result.metadata["stepback_query"] == "general"
    assert result.citation_validation.ok


def test_stepback_generates_no_citations_for_empty_evidence():
    llm = Reader(["general", "Unknown."])
    result = StepBackRAGEngine(Retriever({}), llm).ask("specific")
    assert not result.sources
    assert result.metadata["stepback_principles"] == "Preuves générales insuffisantes."


def test_chain_of_note_carries_previous_notes_and_preserves_actual_evidence():
    first = Document("Alpha happened in 2001.", doc_id="first")
    second = Document("Beta happened in 2002.", doc_id="second")
    llm = Reader([note("contextual", "Alpha happened in 2001."), note("direct", "Beta happened in 2002."),
                  "Beta followed Alpha. [source: first] [source: second]"])
    result = ChainOfNoteRAGEngine(Retriever({"query": [first, second]}), llm).ask("query")
    assert '"doc_id": "first"' in llm.prompts[1]
    assert all(document.content in result.prompt for document in [first, second])
    assert result.retrieved_documents[0].content == first.content
    assert result.metadata["notes"][1]["kind"] == "direct"
    assert result.citation_validation.ok


def test_chain_of_note_unknown_abstains_without_answer_call():
    llm = Reader([note("unknown")])
    result = ChainOfNoteRAGEngine(Retriever({"query": [Document("Unrelated text.")]}), llm).ask("query")
    assert result.metadata["abstained"]
    assert "Je ne sais pas" in result.answer
    assert len(llm.prompts) == 1


@pytest.mark.parametrize("generated", [note("direct", "Invented evidence."), note("direct"), "not json"])
def test_chain_of_note_rejects_unsupported_quotes_and_malformed_notes(generated):
    engine = ChainOfNoteRAGEngine(Retriever({"query": [Document("Actual evidence.")]}), Reader([generated]))
    with pytest.raises(ValueError):
        engine.ask("query")


def test_chain_of_note_reads_late_high_quality_candidate_and_long_tail():
    tail = "padding " * 500 + "Relevant fact at the end."
    llm = Reader(["PERTINENCE: partiellement pertinent\nINFORMATION_CLE: partial\nLIMITES: gaps",
                  "PERTINENCE: directement pertinent\nINFORMATION_CLE: correct\nLIMITES: aucune"])
    engine = ChainOfNoteRetriever(Retriever({"query": [Document("Partial."), Document(tail, doc_id="late")]}), llm)
    result = engine.retrieve("query", top_k=1)
    assert result[0].doc_id == "late"
    assert tail in llm.prompts[-1]


def test_query_decomposition_fuses_ranks_across_different_score_scales():
    base = Retriever({"query": [Document("One", doc_id="one", score=999)],
                      "subquestion": [Document("Two", doc_id="two", score=-99), Document("One", doc_id="one", score=-999)]})
    engine = QueryDecompositionRetriever(base, Reader(["subquestion\nSubquestion"]))
    result = engine.retrieve("query")
    assert base.calls == ["query", "subquestion"]
    assert result[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert result[0].metadata["matched_subquestions"] == ["query", "subquestion"]
