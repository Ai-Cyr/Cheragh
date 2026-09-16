"""Semantic GraphRAG mechanisms, provenance and source-authorization contracts."""
import json
import re

import numpy as np
import pytest

from cheragh import Document, EmbeddingModel
from cheragh.community_graph.engine import CommunityGraphRAGEngine, _weighted_adjacency
from cheragh.community_graph.extraction import LLMGraphExtractor, SemanticKnowledgeGraph
from cheragh.community_graph.local import LocalGraphSearchConfig


class RecordingLLM:
    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.respond(prompt)


def entity(name, kind, description, evidence):
    return {"name": name, "type": kind, "description": description, "evidence": evidence}


def relation(source, target, description, evidence, weight=7):
    return {"source": source, "target": target, "description": description, "evidence": evidence, "weight": weight}


def extract_response(prompt):
    if prompt.startswith("GraphRAG CONSOLIDATE"):
        return "An optical telescope that observes distant galaxies."
    data = json.loads(prompt.split("\n", 1)[1])
    text, previous = data["TEXT"], data["PREVIOUS"]
    if previous["entities"]:
        if "Aurora Lab" in text and not any(row["name"] == "Aurora Lab" for row in previous["entities"]):
            return json.dumps({"entities": [entity("Aurora Lab", "ORGANIZATION", "An astronomy laboratory", "Aurora Lab")],
                               "relationships": [relation("Mira", "Aurora Lab", "Works at the laboratory", "Mira works at Aurora Lab.")]})
        return '{"entities": [], "relationships": []}'
    if "Mira" in text:
        return json.dumps({"entities": [
            entity("Mira", "PERSON", "An instrument designer", "Mira"),
            entity("Helios", "DEVICE", "An optical telescope", "Helios"),
        ], "relationships": [relation("Mira", "Helios", "Designed the optical telescope", "Mira designs Helios.")]})
    return json.dumps({"entities": [entity("Helios", "DEVICE", "Observes distant galaxies", "Helios")],
                       "relationships": []})


def corpus():
    return [Document("Mira designs Helios. Mira works at Aurora Lab.", doc_id="design"),
            Document("Helios observes distant galaxies.", doc_id="observations")]


class SemanticVectors(EmbeddingModel):
    """Controlled semantic encoder: the query has no entity-name token overlap."""
    def embed_query(self, text):
        if text == "stellar observation apparatus" or text.startswith("Helios"):
            return np.asarray([1.0, 0.0, 0.0])
        if text.startswith("Mira"):
            return np.asarray([0.0, 1.0, 0.0])
        return np.asarray([0.0, 0.0, 1.0])

    def embed_documents(self, texts):
        return np.asarray([self.embed_query(text) for text in texts])


def test_extractor_gleans_missing_relations_consolidates_descriptions_and_preserves_sources():
    llm = RecordingLLM(extract_response)
    graph = LLMGraphExtractor(llm, token_counter=len)(corpus())
    assert isinstance(graph, SemanticKnowledgeGraph)
    assert set(graph.entities()) == {"mira", "helios", "aurora lab"}
    assert len(graph.triples) == 2
    assert graph.entity_records["helios"].source_doc_ids == ("design", "observations")
    assert "distant galaxies" in graph.entity_records["helios"].description
    assert any(call[0].startswith("GraphRAG CONSOLIDATE") for call in llm.calls)
    assert all(kwargs["max_tokens"] == 3000 for _, kwargs in llm.calls)
    _, adjacency = _weighted_adjacency(graph)
    assert adjacency["mira"]["helios"] == 7
    assert all(triple.doc_id == "design" for triple in graph.triples)
    assert graph.triples[0].metadata["text_unit_indices"] == [0]


def test_extractor_keeps_isolated_entities_without_fabricating_cooccurrence_edges():
    llm = RecordingLLM(lambda _: json.dumps({"entities": [
        entity("Alpha", "ORGANIZATION", "A named organization", "Alpha"),
        entity("Beta", "ORGANIZATION", "Another organization", "Beta"),
    ], "relationships": []}))
    graph = LLMGraphExtractor(llm, max_gleanings=0)([Document("Alpha and Beta are mentioned.", doc_id="source")])
    assert graph.entities() == ["alpha", "beta"]
    assert graph.triples == []


@pytest.mark.parametrize("mutate", [
    lambda value: value["entities"][0].update(evidence="Fabricated evidence"),
    lambda value: value["relationships"][0].update(target="Unknown"),
    lambda value: value["relationships"][0].update(weight=float("nan")),
    lambda value: value["relationships"][0].update(evidence="Unsupported relationship"),
])
def test_extraction_rejects_unverifiable_or_invalid_model_output(mutate):
    data = {"entities": [entity("Alpha", "PERSON", "A person", "Alpha"),
                         entity("Beta", "PERSON", "A person", "Beta")],
            "relationships": [relation("Alpha", "Beta", "knows", "Alpha knows Beta")]}
    mutate(data)
    with pytest.raises(ValueError):
        LLMGraphExtractor(RecordingLLM(lambda _: json.dumps(data)), max_gleanings=0)([
            Document("Alpha knows Beta", doc_id="source"),
        ])


def test_text_unit_splitting_preserves_all_characters_and_stays_within_budget():
    text = "Astronomy evidence is split across bounded units. " * 20
    llm = RecordingLLM(lambda _: '{"entities": [], "relationships": []}')
    LLMGraphExtractor(llm, text_unit_tokens=120, max_input_tokens=1200,
                      token_counter=len)([Document(text, doc_id="source")])
    units = [json.loads(prompt.split("\n", 1)[1])["TEXT"] for prompt, _ in llm.calls]
    assert "".join(units) == text
    assert max(map(len, units)) <= 120
    assert max(len(prompt) for prompt, _ in llm.calls) <= 1200


def test_model_call_budget_fails_without_returning_a_partial_graph():
    llm = RecordingLLM(extract_response)
    with pytest.raises(ValueError, match="max_model_calls"):
        LLMGraphExtractor(llm, max_model_calls=1)(corpus())
    assert len(llm.calls) == 1


def semantic_engine(llm=None):
    return CommunityGraphRAGEngine(
        corpus(), graph_extractor=LLMGraphExtractor(RecordingLLM(extract_response)),
        embedding_model=SemanticVectors(),
        local_search_config=LocalGraphSearchConfig(max_entities=1, max_input_tokens=20000, token_counter=len),
        llm_client=llm or RecordingLLM(lambda _: "unused"),
    )


def test_semantic_local_search_links_synonym_query_to_entities_relationships_and_text():
    engine = semantic_engine()
    documents = engine.local_search("stellar observation apparatus", top_k=2)
    assert {doc.doc_id for doc in documents} == {"design", "observations"}
    assert all(doc.metadata["matched_entities"] == ["helios"] for doc in documents)
    assert isinstance(engine.graph, SemanticKnowledgeGraph)
    assert engine.graph.entity_records["helios"].description == engine._graph.entity_records["helios"].description


def test_local_answer_context_contains_entity_relationship_report_and_raw_source_records():
    def answer(prompt):
        assert "An optical telescope that observes distant galaxies" in prompt
        assert "Designed the optical telescope" in prompt
        assert "Mira designs Helios." in prompt
        assert "[source: community:" in prompt
        cite = re.search(r"\[source: graph-relationship:[^\]]+\]", prompt).group()
        return f"Mira designed Helios. {cite}"

    llm = RecordingLLM(answer)
    response = semantic_engine(llm).ask_local("stellar observation apparatus")
    assert response.citation_validation.ok
    assert response.metadata["architecture"] == "community_graph_rag_semantic_local"
    kinds = {source.metadata["retrieval_method"] for source in response.sources}
    assert {"community_graph_entity", "community_graph_relationship", "community_report",
            "community_graph_semantic_local"} <= kinds
    assert llm.calls[0][1]["max_tokens"] == 1500
    assert response.trace is not None


def test_local_semantic_descriptions_and_reports_cannot_leak_mixed_authorization():
    llm = RecordingLLM(lambda _: pytest.fail("No model should see unauthorized mixed entity descriptions"))
    engine = semantic_engine(llm)
    response = engine.ask_local("stellar observation apparatus", allowed_doc_ids=["design"])
    assert response.sources == []
    assert response.metadata["access_control_enabled"]
    assert "no_authorized_local_evidence" in response.warnings
    assert "distant galaxies" not in response.prompt
    assert llm.calls == []


def test_local_citations_fail_closed_and_prompt_budget_is_enforced():
    engine = semantic_engine(RecordingLLM(lambda _: "Invented [source: secret]"))
    response = engine.ask_local("stellar observation apparatus")
    assert response.sources == []
    assert "invalid_local_citations_answer_withheld" in response.warnings
    engine = CommunityGraphRAGEngine(
        corpus(), graph_extractor=LLMGraphExtractor(RecordingLLM(extract_response)),
        embedding_model=SemanticVectors(), local_search_config=LocalGraphSearchConfig(max_input_tokens=20),
    )
    with pytest.raises(ValueError, match="question and instructions"):
        engine.ask_local("stellar observation apparatus")
