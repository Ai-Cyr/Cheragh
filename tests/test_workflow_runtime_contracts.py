"""Composition must retain the resource and document boundaries of RAGEngine."""
from itertools import repeat

import pytest

from cheragh import Document, StaticLLMClient
from cheragh.pipeline import AdvancedRAGPipeline
from cheragh.workflow import CompressNode, GenerateNode, RAGWorkflow, RetrieveNode, TransformQueryNode


class RepeatingRetriever:
    def __init__(self, document):
        self.document = document

    def retrieve(self, query, top_k=5):
        return repeat(self.document)


def test_pipeline_bounds_iterable_and_preserves_source_identity():
    pipeline = AdvancedRAGPipeline(
        RepeatingRetriever(Document("Evidence", doc_id="evidence")),
        StaticLLMClient("Supported [source: evidence]"),
        top_k=2,
    )
    result = pipeline.run("question")
    assert len(result["sources"]) == 2
    assert result["sources"][0]["doc_id"] == "evidence"


def test_retrieve_node_isolates_nested_source_metadata():
    source = Document("Evidence", metadata={"acl": {"roles": ["reader"]}}, doc_id="evidence")
    node = RetrieveNode(RepeatingRetriever(source), top_k=1)
    documents = node.run({"query": "question"})["documents"]
    documents[0].metadata["acl"]["roles"].clear()
    assert source.metadata["acl"]["roles"] == ["reader"]


def test_compression_cannot_mutate_input_evidence_and_bounds_output():
    class MutatingCompressor:
        def compress(self, query, documents):
            documents[0].metadata["acl"].clear()
            return repeat(documents[0])

    source = Document("Evidence", metadata={"acl": ["reader"]}, doc_id="evidence")
    state = {"query": "question", "documents": [source]}
    result = CompressNode(MutatingCompressor(), top_k=2).run(state)
    assert len(result["documents"]) == 2
    assert source.metadata["acl"] == ["reader"]


def test_generate_node_limits_context_before_calling_provider():
    class CapturingLLM:
        def generate(self, prompt):
            assert prompt.count("[source: evidence]") == 2
            return "answer"

    node = GenerateNode(CapturingLLM(), top_k=2)
    result = node.run({"query": "question", "documents": repeat(Document("Evidence", doc_id="evidence"))})
    assert result["answer"] == "answer"


@pytest.mark.parametrize("invalid", [None, 12, "", "  "])
def test_workflow_nodes_and_pipeline_reject_invalid_queries(invalid):
    source = RepeatingRetriever(Document("Evidence", doc_id="evidence"))
    with pytest.raises((ValueError, TypeError)):
        RetrieveNode(source).run({"query": invalid})
    with pytest.raises((ValueError, TypeError)):
        AdvancedRAGPipeline(source, StaticLLMClient("answer")).run(invalid)
    with pytest.raises((ValueError, TypeError)):
        GenerateNode(StaticLLMClient("answer")).run({"query": invalid})


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True])
def test_composed_retrieval_rejects_invalid_scores_before_generation(score):
    source = RepeatingRetriever(Document("Evidence", score=score))
    with pytest.raises((TypeError, ValueError)):
        RetrieveNode(source).run({"query": "question"})
    with pytest.raises((TypeError, ValueError)):
        AdvancedRAGPipeline(source, StaticLLMClient("answer")).run("question")


def test_transform_node_bounds_and_validates_custom_query_stream():
    class Transformer:
        def transform(self, query):
            return (f"variant {index}" for index in range(1000000))

    variants = TransformQueryNode(Transformer()).run({"query": "question"})["query_variants"]
    assert len(variants) == 32


def test_non_text_generation_is_rejected():
    class InvalidLLM:
        def generate(self, prompt):
            return {"answer": "not the protocol"}

    with pytest.raises(TypeError, match="must return a str"):
        GenerateNode(InvalidLLM()).run({"query": "question"})
    with pytest.raises(TypeError, match="must return a str"):
        AdvancedRAGPipeline(RepeatingRetriever(Document("Evidence")), InvalidLLM()).run("question")


def test_named_workflow_inputs_ignore_unrelated_state_and_execute_once():
    calls = []

    def answer(query, *, documents):
        calls.append(query)
        return {"answer": query + documents[0].content}

    workflow = RAGWorkflow().add_node("answer", answer)
    result = workflow.run("question", {"documents": [Document("evidence")], "trace": object()})
    assert result.answer == "questionevidence"
    assert calls == ["question"]


def test_unknown_start_is_rejected_even_on_empty_workflow():
    with pytest.raises(KeyError, match="Unknown start"):
        RAGWorkflow().run(start_at="missing")


def test_mapping_component_keeps_state_convention():
    workflow = RAGWorkflow().add_node("answer", lambda state: {"answer": state["query"]})
    assert workflow.ask("question").answer == "question"


def test_optional_positional_mapping_component_keeps_state_convention():
    def node(data=None):
        return {"answer": data["query"]}

    assert RAGWorkflow().add_node("node", node).ask("question").answer == "question"
    assert RAGWorkflow().add_node("node", lambda *args: {"answer": args[0]["query"]}).ask("question").answer == "question"


def test_internal_type_error_in_named_component_is_not_retried():
    calls = []

    def fail(query):
        calls.append(query)
        raise TypeError("provider failure")

    with pytest.raises(TypeError, match="provider failure"):
        RAGWorkflow().add_node("fail", fail).ask("question")
    assert calls == ["question"]
