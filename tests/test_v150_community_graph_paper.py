"""Paper-method contracts: graph coverage, bounded aggregation and ACL safety."""
import json
from contextvars import ContextVar
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from cheragh import Document
from cheragh.community_graph import (
    Community, CommunityGraphRAGEngine, GlobalMapReduceConfig,
    LeidenCommunityDetector, LLMCommunitySummarizer,
)
from cheragh.graph import KnowledgeGraph, KnowledgeTriple
from cheragh.security import AccessPolicy, Principal


class ScriptedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response(prompt)


def fixtures():
    graph = KnowledgeGraph()
    docs = []
    for i in range(3):
        graph.add_triple(KnowledgeTriple(f"Entity{i}A", "knows", f"Entity{i}B", f"doc{i}"))
        docs.append(Document(f"Topic {i}: information of global importance.", doc_id=f"doc{i}",
                             metadata={"tenant_id": "tenant-a" if i < 2 else "tenant-b"}))
    return docs, graph


def map_then_reduce(prompt):
    if prompt.startswith("GraphRAG MAP"):
        data = json.loads(prompt.split("\n", 1)[1])["DATA"]
        return json.dumps({"points": [
            {"answer": f"Evidence from {item['report_id']} [source: {item['report_id']}]", "score": 80}
            for item in data
        ]})
    data = json.loads(prompt.split("\n", 1)[1])["partial_answers"]
    return " ".join(item["answer"] for item in data)


def test_global_maps_all_reports_without_lexical_top_k_pruning():
    docs, graph = fixtures()
    llm = ScriptedLLM(map_then_reduce)
    engine = CommunityGraphRAGEngine(docs, graph=graph, llm_client=llm, top_k=1)
    answer = engine.ask_global_map_reduce("Overview", config=GlobalMapReduceConfig(max_map_output_tokens=3000))
    assert len(answer.metadata["mapped_report_ids"]) == 3
    assert len(answer.sources) == 3
    assert answer.citation_validation.ok
    assert answer.metadata["coverage_complete"]
    assert answer.trace is not None
    assert {item.metadata["source_doc_ids"][0] for item in answer.sources} == {"doc0", "doc1", "doc2"}


def test_zero_unknown_and_invalid_helpfulness_never_reach_reduce():
    docs, graph = fixtures()

    def response(prompt):
        if prompt.startswith("GraphRAG MAP"):
            return json.dumps({"points": [
                {"answer": "Zero [source: community:0]", "score": 0},
                {"answer": "Low [source: community:0]", "score": 10},
                {"answer": "High [source: community:1]", "score": 99},
                {"answer": "Fake [source: community:999]", "score": 100},
                {"answer": "Missing citation", "score": 100},
                {"answer": "Bool [source: community:0]", "score": True},
                {"answer": "NaN [source: community:0]", "score": float("nan")},
            ]})
        assert "High" in prompt and "Low" in prompt
        assert prompt.index("High") < prompt.index("Low")
        assert all(text not in prompt for text in ("Zero", "Fake", "Missing citation", "Bool", "NaN"))
        return "High [source: community:1]"

    llm = ScriptedLLM(response)
    answer = CommunityGraphRAGEngine(docs, graph=graph, llm_client=llm).ask_global_map_reduce("Overview")
    assert answer.metadata["rejected_map_points"] == 4
    assert "invalid_map_points_discarded" in answer.warnings
    assert answer.metadata["map_points"][0]["score"] == 99


def test_report_chunks_preserve_full_content_and_prompt_budgets():
    docs, graph = fixtures()
    huge = "Résumé: " + "climate finance. " * 100
    llm = ScriptedLLM(map_then_reduce)
    config = GlobalMapReduceConfig(max_map_input_tokens=900, max_map_calls=100,
                                  max_reduce_input_tokens=3000, max_map_output_tokens=1500)
    engine = CommunityGraphRAGEngine(docs, graph=graph, summarizer=lambda *args: huge, llm_client=llm)
    engine.ask_global_map_reduce("Themes", config=config, token_counter=len)
    assembled = {}
    for prompt, kwargs in llm.calls:
        if prompt.startswith("GraphRAG MAP"):
            assert len(prompt) <= 900
            assert kwargs["max_tokens"] == 1500
            for fragment in json.loads(prompt.split("\n", 1)[1])["DATA"]:
                assembled.setdefault(fragment["report_id"], "")
                assembled[fragment["report_id"]] += fragment["text"]
        else:
            assert len(prompt) <= 3000
    assert assembled == {report.report_id: f"{report.title}\n{report.summary}" for report in engine.reports}


def test_map_budget_overflow_fails_before_model_calls():
    docs, graph = fixtures()
    llm = ScriptedLLM(map_then_reduce)
    engine = CommunityGraphRAGEngine(docs, graph=graph, summarizer=lambda *args: "long " * 1000, llm_client=llm)
    with pytest.raises(ValueError, match="max_map_calls"):
        engine.ask_global_map_reduce("Overview", config=GlobalMapReduceConfig(max_map_input_tokens=1000, max_map_calls=1))
    assert llm.calls == []


def test_acl_mixed_reports_and_unknown_provenance_never_reach_prompts():
    docs, graph = fixtures()
    graph.add_triple(KnowledgeTriple("Entity0A", "knows", "Entity0B", "doc2"))
    graph.add_triple(KnowledgeTriple("Entity1A", "knows", "Entity1B", "missing-private"))
    llm = ScriptedLLM(map_then_reduce)
    engine = CommunityGraphRAGEngine(docs, graph=graph, llm_client=llm)
    answer = engine.ask_global_map_reduce("Overview", principal=Principal("alice", tenant_ids={"tenant-a"}),
                                         access_policy=AccessPolicy())
    assert llm.calls == []
    assert answer.sources == []
    assert answer.metadata["report_count"] == 0
    assert answer.metadata["excluded_report_count"] == 3
    assert "private" not in json.dumps(answer.to_dict())


def test_acl_subset_retains_only_completely_authorized_reports():
    docs, graph = fixtures()
    llm = ScriptedLLM(map_then_reduce)
    answer = CommunityGraphRAGEngine(docs, graph=graph, llm_client=llm).ask_global_map_reduce(
        "Overview", allowed_doc_ids={"doc0"})
    assert answer.metadata["mapped_report_ids"] == ["community:0"]
    assert len(answer.sources) == 1
    assert all("Entity1" not in prompt and "Entity2" not in prompt for prompt, _ in llm.calls)
    assert not answer.metadata["coverage_complete"]
    with pytest.raises(TypeError, match="not a string"):
        CommunityGraphRAGEngine(docs, graph=graph).ask_global_map_reduce("Overview", allowed_doc_ids="doc0")


def test_detector_cannot_hide_private_graph_dependencies_from_acl():
    docs, graph = fixtures()
    roots = CommunityGraphRAGEngine(docs, graph=graph).communities
    # The injected detector claims public provenance but includes private
    # triples. Report construction must recover the actual dependency chain.
    malicious = [Community(root.community_id, root.entities, root.triples, ("doc0",)) for root in roots]
    llm = ScriptedLLM(map_then_reduce)
    engine = CommunityGraphRAGEngine(docs, graph=graph, community_detector=lambda graph: malicious,
                                     llm_client=llm)
    answer = engine.ask_global_map_reduce("Overview", allowed_doc_ids={"doc0"})
    assert answer.metadata["mapped_report_ids"] == ["community:0"]
    assert all("Entity1" not in prompt and "Entity2" not in prompt for prompt, _ in llm.calls)


def test_parallel_map_keeps_request_context_and_preserves_coverage():
    docs, graph = fixtures()
    tenant = ContextVar("test_graph_tenant", default=None)

    def response(prompt):
        assert tenant.get() == "tenant-a"
        return map_then_reduce(prompt)

    llm = ScriptedLLM(response)
    engine = CommunityGraphRAGEngine(docs, graph=graph, llm_client=llm)
    token = tenant.set("tenant-a")
    try:
        answer = engine.ask_global_map_reduce("Overview", token_counter=len,
            config=GlobalMapReduceConfig(max_map_input_tokens=900, max_concurrency=2))
    finally:
        tenant.reset(token)
    assert answer.metadata["map_calls"] > 1
    assert len(answer.metadata["mapped_report_ids"]) == 3


def test_reduce_budget_prefers_most_helpful_points():
    docs, graph = fixtures()

    def response(prompt):
        if prompt.startswith("GraphRAG MAP"):
            return json.dumps({"points": [
                {"answer": "Low priority " + "detail " * 20 + "[source: community:0]", "score": 10},
                {"answer": "High priority " + "detail " * 20 + "[source: community:1]", "score": 90},
            ]})
        assert "High priority" in prompt
        assert "Low priority" not in prompt
        return "Selected [source: community:1]"

    llm = ScriptedLLM(response)
    answer = CommunityGraphRAGEngine(docs, graph=graph, llm_client=llm).ask_global_map_reduce("Overview",
        config=GlobalMapReduceConfig(max_reduce_input_tokens=700), token_counter=len)
    assert "reduce_budget_excluded_points" in answer.warnings
    assert len(answer.metadata["map_points"]) == 1


def test_uncited_or_unknown_reduce_answer_is_withheld():
    docs, graph = fixtures()
    llm = ScriptedLLM(lambda prompt: map_then_reduce(prompt) if prompt.startswith("GraphRAG MAP")
                      else "Invented finding [source: secret-document]")
    answer = CommunityGraphRAGEngine(docs, graph=graph, llm_client=llm).ask_global_map_reduce("Overview")
    assert "Invented" not in answer.answer
    assert "invalid_reduce_citations_answer_withheld" in answer.warnings
    assert answer.sources == []
    assert answer.citations == []


def test_backend_failure_does_not_silently_return_partial_answer():
    docs, graph = fixtures()

    def fail(prompt):
        raise TimeoutError("provider unavailable")

    engine = CommunityGraphRAGEngine(docs, graph=graph, llm_client=ScriptedLLM(fail))
    with pytest.raises(TimeoutError):
        engine.ask_global_map_reduce("Overview")


def test_hierarchy_preserves_shallow_leaves_and_does_not_duplicate_entities():
    docs, graph = fixtures()
    baseline = CommunityGraphRAGEngine(docs, graph=graph)
    roots = baseline.communities
    hierarchy = [Community(0, roots[0].entities + roots[1].entities,
                           roots[0].triples + roots[1].triples, ("doc0", "doc1")),
                 Community(1, roots[2].entities, roots[2].triples, ("doc2",)),
                 Community(2, roots[0].entities, roots[0].triples, ("doc0",), 1, 0),
                 Community(3, roots[1].entities, roots[1].triples, ("doc1",), 1, 0)]
    llm = ScriptedLLM(map_then_reduce)
    engine = CommunityGraphRAGEngine(docs, graph=graph, community_detector=lambda graph: hierarchy, llm_client=llm)
    result = engine.ask_global_map_reduce("Overview", level=1)
    assert set(result.metadata["mapped_report_ids"]) == {"community:1", "community:2", "community:3"}
    assert engine.local_search("Entity0A")[0].metadata["community_ids"] == [2]
    leaves = engine.ask_global_map_reduce("Overview", level=None)
    assert set(leaves.metadata["mapped_report_ids"]) == set(result.metadata["mapped_report_ids"])
    hierarchy.append(Community(4, roots[0].entities, (), ("doc0",), 1, 0))
    with pytest.raises(ValueError, match="overlap"):
        CommunityGraphRAGEngine(docs, graph=graph, community_detector=lambda graph: hierarchy)


def test_hierarchy_rejects_incomplete_children():
    docs, graph = fixtures()
    roots = CommunityGraphRAGEngine(docs, graph=graph).communities
    malformed = [*roots, Community(3, (roots[0].entities[0],), (), ("doc0",), 1, 0)]
    with pytest.raises(ValueError, match="cover its parent"):
        CommunityGraphRAGEngine(docs, graph=graph, community_detector=lambda graph: malformed)


def test_leiden_adapter_uses_weighted_edges_and_preserves_isolates():
    docs, graph = fixtures()
    graph.add_triple(KnowledgeTriple("Entity0A", "knows", "Entity0B", "other-document"))
    graph.entity_to_doc_ids["isolated"].add("doc0")
    calls = []

    def backend(edges, **kwargs):
        calls.append((edges, kwargs))
        return [SimpleNamespace(node=node, cluster=i, parent_cluster=None, level=0)
                for i in range(3) for node in (f"entity{i}a", f"entity{i}b")]

    module = ModuleType("graspologic.partition")
    module.hierarchical_leiden = backend
    with patch.dict("sys.modules", {"graspologic.partition": module}):
        result = LeidenCommunityDetector(max_cluster_size=2, random_seed=7)(graph)
    assert ("entity0a", "entity0b", 2.0) in calls[0][0]
    assert calls[0][1] == {"max_cluster_size": 2, "resolution": 1.0, "random_seed": 7}
    assert len(result) == 4
    assert ("isolated",) in [item.entities for item in result]


def test_llm_summarizer_builds_children_before_parent_with_bounded_context():
    docs, graph = fixtures()
    for doc in docs:
        doc.content = "large " * 500
    roots = CommunityGraphRAGEngine(docs, graph=graph).communities
    hierarchy = [Community(0, tuple(entity for root in roots for entity in root.entities),
                           tuple(triple for root in roots for triple in root.triples), tuple(doc.doc_id for doc in docs))]
    hierarchy.extend(Community(index + 1, root.entities, root.triples, root.doc_ids, 1, 0)
                     for index, root in enumerate(roots))
    llm = ScriptedLLM(lambda prompt: "Child report with provenance.")
    engine = CommunityGraphRAGEngine(docs, graph=graph, community_detector=lambda graph: hierarchy,
        summarizer=LLMCommunitySummarizer(llm, max_input_tokens=700, max_output_tokens=100, token_counter=len))
    assert len(engine.reports) == 4
    assert len(llm.calls) == 4
    assert all(len(prompt) <= 700 for prompt, _ in llm.calls)
    assert "Child report with provenance" in llm.calls[-1][0]
    assert '"report_id": "community:1"' in llm.calls[-1][0]


@pytest.mark.parametrize("kwargs", [{"max_map_calls": True}, {"max_concurrency": 0}, {"random_seed": -1}])
def test_budget_config_is_strict(kwargs):
    with pytest.raises((TypeError, ValueError)):
        GlobalMapReduceConfig(**kwargs)


def test_real_leiden_backend_when_extra_is_installed():
    pytest.importorskip("graspologic.partition")
    graph = KnowledgeGraph()
    # A chain of eight triangles yields both shallow leaves and communities
    # which recursively split; exercise the actual backend's parent IDs.
    for group in range(8):
        for first in range(3):
            for second in range(first + 1, 3):
                graph.add_triple(KnowledgeTriple(f"g{group}n{first}", "r", f"g{group}n{second}", f"d{group}"))
        if group:
            graph.add_triple(KnowledgeTriple(f"g{group - 1}n0", "r", f"g{group}n0", f"b{group}"))
    detector = LeidenCommunityDetector(max_cluster_size=3)
    first = detector(graph)
    second = detector(graph)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert any(item.level > 0 for item in first)
    assert {entity.casefold() for item in first if item.level == 0 for entity in item.entities} == set(graph.entities())
