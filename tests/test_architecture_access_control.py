"""Authorization must cover every evidence path used by architecture wrappers."""
from types import SimpleNamespace

import pytest

from cheragh import Document, HashingEmbedding, RAGEngine, StaticLLMClient
from cheragh.corrective import CorrectiveRAGEngine
from cheragh.long_rag import LongRAGEngine, LongRAGRetriever
from cheragh.security import AccessControlledRAGEngine, AccessControlledRetriever, AccessPolicy, Principal
from cheragh.self_rag import SelfRAGEngine, ScriptedEvidenceCritic
from cheragh.self_rag.segmented import SegmentedSelfRAGResult, SegmentSearchTrace


class ListRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=5):
        return self.documents[:top_k]


class RecordingLLM:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.answer


@pytest.mark.parametrize("return_details", [False, True])
def test_corrective_external_evidence_is_authorized_before_grading_and_generation(return_details):
    secret = Document("PRIVATE_NEVER_SEND evidence", metadata={"tenant_id": "other"}, doc_id="secret")
    llm = RecordingLLM("PRIVATE_NEVER_SEND evidence")
    corrective = CorrectiveRAGEngine(
        RAGEngine(ListRetriever([]), llm), external_retriever=ListRetriever([secret]),
        max_retries=0, return_details=return_details,
    )
    result = AccessControlledRAGEngine(corrective).ask("evidence", principal=Principal("u", tenant_ids={"allowed"}))
    assert result.retrieved_documents == []
    assert llm.prompts == []
    assert result.metadata["access_control"]["denied_documents"] == 1
    assert corrective.external_retriever.documents == [secret]


def test_corrective_keeps_authorized_external_evidence():
    safe = Document("authorized evidence", metadata={"tenant_id": "allowed"}, doc_id="safe")
    llm = RecordingLLM("authorized evidence [source: safe]")
    corrective = CorrectiveRAGEngine(
        RAGEngine(ListRetriever([]), llm), external_retriever=ListRetriever([safe]), max_retries=0,
    )
    result = AccessControlledRAGEngine(corrective).ask("evidence", principal=Principal("u", tenant_ids={"allowed"}))
    assert [document.doc_id for document in result.retrieved_documents] == ["safe"]
    assert len(llm.prompts) == 1


def test_self_rag_wrapper_preserves_result_and_serializes_access_metadata():
    safe = Document("evidence", metadata={"tenant_id": "allowed"}, doc_id="safe")
    engine = SelfRAGEngine(ListRetriever([safe]), StaticLLMClient("evidence"), evidence_critic=ScriptedEvidenceCritic())
    result = AccessControlledRAGEngine(engine).ask("evidence", principal=Principal("u", tenant_ids={"allowed"}))
    assert result.status == "supported"
    assert result.to_dict()["metadata"]["access_control"]["user_id"] == "u"


def test_segmented_self_rag_result_serializes_access_metadata():
    result = SegmentedSelfRAGResult("query", "answer", (), [], SegmentSearchTrace(), 0.0, "completed")
    result.metadata["access_control"] = {"enabled": True}
    assert result.to_dict()["metadata"]["access_control"]["enabled"] is True


def test_long_rag_wrapper_authorizes_original_documents_before_grouping():
    documents = [
        Document("authorized evidence", metadata={"tenant_id": "allowed"}, doc_id="safe"),
        Document("PRIVATE_NEVER_SEND evidence", metadata={"tenant_id": "other"}, doc_id="secret"),
    ]
    retriever = LongRAGRetriever(documents, HashingEmbedding(), adjacency={"safe": ["secret"]})
    llm = RecordingLLM("authorized evidence [source: safe]")
    wrapped = AccessControlledRAGEngine(LongRAGEngine(retriever, llm))
    result = wrapped.ask("evidence", principal=Principal("u", tenant_ids={"allowed"}))
    assert [document.doc_id for document in result.retrieved_documents] == ["safe"]
    assert len(llm.prompts) == 1
    assert "PRIVATE_NEVER_SEND" not in llm.prompts[0]
    assert result.metadata["group_source_doc_ids"] == [["safe"]]
    assert result.metadata["access_control_enabled"] is True


def test_long_rag_scoped_view_cannot_override_authorization():
    retriever = LongRAGRetriever([Document("evidence", doc_id="secret", metadata={"tenant_id": "other"})], HashingEmbedding())
    scoped = AccessControlledRetriever(retriever, Principal("u", tenant_ids={"allowed"}))
    with pytest.raises(ValueError, match="cannot override principal"):
        scoped.retrieve_groups("evidence", principal=Principal("admin", roles={"admin"}))
    assert scoped.retrieve("evidence") == []


def test_authorized_hierarchical_views_are_refreshed_after_revocation():
    safe = Document("evidence", metadata={"tenant_id": "allowed"}, doc_id="safe")

    class HierarchicalRetriever:
        def for_principal(self, principal, *, policy):
            return ListRetriever(policy.filter_documents([safe], principal))

    principal = Principal("u", tenant_ids={"allowed"})
    scoped = AccessControlledRetriever(HierarchicalRetriever(), principal)
    assert len(scoped.retrieve("evidence")) == 1
    principal.tenant_ids.clear()
    assert scoped.retrieve("evidence") == []


@pytest.mark.parametrize("hierarchical", [False, True])
def test_access_control_respects_candidate_limits_for_lazy_retrievers(hierarchical):
    document = Document("evidence", metadata={"tenant_id": "allowed"}, doc_id="safe")

    class GuardedRetriever:
        def retrieve(self, query, top_k=5):
            for _ in range(top_k):
                yield document
            raise AssertionError("read past candidate budget")

    raw = GuardedRetriever()
    source = SimpleNamespace(for_principal=lambda principal, policy: raw) if hierarchical else raw
    scoped = AccessControlledRetriever(source, Principal("u", tenant_ids={"allowed"}), overfetch_factor=1)
    result = scoped.retrieve("evidence", top_k=1)
    assert len(result) == 1


def test_access_policy_result_does_not_alias_source_metadata():
    document = Document("evidence", metadata={"tenant_id": "allowed", "nested": {"value": 1}})
    result = AccessPolicy().filter_documents([document], Principal("u", tenant_ids={"allowed"}))
    result[0].metadata["nested"]["value"] = 2
    assert document.metadata["nested"]["value"] == 1
