import pytest

from cheragh import AccessPolicy, Document, MultiTenantRAGEngine, Principal, RAGEngine, StaticLLMClient
from cheragh.base import BaseRetriever


class Retriever(BaseRetriever):
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=5):
        return self.documents[:top_k]


@pytest.mark.parametrize("mode", ["retrieve", "ask_retriever", "ask_engine"])
@pytest.mark.parametrize("admin", [False, True])
def test_selected_collection_isolates_even_multi_scope_users(mode, admin):
    documents = [
        Document("other tenant", {"tenant_id": "b", "collection_id": "contracts"}, "foreign-tenant"),
        Document("other collection", {"tenant_id": "a", "collection_id": "invoices"}, "foreign-collection"),
        Document("selected", {"tenant_id": "a", "collection_id": "contracts"}, "selected"),
        Document("implicitly scoped by registry", doc_id="implicit"),
    ]
    retriever = Retriever(documents)
    target = (
        RAGEngine(retriever, StaticLLMClient("answer"))
        if mode == "ask_engine" else retriever
    )
    engine = MultiTenantRAGEngine()
    engine.add_collection("a", "contracts", target)
    principal = Principal(
        "user", roles={"admin"} if admin else set(),
        tenant_ids={"a", "b"}, collection_ids={"contracts", "invoices"},
    )
    operation = engine.retrieve if mode == "retrieve" else engine.ask
    result = operation("query", tenant_id="a", principal=principal, top_k=4)
    selected = result.retrieved_documents if mode == "ask_engine" else result
    assert [doc.doc_id for doc in selected] == ["selected", "implicit"]
    if mode == "ask_engine":
        assert "other tenant" not in result.prompt
        assert "other collection" not in result.prompt
    assert documents[-1].metadata == {}
    assert principal.tenant_ids == {"a", "b"}


@pytest.mark.parametrize("mode", ["retrieve", "ask_retriever", "ask_engine"])
def test_tenant_retrieval_finds_evidence_beyond_denied_prefix(mode):
    documents = [Document("denied", {"allowed_users": ["someone-else"]}, f"denied-{i}") for i in range(25)]
    documents.append(Document("allowed", doc_id="allowed"))
    retriever = Retriever(documents)
    target = RAGEngine(retriever, StaticLLMClient("answer")) if mode == "ask_engine" else retriever
    engine = MultiTenantRAGEngine()
    engine.add_collection("a", "contracts", target)
    operation = engine.retrieve if mode == "retrieve" else engine.ask
    result = operation("query", tenant_id="a", principal=Principal("user", tenant_ids={"a"}), top_k=1)
    selected = result.retrieved_documents if mode == "ask_engine" else result
    assert [doc.doc_id for doc in selected] == ["allowed"]


@pytest.mark.parametrize("mode", ["retrieve", "ask_retriever", "ask_engine"])
def test_custom_batch_authorization_is_preserved(mode):
    class DenyAll(AccessPolicy):
        def filter_documents(self, documents, principal=None):
            return []

    retriever = Retriever([Document("private", doc_id="private")])
    target = RAGEngine(retriever, StaticLLMClient("answer")) if mode == "ask_engine" else retriever
    engine = MultiTenantRAGEngine(access_policy=DenyAll())
    engine.add_collection("a", "contracts", target)
    operation = engine.retrieve if mode == "retrieve" else engine.ask
    result = operation("query", tenant_id="a", principal=Principal("user", tenant_ids={"a"}))
    selected = result.retrieved_documents if mode == "ask_engine" else result
    assert selected == []
