from cheragh import (
    AccessControlledRetriever,
    AccessPolicy,
    Document,
    FeedbackLoop,
    MultiTenantRAGEngine,
    Principal,
    SQLRAGEngine,
    StructuredRAG,
)
from cheragh.base import BaseRetriever


class DemoRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 5):
        return [
            Document("Contrat tenant A : préavis de 30 jours.", metadata={"tenant_id": "tenant-a", "classification": "internal"}, doc_id="a-contract"),
            Document("Contrat tenant B : préavis de 60 jours.", metadata={"tenant_id": "tenant-b", "classification": "internal"}, doc_id="b-contract"),
        ][:top_k]


sales = SQLRAGEngine.from_records(
    "sales",
    [
        {"client": "Alpha", "revenue": 100, "quarter": "Q1"},
        {"client": "Beta", "revenue": 180, "quarter": "Q1"},
    ],
)
print(sales.ask("Quel est le revenu total ?").answer)

structured = StructuredRAG.from_records([
    {"name": "Alice", "score": 10},
    {"name": "Bob", "score": 5},
])
print(structured.ask("Qui a le score le plus grand ?").answer)

principal = Principal(user_id="u1", tenant_ids={"tenant-a"}, max_classification="internal")
secure_retriever = AccessControlledRetriever(DemoRetriever(), principal=principal, policy=AccessPolicy())
print([doc.doc_id for doc in secure_retriever.retrieve("préavis", top_k=5)])

mt = MultiTenantRAGEngine()
mt.add_collection("tenant-a", "contracts", DemoRetriever(), default=True)
print([doc.doc_id for doc in mt.retrieve("préavis", tenant_id="tenant-a", principal=principal)])

feedback = FeedbackLoop()
feedback.log_feedback("préavis ?", "bad", correct_answer="30 jours", correct_source_ids=["a-contract"])
print(feedback.summary().to_dict())
