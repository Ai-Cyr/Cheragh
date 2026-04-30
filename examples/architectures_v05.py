"""Minimal examples for advanced RAG architectures v0.5.0."""
from cheragh import (
    ConversationalRAGEngine,
    CorrectiveRAGEngine,
    Document,
    GenerateNode,
    HashingEmbedding,
    ParentChildRetriever,
    RAGEngine,
    RAGWorkflow,
    RetrieveNode,
    StaticLLMClient,
)


docs = [Document("Le contrat A prévoit un préavis de 30 jours.", doc_id="contract-a")]
embedder = HashingEmbedding(dimension=128)
llm = StaticLLMClient("Le préavis est de 30 jours. [source: contract-a]")

engine = RAGEngine.from_documents(docs, embedding_model=embedder, retriever_type="memory", llm_client=llm)

# Corrective RAG
corrective = CorrectiveRAGEngine(base_engine=engine)
print(corrective.ask("Quel est le préavis ?").answer)

# Conversational RAG
chat = ConversationalRAGEngine(engine)
chat.ask("Résume le contrat A", session_id="demo")
print(chat.ask("Et le préavis ?", session_id="demo").metadata["conversation"])

# Parent-child retrieval
children = [Document("préavis de 30 jours", metadata={"parent_doc_id": "contract-a"}, doc_id="contract-a#child-0")]
parent_child = ParentChildRetriever(docs, child_documents=children, embedding_model=embedder)
print(parent_child.retrieve("préavis", top_k=1)[0].metadata["matched_child_count"])

# Workflow minimal
workflow = RAGWorkflow()
workflow.add_node("retrieve", RetrieveNode(engine.retriever, top_k=1))
workflow.add_node("generate", GenerateNode(llm))
workflow.connect("retrieve", "generate")
print(workflow.ask("Quel est le préavis ?").answer)
