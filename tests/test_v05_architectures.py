import unittest

from cheragh import (
    CorrectiveRAGEngine,
    ConversationalRAGEngine,
    Document,
    GenerateNode,
    HashingEmbedding,
    ParentChildRetriever,
    RAGEngine,
    RAGWorkflow,
    RetrieveNode,
    StaticLLMClient,
)
from cheragh.ingestion import HierarchicalChunker


class TestV05Architectures(unittest.TestCase):
    def test_parent_child_retriever_returns_parent_context(self):
        parent = Document(
            "Section contrat. La clause de résiliation prévoit un préavis de 30 jours. Autre contexte utile.",
            doc_id="contract-parent",
        )
        child = Document(
            "La clause de résiliation prévoit un préavis de 30 jours.",
            metadata={"parent_doc_id": "contract-parent"},
            doc_id="contract-parent#child-0",
        )
        retriever = ParentChildRetriever(
            parent_documents=[parent],
            child_documents=[child],
            embedding_model=HashingEmbedding(64),
            top_k_children=3,
            top_k_parents=2,
        )
        results = retriever.retrieve("préavis résiliation", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].doc_id, "contract-parent")
        self.assertIn("child_matches", results[0].metadata)
        self.assertIn("Autre contexte", results[0].content)

    def test_parent_child_from_hierarchical_chunks(self):
        chunks = HierarchicalChunker(chunk_size=60, min_chunk_size=5).split_documents([
            Document("# A\nLa politique de remboursement est de 14 jours.", doc_id="doc")
        ])
        retriever = ParentChildRetriever.from_hierarchical_chunks(chunks, embedding_model=HashingEmbedding(64))
        results = retriever.retrieve("remboursement", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("remboursement", results[0].content.lower())

    def test_corrective_rag_engine_falls_back_on_low_context(self):
        docs = [Document("Le RAG combine retrieval et génération.", doc_id="rag")]
        base = RAGEngine.from_documents(
            docs,
            embedding_model=HashingEmbedding(64),
            retriever_type="memory",
            llm_client=StaticLLMClient("Réponse [source: rag]"),
            require_citations=True,
        )
        corrective = CorrectiveRAGEngine(base_engine=base, min_context_score=0.9, max_retries=0)
        response = corrective.ask("météo demain tokyo")
        self.assertIn("pas suffisamment fiable", response.answer)
        self.assertIn("corrective_low_context", response.warnings)

    def test_corrective_rag_engine_passes_when_context_good(self):
        docs = [Document("Le RAG combine retrieval et génération.", doc_id="rag")]
        base = RAGEngine.from_documents(
            docs,
            embedding_model=HashingEmbedding(64),
            retriever_type="memory",
            llm_client=StaticLLMClient("Le RAG combine retrieval et génération. [source: rag]"),
            require_citations=True,
        )
        corrective = CorrectiveRAGEngine(base_engine=base, min_context_score=0.01, max_retries=1)
        response = corrective.ask("Comment fonctionne le RAG ?")
        self.assertIn("rag", response.answer.lower())
        self.assertTrue(response.metadata["corrective"])
        self.assertIn("retrieval_grade", response.metadata)

    def test_conversational_rag_engine_stores_history(self):
        docs = [Document("Le contrat A contient une clause de préavis de 30 jours.", doc_id="c1")]
        engine = RAGEngine.from_documents(
            docs,
            embedding_model=HashingEmbedding(64),
            retriever_type="memory",
            llm_client=StaticLLMClient("Le préavis est de 30 jours. [source: c1]"),
            require_citations=True,
        )
        chat = ConversationalRAGEngine(engine, max_history_turns=2)
        first = chat.ask("Résume le contrat A", session_id="s")
        second = chat.ask("Et le préavis ?", session_id="s")
        self.assertEqual(len(chat.history("s")), 2)
        self.assertEqual(second.metadata["conversation"]["history_turns_used"], 1)
        self.assertIn("standalone_query", second.metadata["conversation"])
        self.assertIn("30 jours", first.answer)

    def test_rag_workflow_minimal_retrieve_generate(self):
        docs = [Document("Le RAG récupère des documents avant de générer.", doc_id="d1")]
        engine = RAGEngine.from_documents(docs, embedding_model=HashingEmbedding(64), retriever_type="memory")
        workflow = RAGWorkflow()
        workflow.add_node("retrieve", RetrieveNode(engine.retriever, top_k=1))
        workflow.add_node("generate", GenerateNode(StaticLLMClient("OK [source: d1]")))
        workflow.connect("retrieve", "generate")
        result = workflow.ask("Comment fonctionne le RAG ?")
        self.assertEqual(result.answer, "OK [source: d1]")
        self.assertEqual(result.executed_nodes, ["retrieve", "generate"])
        self.assertIn("documents", result.state)


if __name__ == "__main__":
    unittest.main()
