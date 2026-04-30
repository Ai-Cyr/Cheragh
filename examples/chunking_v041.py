"""Minimal v0.4.1 advanced chunking example."""
from cheragh import Document, HashingEmbedding
from cheragh.ingestion import CodeChunker, HierarchicalChunker, SemanticChunker, TableChunker


docs = [
    Document("Le chat dort. Le chien joue. Le RAG récupère des documents. Le retrieval améliore la réponse.", doc_id="semantic"),
    Document("# Intro\nTexte.\n\n## Détails\nTexte détaillé.", doc_id="markdown"),
    Document("| Produit | CA |\n| --- | ---: |\n| A | 10 |\n| B | 20 |", doc_id="table"),
    Document("def load():\n    return 1\n\nclass Engine:\n    pass", metadata={"source": "app.py"}, doc_id="code"),
]

if __name__ == "__main__":
    print("Semantic:", SemanticChunker(HashingEmbedding(64), min_chunk_size=10).split_documents(docs[:1]))
    print("Hierarchical:", HierarchicalChunker(min_chunk_size=5).split_documents(docs[1:2]))
    print("Table:", TableChunker(rows_per_chunk=1).split_documents(docs[2:3]))
    print("Code:", CodeChunker().split_documents(docs[3:4]))
