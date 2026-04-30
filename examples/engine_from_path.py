"""Build a RAG engine from a folder without external services."""
from pathlib import Path
import tempfile

from cheragh import RAGEngine, HashingEmbedding

with tempfile.TemporaryDirectory() as tmp:
    docs = Path(tmp) / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text("Le RAG combine retrieval et génération avec des sources.", encoding="utf-8")
    (docs / "python.md").write_text("Un package Python moderne utilise pyproject.toml.", encoding="utf-8")

    engine = RAGEngine.from_path(docs, embedding_model=HashingEmbedding(dimension=128), top_k=2)
    response = engine.ask("Qu'est-ce que le RAG ?")
    print(response.answer)
    print(response.to_dict()["sources"])
