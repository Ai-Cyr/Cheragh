"""Chroma vector store adapter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..base import BaseRetriever, Document, EmbeddingModel


def require_chromadb():
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Chroma support requires chromadb. Install with: pip install cheragh[chroma]") from exc
    return chromadb


class ChromaVectorStore:
    """Vector store backed by ChromaDB.

    Embeddings are computed by the provided ``EmbeddingModel`` and passed to
    Chroma explicitly, avoiding Chroma-specific embedding functions.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        collection_name: str = "cheragh",
        path: str | Path | None = None,
        client=None,
    ):
        chromadb = require_chromadb() if client is None else None
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        if client is None:
            client = chromadb.PersistentClient(path=str(path)) if path else chromadb.Client()
        self.client = client
        self.collection = client.get_or_create_collection(collection_name)

    def add_documents(self, documents: Iterable[Document]) -> None:
        docs = list(documents)
        if not docs:
            return
        ids = [doc.doc_id or f"doc-{i}" for i, doc in enumerate(docs)]
        embeddings = self.embedding_model.embed_documents([doc.content for doc in docs]).tolist()
        metadatas = [_safe_metadata(doc.metadata) for doc in docs]
        self.collection.upsert(ids=ids, documents=[doc.content for doc in docs], metadatas=metadatas, embeddings=embeddings)

    def similarity_search(self, query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[Document]:
        embedding = self.embedding_model.embed_query(query).tolist()
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        output: list[Document] = []
        for doc_id, content, metadata, distance in zip(ids, docs, metadatas, distances):
            score = 1.0 / (1.0 + float(distance)) if distance is not None else None
            output.append(Document(content=content or "", metadata=dict(metadata or {}), doc_id=doc_id, score=score))
        return output

    def as_retriever(self, filters: Optional[dict] = None) -> "ChromaRetriever":
        return ChromaRetriever(self, filters=filters)


@dataclass
class ChromaRetriever(BaseRetriever):
    store: ChromaVectorStore
    filters: Optional[dict] = None

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.store.similarity_search(query, top_k=top_k, filters=self.filters)


def _safe_metadata(metadata: dict) -> dict:
    safe = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe
