"""Dependency-free vector store backed by NumPy arrays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import TYPE_CHECKING, Iterable, Optional

from ..base import BaseRetriever, Document, EmbeddingModel, _numpy, cosine_similarity
from ..filters import metadata_matches

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


class MemoryVectorStore:
    """Simple in-memory vector store with JSONL/NPY persistence.

    It is useful for tests, prototypes and small corpora. For large production
    corpora, use the same interface to build a Qdrant/Chroma/pgvector adapter.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.documents: list[Document] = []
        self.embeddings: np.ndarray | None = None

    def add_documents(self, documents: Iterable[Document]) -> None:
        np = _numpy()
        new_docs = list(documents)
        if not new_docs:
            return
        new_embeddings = self.embedding_model.embed_documents([doc.content for doc in new_docs])
        self.documents.extend(new_docs)
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def similarity_search(self, query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[Document]:
        np = _numpy()
        if not self.documents or self.embeddings is None:
            return []
        candidate_indices = self._matching_indices(filters)
        if not candidate_indices:
            return []
        query_vec = self.embedding_model.embed_query(query)
        matrix = self.embeddings[candidate_indices]
        scores = cosine_similarity(query_vec, matrix)
        order = np.argsort(scores)[::-1][:top_k]
        results: list[Document] = []
        for local_idx in order:
            idx = candidate_indices[int(local_idx)]
            doc = self.documents[idx]
            results.append(
                Document(
                    content=doc.content,
                    metadata=dict(doc.metadata),
                    doc_id=doc.doc_id,
                    score=float(scores[int(local_idx)]),
                )
            )
        return results

    def as_retriever(self, filters: Optional[dict] = None) -> "VectorStoreRetriever":
        return VectorStoreRetriever(self, filters=filters)

    def save(self, path: str | Path) -> None:
        """Persist documents and embeddings to ``path``.

        The embedding model itself is not serialized; pass the same compatible
        embedder to :meth:`load`.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        docs_path = p / "documents.jsonl"
        with docs_path.open("w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(json.dumps(_document_to_dict(doc), ensure_ascii=False) + "\n")
        np = _numpy()
        np.save(p / "embeddings.npy", self.embeddings if self.embeddings is not None else np.zeros((0, 0)))
        (p / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "count": len(self.documents),
                    "embedding_model": self.embedding_model.get_fingerprint(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, embedding_model: EmbeddingModel) -> "MemoryVectorStore":
        p = Path(path)
        store = cls(embedding_model=embedding_model)
        docs_path = p / "documents.jsonl"
        embeddings_path = p / "embeddings.npy"
        if not docs_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(f"Missing vector store files in {p}")
        with docs_path.open("r", encoding="utf-8") as f:
            store.documents = [_document_from_dict(json.loads(line)) for line in f if line.strip()]
        np = _numpy()
        store.embeddings = np.load(embeddings_path, allow_pickle=False)
        if len(store.documents) != len(store.embeddings):
            raise ValueError("Vector store is corrupted: document count != embedding count")
        return store

    def _matching_indices(self, filters: Optional[dict]) -> list[int]:
        if not filters:
            return list(range(len(self.documents)))
        matches: list[int] = []
        for idx, doc in enumerate(self.documents):
            if metadata_matches(doc.metadata, filters):
                matches.append(idx)
        return matches


@dataclass
class VectorStoreRetriever(BaseRetriever):
    """Retriever adapter around :class:`MemoryVectorStore`."""

    store: MemoryVectorStore
    filters: Optional[dict] = None

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.store.similarity_search(query, top_k=top_k, filters=self.filters)


def _document_to_dict(doc: Document) -> dict:
    return {"content": doc.content, "metadata": doc.metadata, "doc_id": doc.doc_id, "score": doc.score}


def _document_from_dict(data: dict) -> Document:
    return Document(
        content=data.get("content", ""),
        metadata=data.get("metadata") or {},
        doc_id=data.get("doc_id"),
        score=data.get("score"),
    )
