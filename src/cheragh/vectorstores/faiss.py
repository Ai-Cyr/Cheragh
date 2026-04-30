"""FAISS vector store adapter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable, Optional

import numpy as np

from ..base import BaseRetriever, Document, EmbeddingModel
from .memory import _document_from_dict, _document_to_dict


def require_faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("FAISS support requires faiss-cpu. Install with: pip install cheragh[faiss]") from exc
    return faiss


class FaissVectorStore:
    """Vector store backed by FAISS ``IndexFlatIP``.

    Embeddings are assumed to be normalized. If your embedding provider does not
    normalize, pass ``normalize=True`` so inner product behaves like cosine.
    """

    def __init__(self, embedding_model: EmbeddingModel, normalize: bool = True):
        self.embedding_model = embedding_model
        self.normalize = normalize
        self.documents: list[Document] = []
        self.index = None
        self.dimension: int | None = None

    def add_documents(self, documents: Iterable[Document]) -> None:
        docs = list(documents)
        if not docs:
            return
        vectors = np.asarray(self.embedding_model.embed_documents([doc.content for doc in docs]), dtype=np.float32)
        vectors = self._normalize(vectors) if self.normalize else vectors
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        faiss = require_faiss()
        if self.index is None:
            self.dimension = int(vectors.shape[1])
            self.index = faiss.IndexFlatIP(self.dimension)
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")
        self.index.add(vectors)
        self.documents.extend(docs)

    def similarity_search(self, query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[Document]:
        if not self.documents or self.index is None:
            return []
        candidate_indices = self._matching_indices(filters)
        q = np.asarray(self.embedding_model.embed_query(query), dtype=np.float32)[np.newaxis, :]
        q = self._normalize(q) if self.normalize else q
        # Fast path without metadata filtering.
        if len(candidate_indices) == len(self.documents):
            scores, indices = self.index.search(q, min(top_k, len(self.documents)))
            return [self._result(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if int(idx) >= 0]
        # Filtered path: reconstruct candidate vectors and score with NumPy.
        vectors = np.vstack([self.index.reconstruct(int(idx)) for idx in candidate_indices])
        scores = (vectors @ q.T).ravel()
        order = np.argsort(scores)[::-1][:top_k]
        return [self._result(candidate_indices[int(i)], float(scores[int(i)])) for i in order]

    def as_retriever(self, filters: Optional[dict] = None) -> "FaissRetriever":
        return FaissRetriever(self, filters=filters)

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise ValueError("Cannot save an empty FAISS index")
        faiss = require_faiss()
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        with (p / "documents.jsonl").open("w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(json.dumps(_document_to_dict(doc), ensure_ascii=False) + "\n")
        (p / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "count": len(self.documents),
                    "dimension": self.dimension,
                    "normalize": self.normalize,
                    "embedding_model": self.embedding_model.get_fingerprint(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, embedding_model: EmbeddingModel) -> "FaissVectorStore":
        faiss = require_faiss()
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
        store = cls(embedding_model=embedding_model, normalize=bool(manifest.get("normalize", True)))
        store.index = faiss.read_index(str(p / "index.faiss"))
        store.dimension = int(manifest["dimension"])
        with (p / "documents.jsonl").open("r", encoding="utf-8") as f:
            store.documents = [_document_from_dict(json.loads(line)) for line in f if line.strip()]
        if len(store.documents) != store.index.ntotal:
            raise ValueError("FAISS store is corrupted: document count != index count")
        return store

    def _matching_indices(self, filters: Optional[dict]) -> list[int]:
        if not filters:
            return list(range(len(self.documents)))
        matches: list[int] = []
        for idx, doc in enumerate(self.documents):
            ok = True
            for key, expected in filters.items():
                actual = doc.metadata.get(key)
                if isinstance(expected, (list, tuple, set)):
                    ok = actual in expected
                else:
                    ok = actual == expected
                if not ok:
                    break
            if ok:
                matches.append(idx)
        return matches

    def _result(self, idx: int, score: float) -> Document:
        doc = self.documents[idx]
        return Document(content=doc.content, metadata=dict(doc.metadata), doc_id=doc.doc_id, score=score)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


@dataclass
class FaissRetriever(BaseRetriever):
    store: FaissVectorStore
    filters: Optional[dict] = None

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.store.similarity_search(query, top_k=top_k, filters=self.filters)
