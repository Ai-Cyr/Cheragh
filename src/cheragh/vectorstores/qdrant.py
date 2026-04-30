"""Qdrant vector store adapter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..base import BaseRetriever, Document, EmbeddingModel


def require_qdrant_client():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Qdrant support requires qdrant-client. Install with: pip install cheragh[qdrant]") from exc
    return QdrantClient, models


class QdrantVectorStore:
    """Vector store backed by Qdrant.

    Supports local file-backed Qdrant via ``path`` or remote Qdrant via ``url``.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        collection_name: str = "cheragh",
        path: str | Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
        client=None,
        distance: str = "Cosine",
    ):
        QdrantClient, models = require_qdrant_client() if client is None else (None, None)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.distance = distance
        self.client = client or QdrantClient(path=str(path) if path else None, url=url, api_key=api_key)
        self._models = models

    def add_documents(self, documents: Iterable[Document]) -> None:
        QdrantClient, models = require_qdrant_client()
        docs = list(documents)
        if not docs:
            return
        vectors = np.asarray(self.embedding_model.embed_documents([doc.content for doc in docs]), dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        self._ensure_collection(vectors.shape[1], models)
        points = []
        for idx, (doc, vector) in enumerate(zip(docs, vectors)):
            point_id = _stable_qdrant_id(doc.doc_id or f"doc-{idx}")
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={"content": doc.content, "doc_id": doc.doc_id or str(point_id), **doc.metadata},
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def similarity_search(self, query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[Document]:
        _, models = require_qdrant_client()
        query_vec = self.embedding_model.embed_query(query).tolist()
        q_filter = _to_qdrant_filter(filters, models) if filters else None
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=top_k,
            query_filter=q_filter,
            with_payload=True,
        )
        output: list[Document] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            content = str(payload.pop("content", ""))
            doc_id = str(payload.pop("doc_id", hit.id))
            output.append(Document(content=content, metadata=payload, doc_id=doc_id, score=float(hit.score)))
        return output

    def as_retriever(self, filters: Optional[dict] = None) -> "QdrantRetriever":
        return QdrantRetriever(self, filters=filters)

    def _ensure_collection(self, size: int, models) -> None:
        existing = [collection.name for collection in self.client.get_collections().collections]
        if self.collection_name in existing:
            return
        distance = getattr(models.Distance, self.distance.upper(), models.Distance.COSINE)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=size, distance=distance),
        )


@dataclass
class QdrantRetriever(BaseRetriever):
    store: QdrantVectorStore
    filters: Optional[dict] = None

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.store.similarity_search(query, top_k=top_k, filters=self.filters)


def _to_qdrant_filter(filters: dict, models):
    must = []
    for key, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(value))))
        else:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=must)


def _stable_qdrant_id(doc_id: str) -> int:
    # Qdrant accepts integers or UUIDs. A deterministic positive int keeps this
    # adapter dependency-free and stable across runs.
    import hashlib

    digest = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)
