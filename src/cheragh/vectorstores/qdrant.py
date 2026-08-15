"""Qdrant vector store adapter."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
from uuid import uuid4

import numpy as np

from ..base import BaseRetriever, Document, EmbeddingModel, _validate_top_k
from ..filters import metadata_matches


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
        _, models = require_qdrant_client()
        self._models = models
        docs = list(documents)
        if not docs:
            return
        vectors = np.asarray(self.embedding_model.embed_documents([doc.content for doc in docs]), dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        self._ensure_collection(vectors.shape[1], models)
        points = []
        for doc, vector in zip(docs, vectors):
            if not doc.doc_id:
                # A fresh UUID prevents anonymous documents from successive
                # ``add_documents`` calls from reusing the old ``doc-0`` ID.
                effective_doc_id = str(uuid4())
                point_id = effective_doc_id
            else:
                effective_doc_id = str(doc.doc_id)
                point_id = _stable_qdrant_id(effective_doc_id)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    # Reserved fields are applied last so untrusted document
                    # metadata cannot spoof the stored content or identity.
                    payload={**doc.metadata, "content": doc.content, "doc_id": effective_doc_id},
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def similarity_search(self, query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[Document]:
        """Return the exact canonical filtered top-k.

        Necessary equality, membership and numeric range conditions
        are pushed down when Qdrant models are available. Unsupported operators
        remain exact through local post-filtering, at the cost of potentially
        transferring every candidate admitted by the native prefilter.
        """

        top_k = _validate_top_k(top_k)
        query_vec = self.embedding_model.embed_query(query).tolist()
        if not filters:
            return self._search(query_vec, top_k)

        # Qdrant and Cheragh do not assign identical semantics to every filter
        # operator and metadata type. In particular, ``$contains`` on strings
        # and sequences cannot be translated faithfully for all payloads.
        # Push down the conservative subset and apply the full predicate used
        # by Memory, Hybrid and FAISS afterward. Progressive over-fetch keeps
        # the common case small while an exact candidate count bounds the worst
        # case and preserves the true filtered top-k.
        native_filter = _to_qdrant_filter(filters, self._models) if self._models is not None else None
        count_kwargs = {"collection_name": self.collection_name, "exact": True}
        if native_filter is not None:
            count_kwargs["count_filter"] = native_filter
        count_result = self.client.count(**count_kwargs)
        total = int(getattr(count_result, "count", count_result))
        if total <= 0:
            return []
        for limit in _candidate_limits(top_k, total):
            matches = [
                document
                for document in self._search(query_vec, limit, query_filter=native_filter)
                if metadata_matches(document.metadata, filters)
            ]
            if len(matches) >= top_k or limit >= total:
                return matches[:top_k]
        return []  # pragma: no cover - the final limit always reaches ``total``

    def _search(self, query_vector: list[float], limit: int, *, query_filter=None) -> list[Document]:
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
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


def _stable_qdrant_id(doc_id: str) -> int:
    # Qdrant accepts integers or UUIDs. A deterministic positive int keeps this
    # adapter dependency-free and stable across runs.
    import hashlib

    digest = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)


def _candidate_limits(top_k: int, total: int) -> Iterator[int]:
    """Yield increasingly large, collection-bounded ranked prefixes."""

    limit = min(total, top_k * 4)
    while True:
        yield limit
        if limit >= total:
            return
        limit = min(total, limit * 2)


def _to_qdrant_filter(filters: Mapping[str, Any], models) -> Any:
    """Compile a safe necessary-condition subset of canonical filters."""

    must = []
    for key, expected in filters.items():
        if isinstance(expected, Mapping):
            range_values = {
                operator[1:]: value
                for operator, value in expected.items()
                if operator in {"$gt", "$gte", "$lt", "$lte"} and _is_number(value)
            }
            if range_values:
                must.append(models.FieldCondition(key=key, range=models.Range(**range_values)))
            for operator, value in expected.items():
                if operator == "$eq" and _is_qdrant_scalar(value):
                    must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
                elif operator == "$in":
                    values = _qdrant_membership_values(value)
                    if values is not None:
                        must.append(models.FieldCondition(key=key, match=models.MatchAny(any=values)))
                # Negative predicates are deliberately not pushed down. Qdrant
                # applies MatchValue/MatchAny element-wise to array payloads,
                # whereas Cheragh's canonical $ne/$nin compare the metadata
                # value itself. A native must_not would therefore remove valid
                # candidates before the exact local predicate can inspect them.
        elif isinstance(expected, (list, tuple, set, frozenset)):
            values = _qdrant_membership_values(expected)
            if values is not None:
                must.append(models.FieldCondition(key=key, match=models.MatchAny(any=values)))
        elif _is_qdrant_scalar(expected):
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=expected)))
    kwargs = {}
    if must:
        kwargs["must"] = must
    return models.Filter(**kwargs) if kwargs else None


def _qdrant_membership_values(value: Any) -> Optional[list[str | int | bool]]:
    values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    candidates = list(values)
    if not candidates or not all(_is_qdrant_scalar(candidate) for candidate in candidates):
        return None
    return candidates


def _is_qdrant_scalar(value: Any) -> bool:
    # Qdrant exact-match payload conditions support keyword, integer and bool;
    # floating-point equality remains a local canonical check.
    return isinstance(value, (str, int, bool))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
