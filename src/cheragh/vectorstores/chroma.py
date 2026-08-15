"""Chroma vector store adapter."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
from uuid import uuid4

from ..base import BaseRetriever, Document, EmbeddingModel, _validate_top_k
from ..filters import metadata_matches


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
        # Explicit document IDs retain Chroma's upsert semantics.  Anonymous
        # documents need globally unique IDs: restarting ``enumerate`` for every
        # call used to overwrite ``doc-0``, ``doc-1``, ... from earlier batches.
        ids = [str(doc.doc_id) if doc.doc_id else f"auto-{uuid4().hex}" for doc in docs]
        embeddings = self.embedding_model.embed_documents([doc.content for doc in docs]).tolist()
        metadatas = [_safe_metadata(doc.metadata) for doc in docs]
        self.collection.upsert(ids=ids, documents=[doc.content for doc in docs], metadatas=metadatas, embeddings=embeddings)

    def similarity_search(self, query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[Document]:
        """Return the exact canonical filtered top-k.

        Supported necessary conditions are pushed to Chroma to reduce data
        transfer (and strengthen tenant isolation). Operators without faithful
        Chroma semantics, such as ``$exists`` and ``$contains``, are evaluated
        locally and can therefore require scanning the whole collection.
        """

        top_k = _validate_top_k(top_k)
        embedding = self.embedding_model.embed_query(query).tolist()
        if not filters:
            return self._query(embedding, top_k)

        # Chroma's native ``where`` language is not identical to Cheragh's
        # shared Mongo-style metadata contract (notably ``$exists`` and
        # ``$contains``). Query progressively larger ranked prefixes and apply
        # the canonical predicate locally. Once a prefix contains ``top_k``
        # matches, no later result can outrank them; the collection count keeps
        # the worst-case request finite.
        total = int(self.collection.count())
        if total <= 0:
            return []
        native_filter = _to_chroma_filter(filters)
        for limit in _candidate_limits(top_k, total):
            candidates = self._query(embedding, limit, where=native_filter)
            matches = [
                document
                for document in candidates
                if metadata_matches(document.metadata, filters)
            ]
            if len(matches) >= top_k or len(candidates) < limit or limit >= total:
                return matches[:top_k]
        return []  # pragma: no cover - the final limit always reaches ``total``

    def _query(
        self,
        embedding: list[float],
        n_results: int,
        *,
        where: Optional[dict] = None,
    ) -> list[Document]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        output: list[Document] = []
        for doc_id, content, metadata, distance in zip(ids, docs, metadatas, distances):
            score = 1.0 / (1.0 + float(distance)) if distance is not None else None
            output.append(
                Document(
                    content=content or "",
                    metadata=_restore_metadata(dict(metadata or {})),
                    doc_id=doc_id,
                    score=score,
                )
            )
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
        if isinstance(value, str):
            safe[key] = _escape_metadata_string(value)
        elif isinstance(value, (int, float, bool)):
            safe[key] = value
        else:
            safe[key] = _encode_metadata_value(value)
    return safe


_JSON_METADATA_PREFIX = "\x1echeragh-json-v1:"
_STRING_METADATA_PREFIX = "\x1echeragh-string-v1:"


def _escape_metadata_string(value: str) -> str:
    if value.startswith((_JSON_METADATA_PREFIX, _STRING_METADATA_PREFIX)):
        return f"{_STRING_METADATA_PREFIX}{value}"
    return value


def _encode_metadata_value(value: object) -> str:
    def fallback(item: object):
        if isinstance(item, (set, frozenset)):
            return list(item)
        return str(item)

    try:
        encoded = json.dumps(value, default=fallback, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        encoded = json.dumps(str(value), ensure_ascii=False)
    return f"{_JSON_METADATA_PREFIX}{encoded}"


def _restore_metadata(metadata: dict) -> dict:
    restored = {}
    for key, value in metadata.items():
        if isinstance(value, str) and value.startswith(_STRING_METADATA_PREFIX):
            restored[key] = value[len(_STRING_METADATA_PREFIX) :]
        elif isinstance(value, str) and value.startswith(_JSON_METADATA_PREFIX):
            try:
                restored[key] = json.loads(value[len(_JSON_METADATA_PREFIX) :])
            except json.JSONDecodeError:
                # Preserve pre-existing user data that only happens to share
                # the marker prefix rather than corrupting it on read.
                restored[key] = value
        else:
            restored[key] = value
    return restored


def _candidate_limits(top_k: int, total: int) -> Iterator[int]:
    """Yield increasingly large, collection-bounded ranked prefixes."""

    limit = min(total, top_k * 4)
    while True:
        yield limit
        if limit >= total:
            return
        limit = min(total, limit * 2)


def _to_chroma_filter(filters: Mapping[str, Any]) -> Optional[dict]:
    """Compile only conditions that are necessary under canonical semantics."""

    conditions: list[dict] = []
    for key, expected in filters.items():
        if isinstance(expected, Mapping):
            for operator, value in expected.items():
                condition = _chroma_operator_condition(key, operator, value)
                if condition is not None:
                    conditions.append(condition)
        elif isinstance(expected, (list, tuple, set, frozenset)):
            values = _chroma_membership_values(expected)
            if values is not None:
                conditions.append({key: {"$in": values}})
        else:
            conditions.append({key: {"$eq": _chroma_equality_value(expected)}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _chroma_operator_condition(key: str, operator: str, value: Any) -> Optional[dict]:
    if operator == "$eq":
        return {key: {operator: _chroma_equality_value(value)}}
    if operator == "$in":
        values = _chroma_membership_values(value if isinstance(value, (list, tuple, set, frozenset)) else [value])
        return {key: {operator: values}} if values is not None else None
    if operator in {"$gt", "$gte", "$lt", "$lte"} and _is_number(value):
        return {key: {operator: value}}
    return None


def _chroma_equality_value(value: Any) -> str | int | float | bool:
    if isinstance(value, str):
        return _escape_metadata_string(value)
    if isinstance(value, (int, float, bool)):
        return value
    return _encode_metadata_value(value)


def _chroma_membership_values(values: Iterable[Any]) -> Optional[list[str | int | float | bool]]:
    encoded = [_chroma_equality_value(value) for value in values]
    if not encoded or len({type(value) for value in encoded}) != 1:
        return None
    return encoded


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
