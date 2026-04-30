"""Vector store adapters.

The public names in this module are loaded lazily so importing
``cheragh`` or ``cheragh.vectorstores`` does not import optional
vector-store dependencies until the matching adapter is actually used.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_LAZY_EXPORTS = {
    "MemoryVectorStore": (".memory", "MemoryVectorStore"),
    "VectorStoreRetriever": (".memory", "VectorStoreRetriever"),
    "FaissVectorStore": (".faiss", "FaissVectorStore"),
    "FaissRetriever": (".faiss", "FaissRetriever"),
    "require_faiss": (".faiss", "require_faiss"),
    "ChromaVectorStore": (".chroma", "ChromaVectorStore"),
    "ChromaRetriever": (".chroma", "ChromaRetriever"),
    "require_chromadb": (".chroma", "require_chromadb"),
    "QdrantVectorStore": (".qdrant", "QdrantVectorStore"),
    "QdrantRetriever": (".qdrant", "QdrantRetriever"),
    "require_qdrant_client": (".qdrant", "require_qdrant_client"),
}

__all__ = list(_LAZY_EXPORTS)

if TYPE_CHECKING:  # pragma: no cover - imported only by static type checkers
    from .chroma import ChromaRetriever, ChromaVectorStore, require_chromadb
    from .faiss import FaissRetriever, FaissVectorStore, require_faiss
    from .memory import MemoryVectorStore, VectorStoreRetriever
    from .qdrant import QdrantRetriever, QdrantVectorStore, require_qdrant_client


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
