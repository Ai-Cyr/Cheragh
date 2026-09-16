"""FAISS vector store adapter."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import json
import os
import re
import tempfile
import threading
from typing import Any, Iterable, Optional

import numpy as np

from ..base import BaseRetriever, Document, EmbeddingModel, _snapshot_documents, _validate_top_k
from ..filters import metadata_matches
from ._validation import embedding_matrix, query_vector
from .memory import (
    _document_from_dict, _file_descriptor, _file_sha256, _fsync_directory,
    _manifest_integer, _stage_documents, _stage_text, _store_file_lock,
    _validate_embedding_model,
)


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
        self.index: Any = None
        self.dimension: int | None = None
        self._data_lock = threading.RLock()

    def add_documents(self, documents: Iterable[Document]) -> None:
        docs = _snapshot_documents(documents)
        if not docs:
            return
        with self._data_lock:
            vectors = embedding_matrix(
                self.embedding_model.embed_documents([doc.content for doc in docs]), rows=len(docs),
            )
            vectors = self._normalize(vectors) if self.normalize else vectors
            faiss = require_faiss()
            dimension = int(vectors.shape[1]) if self.index is None else self.dimension
            if vectors.shape[1] != dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {dimension}, got {vectors.shape[1]}")
            # Allocate the Python snapshot before the native index changes.
            # Publishing the references after add() then cannot fail midway
            # because list.extend needs another memory allocation.
            updated_documents = [*self.documents, *docs]
            index = faiss.IndexFlatIP(dimension) if self.index is None else self.index
            index.add(vectors)
            self.index = index
            self.dimension = dimension
            self.documents = updated_documents

    def similarity_search(self, query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[Document]:
        top_k = _validate_top_k(top_k)
        # FAISS permits concurrent searches, but not search/add on the same
        # index. Keep the native index and its document rows in one snapshot.
        with self._data_lock:
            return self._search_locked(query, top_k, filters)

    def _search_locked(self, query: str, top_k: int, filters: Optional[dict]) -> list[Document]:
        if not self.documents or self.index is None:
            return []
        candidate_indices = self._matching_indices(filters)
        if not candidate_indices:
            return []
        q = query_vector(self.embedding_model.embed_query(query), dimension=self.dimension)[np.newaxis, :]
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
        """Publish an immutable, checksummed generation through its manifest.

        Older generations are retained so an interrupted save cannot destroy
        the last readable snapshot. Schema-1 stores remain readable by load().
        """
        faiss = require_faiss()
        with self._data_lock:
            if self.index is None:
                raise ValueError("Cannot save an empty FAISS index")
            documents = _snapshot_documents(self.documents)
            index = faiss.clone_index(self.index)
            dimension = self.dimension
            normalize = self.normalize
        if index.ntotal != len(documents) or index.d != dimension:
            raise ValueError("FAISS store is corrupted: index and document snapshot differ")
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with _store_file_lock(p):
            staged: list[Path] = []
            try:
                document_stage = _stage_documents(p, documents)
                staged.append(document_stage)
                fd, index_name = tempfile.mkstemp(prefix=".index.", suffix=".tmp", dir=p)
                os.close(fd)
                index_stage = Path(index_name)
                staged.append(index_stage)
                faiss.write_index(index, str(index_stage))
                with index_stage.open("rb") as file:
                    os.fsync(file.fileno())
                document_name = f"documents.{_file_sha256(document_stage)}.jsonl"
                index_name = f"index.{_file_sha256(index_stage)}.faiss"
                manifest = {
                    "schema_version": 2, "count": len(documents), "dimension": dimension,
                    "normalize": normalize, "embedding_model": self.embedding_model.get_fingerprint(),
                    "documents": _file_descriptor(document_stage, document_name),
                    "index": _file_descriptor(index_stage, index_name),
                }
                manifest_stage = _stage_text(p, "manifest", json.dumps(manifest, allow_nan=False, indent=2))
                staged.append(manifest_stage)
                # These names describe their bytes, so publishing them cannot
                # replace another generation. The manifest is the sole commit.
                os.replace(document_stage, p / document_name)
                os.replace(index_stage, p / index_name)
                _fsync_directory(p)
                os.replace(manifest_stage, p / "manifest.json")
                _fsync_directory(p)
            finally:
                for temporary in staged:
                    temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path, embedding_model: EmbeddingModel) -> "FaissVectorStore":
        faiss = require_faiss()
        p = Path(path)
        with _store_file_lock(p, exclusive=False):
            manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("Invalid FAISS manifest: expected an object")
            version = _manifest_integer(manifest.get("schema_version", 1), "schema_version", minimum=1)
            if version not in {1, 2}:
                raise ValueError(f"Unsupported FAISS manifest schema_version: {version}")
            count = _manifest_integer(manifest.get("count"), "count", minimum=0)
            dimension = _manifest_integer(manifest.get("dimension"), "dimension", minimum=1)
            normalize = manifest.get("normalize", True)
            if not isinstance(normalize, bool):
                raise ValueError("Invalid FAISS manifest: normalize must be a boolean")
            _validate_embedding_model(manifest, embedding_model)
            if version == 2:
                documents_path = _generation_path(p, manifest, "documents", "jsonl")
                index_path = _generation_path(p, manifest, "index", "faiss")
            else:
                documents_path, index_path = p / "documents.jsonl", p / "index.faiss"
            store = cls(embedding_model=embedding_model, normalize=normalize)
            with documents_path.open("r", encoding="utf-8") as file:
                store.documents = [_document_from_dict(json.loads(line)) for line in file if line.strip()]
            if len(store.documents) != count:
                raise ValueError("FAISS store is corrupted: manifest count != document count")
            store.index = faiss.read_index(str(index_path))
            store.dimension = dimension
            if store.index.d != dimension:
                raise ValueError("FAISS store is corrupted: manifest dimension != index dimension")
            if store.index.ntotal != count:
                raise ValueError("FAISS store is corrupted: document count != index count")
            if not isinstance(store.index, faiss.IndexFlatIP):
                raise ValueError("FAISS store is corrupted: expected an IndexFlatIP index")
        return store

    def _matching_indices(self, filters: Optional[dict]) -> list[int]:
        if not filters:
            return list(range(len(self.documents)))
        return [idx for idx, doc in enumerate(self.documents) if metadata_matches(doc.metadata, filters)]

    def _result(self, idx: int, score: float) -> Document:
        doc = self.documents[idx]
        return Document(content=doc.content, metadata=deepcopy(doc.metadata), doc_id=doc.doc_id, score=score)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        # Accumulate in float64: finite float32 vectors can overflow a float32
        # sum of squares and otherwise collapse to all-zero embeddings.
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)


def _generation_path(directory: Path, manifest: dict, field: str, suffix: str) -> Path:
    descriptor = manifest.get(field)
    if not isinstance(descriptor, dict):
        raise ValueError(f"Invalid FAISS manifest: missing {field} descriptor")
    digest = descriptor.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"Invalid FAISS manifest: {field} checksum")
    expected_name = f"{field}.{digest}.{suffix}"
    if descriptor.get("filename") != expected_name:
        raise ValueError(f"Invalid FAISS manifest: {field} filename")
    size = _manifest_integer(descriptor.get("size_bytes"), f"{field}.size_bytes", minimum=0)
    path = directory / expected_name
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Invalid FAISS manifest: {field} path escapes store")
    if path.stat().st_size != size or _file_sha256(path) != digest:
        raise ValueError(f"FAISS store is corrupted: {field} checksum mismatch")
    return path


@dataclass
class FaissRetriever(BaseRetriever):
    store: FaissVectorStore
    filters: Optional[dict] = None

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.store.similarity_search(query, top_k=top_k, filters=self.filters)
