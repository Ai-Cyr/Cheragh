"""Dependency-free vector store backed by NumPy arrays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import tempfile
from typing import TYPE_CHECKING, Any, Iterable, Optional

from ..base import (
    BaseRetriever,
    Document,
    EmbeddingModel,
    HashingEmbedding,
    _numpy,
    _snapshot_documents,
    _validate_top_k,
    cosine_similarity,
)
from ..filters import metadata_matches

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


_STORE_MANIFEST_SCHEMA_VERSION = 2
_HASHING_FINGERPRINT = re.compile(
    r"^HashingEmbedding::(?P<dimension>[1-9][0-9]*)::\(\s*(?P<min_n>[1-9][0-9]*)\s*,\s*(?P<max_n>[1-9][0-9]*)\s*\)$"
)


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
        new_docs = _snapshot_documents(documents)
        if not new_docs:
            return
        new_embeddings = np.asarray(self.embedding_model.embed_documents([doc.content for doc in new_docs]))
        if new_embeddings.ndim != 2 or new_embeddings.shape[0] != len(new_docs):
            raise ValueError(
                "Embedding model returned an invalid matrix: expected "
                f"({len(new_docs)}, dimension), got {new_embeddings.shape}"
            )
        expected_dimension = _known_embedding_dimension(self.embedding_model)
        if expected_dimension is not None and new_embeddings.shape[1] != expected_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"model declares {expected_dimension}, returned {new_embeddings.shape[1]}"
            )
        if self.embeddings is not None and len(self.embeddings) and self.embeddings.shape[1] != new_embeddings.shape[1]:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"store contains {self.embeddings.shape[1]}, got {new_embeddings.shape[1]}"
            )
        self.documents.extend(new_docs)
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def similarity_search(self, query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[Document]:
        top_k = _validate_top_k(top_k)
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

        The embedding model itself is not serialized. The built-in hashing
        model can be reconstructed safely from its manifest; other providers
        must be passed to :meth:`load`. Every file is fully staged in the
        destination directory before atomic replacement; ``manifest.json`` is
        committed last so readers never observe a partially written file.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np = _numpy()
        known_dimension = _known_embedding_dimension(self.embedding_model)
        embeddings = self.embeddings
        if embeddings is None:
            embeddings = np.zeros((0, known_dimension or 0))
        if embeddings.ndim != 2:
            raise ValueError(f"Cannot save vector store: embeddings must be 2D, got shape {embeddings.shape}")
        if embeddings.shape[0] != len(self.documents):
            raise ValueError("Cannot save vector store: document count != embedding count")
        if known_dimension is not None and embeddings.shape[1] not in {0, known_dimension}:
            raise ValueError(
                "Cannot save vector store: embedding dimension mismatch "
                f"(model={known_dimension}, vectors={embeddings.shape[1]})"
            )
        dimension = int(embeddings.shape[1] or known_dimension or 0)
        manifest = {
            "schema_version": _STORE_MANIFEST_SCHEMA_VERSION,
            "count": len(self.documents),
            "dimension": dimension,
            "embedding_model": self.embedding_model.get_fingerprint(),
        }
        embedding_descriptor = _embedding_descriptor(self.embedding_model)
        if embedding_descriptor is not None:
            manifest["embedding"] = embedding_descriptor
        staged: list[tuple[Path, Path]] = []
        try:
            staged.append(
                (
                    _stage_documents(p, self.documents),
                    p / "documents.jsonl",
                )
            )
            staged.append((_stage_embeddings(p, embeddings), p / "embeddings.npy"))
            staged.append(
                (
                    _stage_text(
                        p,
                        "manifest",
                        json.dumps(manifest, ensure_ascii=False, indent=2),
                    ),
                    p / "manifest.json",
                )
            )
            # ``staged`` is ordered documents, embeddings, manifest; commit the
            # manifest last so readers can use it as the snapshot marker.
            for temporary_path, destination in staged:
                os.replace(temporary_path, destination)
            _fsync_directory(p)
        finally:
            for temporary_path, _ in staged:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedding_model: EmbeddingModel | None = None,
    ) -> "MemoryVectorStore":
        """Load and validate a persisted store.

        ``embedding_model`` remains accepted for every provider and must match
        the fingerprint stored in ``manifest.json``. When it is omitted, only
        a manifest explicitly describing the built-in :class:`HashingEmbedding`
        is reconstructed automatically; external/custom providers must always
        be supplied by the caller.
        """
        p = Path(path)
        manifest = _read_store_manifest(p)
        if embedding_model is None:
            embedding_model = _hashing_embedding_from_manifest(manifest)
        _validate_embedding_model(manifest, embedding_model)
        store = cls(embedding_model=embedding_model)
        docs_path = p / "documents.jsonl"
        embeddings_path = p / "embeddings.npy"
        if not docs_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(f"Missing vector store files in {p}")
        with docs_path.open("r", encoding="utf-8") as f:
            store.documents = [_document_from_dict(json.loads(line)) for line in f if line.strip()]
        np = _numpy()
        store.embeddings = np.load(embeddings_path, allow_pickle=False)
        if store.embeddings.ndim != 2:
            raise ValueError(
                "Vector store is corrupted: embeddings must be a 2D matrix, "
                f"got shape {store.embeddings.shape}"
            )
        if len(store.documents) != len(store.embeddings):
            raise ValueError("Vector store is corrupted: document count != embedding count")
        manifest_count = int(manifest["count"])
        if manifest_count != len(store.documents):
            raise ValueError(
                "Vector store is corrupted: manifest count "
                f"{manifest_count} != document count {len(store.documents)}"
            )
        manifest_dimension = manifest.get("dimension")
        stored_dimension = int(store.embeddings.shape[1])
        if manifest_dimension is not None and stored_dimension and int(manifest_dimension) != stored_dimension:
            raise ValueError(
                "Vector store is corrupted: manifest embedding dimension "
                f"{manifest_dimension} != stored dimension {stored_dimension}"
            )
        model_dimension = _known_embedding_dimension(embedding_model)
        if model_dimension is not None and stored_dimension and model_dimension != stored_dimension:
            raise ValueError(
                "Embedding dimension mismatch for vector store: "
                f"index uses {stored_dimension}, provided model uses {model_dimension}"
            )
        return store

    @classmethod
    def embedding_model_from_manifest(cls, path: str | Path) -> EmbeddingModel:
        """Reconstruct the safe built-in embedder declared by an index.

        Currently this intentionally supports only ``HashingEmbedding``. This
        method never imports or instantiates arbitrary provider classes named by
        untrusted manifest data.
        """

        return _hashing_embedding_from_manifest(_read_store_manifest(Path(path)))

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


def _read_store_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing vector store manifest: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid vector store manifest: {manifest_path}") from exc
    if not isinstance(data, dict):
        raise ValueError("Invalid vector store manifest: expected a JSON object")

    schema_version = _manifest_integer(data.get("schema_version", 1), "schema_version", minimum=1)
    if schema_version not in {1, _STORE_MANIFEST_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported vector store manifest schema_version: {schema_version}")
    data["schema_version"] = schema_version
    data["count"] = _manifest_integer(data.get("count"), "count", minimum=0)

    fingerprint = data.get("embedding_model")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ValueError("Invalid vector store manifest: embedding_model must be a non-empty string")
    data["embedding_model"] = fingerprint.strip()

    if "dimension" in data:
        data["dimension"] = _manifest_integer(data["dimension"], "dimension", minimum=0)
    descriptor = data.get("embedding")
    if descriptor is not None and not isinstance(descriptor, dict):
        raise ValueError("Invalid vector store manifest: embedding must be an object")
    return data


def _manifest_integer(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Invalid vector store manifest: {field} must be an integer >= {minimum}")
    return value


def _hashing_embedding_from_manifest(manifest: dict[str, Any]) -> HashingEmbedding:
    descriptor = manifest.get("embedding")
    if int(manifest["schema_version"]) >= 2:
        if not isinstance(descriptor, dict) or descriptor.get("provider") != "hashing":
            raise ValueError(
                "Automatic vector store loading only supports an index whose manifest "
                "explicitly declares provider='hashing'; pass embedding_model for custom providers"
            )
        dimension = _manifest_integer(descriptor.get("dimension"), "embedding.dimension", minimum=1)
        ngram_range = descriptor.get("ngram_range")
        if not isinstance(ngram_range, list) or len(ngram_range) != 2:
            raise ValueError("Invalid vector store manifest: embedding.ngram_range must contain two integers")
        min_n = _manifest_integer(ngram_range[0], "embedding.ngram_range[0]", minimum=1)
        max_n = _manifest_integer(ngram_range[1], "embedding.ngram_range[1]", minimum=min_n)
        model = HashingEmbedding(dimension=dimension, ngram_range=(min_n, max_n))
    else:
        # Schema v1 stored only the stable fingerprint. Parse the exact built-in
        # format for backward compatibility, never an arbitrary import path.
        match = _HASHING_FINGERPRINT.fullmatch(_canonical_fingerprint(str(manifest["embedding_model"])))
        if match is None:
            raise ValueError(
                "Automatic vector store loading cannot reconstruct this legacy embedding model; "
                "pass embedding_model explicitly"
            )
        model = HashingEmbedding(
            dimension=int(match.group("dimension")),
            ngram_range=(int(match.group("min_n")), int(match.group("max_n"))),
        )
    _validate_embedding_model(manifest, model)
    return model


def _validate_embedding_model(manifest: dict[str, Any], embedding_model: EmbeddingModel) -> None:
    expected = str(manifest["embedding_model"])
    actual = embedding_model.get_fingerprint()
    if _canonical_fingerprint(expected) != _canonical_fingerprint(actual):
        raise ValueError(
            "Embedding model mismatch for vector store: "
            f"index expects {expected!r}, provided model is {actual!r}"
        )
    expected_dimension = manifest.get("dimension")
    actual_dimension = _known_embedding_dimension(embedding_model)
    if expected_dimension not in {None, 0} and actual_dimension is not None and int(expected_dimension) != actual_dimension:
        raise ValueError(
            "Embedding dimension mismatch for vector store: "
            f"index expects {expected_dimension}, provided model uses {actual_dimension}"
        )


def _canonical_fingerprint(fingerprint: str) -> str:
    # Caching does not change vectors, so cached and uncached forms are
    # persistence-compatible.
    while fingerprint.startswith("Cached::"):
        fingerprint = fingerprint[len("Cached::") :]
    return fingerprint


def _unwrap_embedding_model(embedding_model: EmbeddingModel) -> EmbeddingModel:
    try:
        from ..cache.decorators import CachedEmbeddingModel
    except ImportError:  # pragma: no cover - cache module is part of cheragh
        return embedding_model
    while isinstance(embedding_model, CachedEmbeddingModel):
        embedding_model = embedding_model.model
    return embedding_model


def _known_embedding_dimension(embedding_model: EmbeddingModel) -> int | None:
    model = _unwrap_embedding_model(embedding_model)
    dimension = getattr(model, "dimension", None)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        return None
    return dimension


def _embedding_descriptor(embedding_model: EmbeddingModel) -> dict[str, Any] | None:
    model = _unwrap_embedding_model(embedding_model)
    if not isinstance(model, HashingEmbedding):
        return None
    return {
        "provider": "hashing",
        "dimension": model.dimension,
        "ngram_range": list(model.ngram_range),
    }


def _document_to_dict(doc: Document) -> dict:
    return {"content": doc.content, "metadata": doc.metadata, "doc_id": doc.doc_id, "score": doc.score}


def _document_from_dict(data: dict) -> Document:
    return Document(
        content=data.get("content", ""),
        metadata=data.get("metadata") or {},
        doc_id=data.get("doc_id"),
        score=data.get("score"),
    )


def _stage_documents(directory: Path, documents: Iterable[Document]) -> Path:
    fd, temporary_name = tempfile.mkstemp(prefix=".documents.", suffix=".tmp", dir=directory)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            for document in documents:
                file.write(json.dumps(_document_to_dict(document), ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _stage_text(directory: Path, prefix: str, content: str) -> Path:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{prefix}.", suffix=".tmp", dir=directory)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _stage_embeddings(directory: Path, embeddings: np.ndarray) -> Path:
    np = _numpy()
    fd, temporary_name = tempfile.mkstemp(prefix=".embeddings.", suffix=".tmp", dir=directory)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as file:
            np.save(file, embeddings, allow_pickle=False)
            file.flush()
            os.fsync(file.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError:  # pragma: no cover - platform-specific durability support
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
