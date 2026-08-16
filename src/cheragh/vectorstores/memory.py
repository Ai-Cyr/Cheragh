"""Dependency-free vector store backed by NumPy arrays."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional

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


_STORE_MANIFEST_SCHEMA_VERSION = 3
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
        self._data_lock = threading.RLock()
        # Provider clients are not uniformly thread-safe. Serialize access to a
        # model instance while allowing persistence and scoring to use stable
        # snapshots without holding this lock.
        self._embedding_lock = threading.RLock()

    def add_documents(self, documents: Iterable[Document]) -> None:
        np = _numpy()
        materialized = list(documents)
        for index, document in enumerate(materialized):
            _validate_document(document, label=f"documents[{index}]")
        new_docs = _snapshot_documents(materialized)
        if not new_docs:
            return
        with self._embedding_lock:
            new_embeddings = np.asarray(self.embedding_model.embed_documents([doc.content for doc in new_docs]))
        if new_embeddings.ndim != 2 or new_embeddings.shape[0] != len(new_docs):
            raise ValueError(
                "Embedding model returned an invalid matrix: expected "
                f"({len(new_docs)}, dimension), got {new_embeddings.shape}"
            )
        _validate_finite_embeddings(new_embeddings, label="embedding model output")
        new_embeddings = new_embeddings.astype(float, copy=True)
        _validate_finite_embeddings(new_embeddings, label="embedding model output")
        expected_dimension = _known_embedding_dimension(self.embedding_model)
        if expected_dimension is not None and new_embeddings.shape[1] != expected_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"model declares {expected_dimension}, returned {new_embeddings.shape[1]}"
            )
        with self._data_lock:
            current_embeddings = self.embeddings
            if (
                current_embeddings is not None
                and len(current_embeddings)
                and current_embeddings.shape[1] != new_embeddings.shape[1]
            ):
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"store contains {current_embeddings.shape[1]}, got {new_embeddings.shape[1]}"
                )
            # Publish documents and vectors together under the data lock. New
            # containers keep snapshots already held by readers immutable.
            self.documents = [*self.documents, *new_docs]
            if current_embeddings is None or len(current_embeddings) == 0:
                self.embeddings = new_embeddings.copy()
            else:
                self.embeddings = np.vstack([current_embeddings, new_embeddings])

    def similarity_search(self, query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[Document]:
        top_k = _validate_top_k(top_k)
        np = _numpy()
        with self._data_lock:
            documents = tuple(self.documents)
            embeddings = self.embeddings
        if not documents or embeddings is None:
            return []
        with self._embedding_lock:
            query_vec = np.asarray(self.embedding_model.embed_query(query))
        if query_vec.ndim != 1:
            raise ValueError(f"Embedding model returned an invalid query vector shape: {query_vec.shape}")
        _validate_finite_embeddings(query_vec, label="query embedding")
        query_vec = query_vec.astype(float, copy=True)
        _validate_finite_embeddings(query_vec, label="query embedding")
        if query_vec.shape[0] != embeddings.shape[1]:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"store contains {embeddings.shape[1]}, query has {query_vec.shape[0]}"
            )
        candidate_indices = self._matching_indices_in(documents, filters)
        if not candidate_indices:
            return []
        matrix = embeddings[candidate_indices]
        scores = cosine_similarity(query_vec, matrix)
        order = np.argsort(scores)[::-1][:top_k]
        results: list[Document] = []
        for local_idx in order:
            idx = candidate_indices[int(local_idx)]
            doc = documents[idx]
            results.append(
                Document(
                    content=doc.content,
                    metadata=deepcopy(doc.metadata),
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
        with self._data_lock:
            documents = _snapshot_documents(self.documents)
            embeddings = None if self.embeddings is None else self.embeddings.copy()
        if embeddings is None:
            embeddings = np.zeros((0, known_dimension or 0))
        if embeddings.ndim != 2:
            raise ValueError(f"Cannot save vector store: embeddings must be 2D, got shape {embeddings.shape}")
        if embeddings.shape[0] != len(documents):
            raise ValueError("Cannot save vector store: document count != embedding count")
        _validate_finite_embeddings(embeddings, label="stored embeddings")
        if known_dimension is not None and embeddings.shape[1] not in {0, known_dimension}:
            raise ValueError(
                "Cannot save vector store: embedding dimension mismatch "
                f"(model={known_dimension}, vectors={embeddings.shape[1]})"
            )
        dimension = int(embeddings.shape[1] or known_dimension or 0)
        manifest = {
            "schema_version": _STORE_MANIFEST_SCHEMA_VERSION,
            "count": len(documents),
            "dimension": dimension,
            "embedding_model": self.embedding_model.get_fingerprint(),
        }
        embedding_descriptor = _embedding_descriptor(self.embedding_model)
        if embedding_descriptor is not None:
            manifest["embedding"] = embedding_descriptor
        _persist_store_snapshot(p, documents, embeddings, manifest)

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
        if not p.is_dir():
            raise FileNotFoundError(f"Missing vector store directory: {p}")
        with _store_file_lock(p, exclusive=False):
            return _load_store_snapshot(cls, p, embedding_model)

    @classmethod
    def embedding_model_from_manifest(cls, path: str | Path) -> EmbeddingModel:
        """Reconstruct the safe built-in embedder declared by an index.

        Currently this intentionally supports only ``HashingEmbedding``. This
        method never imports or instantiates arbitrary provider classes named by
        untrusted manifest data.
        """

        return _hashing_embedding_from_manifest(_read_store_manifest(Path(path)))

    def _matching_indices(self, filters: Optional[dict]) -> list[int]:
        with self._data_lock:
            documents = tuple(self.documents)
        return self._matching_indices_in(documents, filters)

    @staticmethod
    def _matching_indices_in(documents: Iterable[Document], filters: Optional[dict]) -> list[int]:
        documents = tuple(documents)
        if not filters:
            return list(range(len(documents)))
        matches: list[int] = []
        for idx, doc in enumerate(documents):
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


def _load_store_snapshot(
    store_class: type[MemoryVectorStore],
    directory: Path,
    embedding_model: EmbeddingModel | None,
) -> MemoryVectorStore:
    manifest = _read_store_manifest(directory)
    if embedding_model is None:
        embedding_model = _hashing_embedding_from_manifest(manifest)
    _validate_embedding_model(manifest, embedding_model)
    store = store_class(embedding_model=embedding_model)
    docs_path = _snapshot_data_path(directory, manifest, "documents", "documents.jsonl")
    embeddings_path = _snapshot_data_path(directory, manifest, "embeddings", "embeddings.npy")
    if not docs_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(f"Missing vector store files in {directory}")
    # Authenticate the bounded snapshot before parsing JSON or allowing NumPy
    # to interpret the NPY header/data.
    _validate_snapshot_integrity(manifest, "documents", docs_path)
    _validate_snapshot_integrity(manifest, "embeddings", embeddings_path)
    with docs_path.open("r", encoding="utf-8") as file:
        documents: list[Document] = []
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                document = _document_from_dict(payload)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Vector store is corrupted: invalid document at line {line_number}"
                ) from exc
            documents.append(document)
        store.documents = documents
    np = _numpy()
    try:
        store.embeddings = np.load(embeddings_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("Vector store is corrupted: invalid embeddings.npy") from exc
    if store.embeddings.ndim != 2:
        raise ValueError(
            "Vector store is corrupted: embeddings must be a 2D matrix, "
            f"got shape {store.embeddings.shape}"
        )
    _validate_finite_embeddings(store.embeddings, label="persisted embeddings", corruption=True)
    store.embeddings = store.embeddings.astype(float, copy=True)
    _validate_finite_embeddings(store.embeddings, label="persisted embeddings", corruption=True)
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
    if manifest_count > 0 and stored_dimension == 0:
        raise ValueError("Vector store is corrupted: non-empty store has zero-dimensional embeddings")
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
    if schema_version not in {1, 2, _STORE_MANIFEST_SCHEMA_VERSION}:
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
    if schema_version >= 3:
        for field, expected_filename in (
            ("documents", "documents.jsonl"),
            ("embeddings", "embeddings.npy"),
        ):
            integrity = data.get(field)
            if not isinstance(integrity, dict):
                raise ValueError(f"Invalid vector store manifest: {field} must be an object")
            if integrity.get("filename") != expected_filename:
                raise ValueError(
                    f"Invalid vector store manifest: {field}.filename must be {expected_filename!r}"
                )
            digest = integrity.get("sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"Invalid vector store manifest: {field}.sha256 must be lowercase SHA-256")
            integrity["size_bytes"] = _manifest_integer(
                integrity.get("size_bytes"),
                f"{field}.size_bytes",
                minimum=0,
            )
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
    if (
        expected_dimension not in {None, 0}
        and actual_dimension is not None
        and int(expected_dimension) != actual_dimension
    ):
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


_PROCESS_PERSISTENCE_LOCK = threading.RLock()


def _persist_store_snapshot(
    directory: Path,
    documents: list[Document],
    embeddings: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    """Commit a recoverable multi-file store snapshot.

    Data files are staged and fsynced first. Schema-v3 manifests authenticate
    both files and are replaced only after their directory entries are durable.
    Before replacing an existing v3 snapshot, hard-linked recovery copies keep
    the manifest's previous generation readable if the process dies halfway
    through the two data-file replacements.
    """

    with _store_file_lock(directory):
        previous = None
        if (directory / "manifest.json").exists():
            previous = _read_store_manifest(directory)
            previous = _upgrade_legacy_manifest(directory, previous)
        staged: list[tuple[Path, Path]] = []
        backups: list[tuple[Path, Path]] = []
        committed = False
        try:
            documents_stage = _stage_documents(directory, documents)
            staged.append((documents_stage, directory / "documents.jsonl"))
            embeddings_stage = _stage_embeddings(directory, embeddings)
            staged.append((embeddings_stage, directory / "embeddings.npy"))
            manifest["documents"] = _file_descriptor(documents_stage, "documents.jsonl")
            manifest["embeddings"] = _file_descriptor(embeddings_stage, "embeddings.npy")
            manifest_stage = _stage_text(
                directory,
                "manifest",
                json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
            )
            staged.append((manifest_stage, directory / "manifest.json"))

            if previous is not None:
                backups = _create_snapshot_backups(directory, previous)
                if backups:
                    _fsync_directory(directory)

            # Keep the historical destination names and commit marker while
            # making the prior generation recoverable through authenticated
            # backup hard links.
            os.replace(documents_stage, directory / "documents.jsonl")
            os.replace(embeddings_stage, directory / "embeddings.npy")
            _fsync_directory(directory)
            os.replace(manifest_stage, directory / "manifest.json")
            _fsync_directory(directory)
            committed = True
        except Exception:
            if backups:
                _restore_snapshot_backups(directory, backups)
            raise
        finally:
            for temporary_path, _ in staged:
                temporary_path.unlink(missing_ok=True)
            if committed:
                removed_backup = False
                for pattern in (
                    ".documents.*.snapshot",
                    ".embeddings.*.snapshot",
                    ".manifest.*.snapshot",
                ):
                    for backup_path in directory.glob(pattern):
                        backup_path.unlink(missing_ok=True)
                        removed_backup = True
                if removed_backup:
                    _fsync_directory(directory)


@contextmanager
def _store_file_lock(
    directory: Path,
    timeout: float = 30.0,
    *,
    exclusive: bool = True,
) -> Iterator[None]:
    """Serialize saves across threads and, where available, processes."""

    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(float(timeout))
        or timeout < 0
    ):
        raise ValueError("vector store lock timeout must be a finite number >= 0")
    lock_path = directory / ".store.lock"
    with _PROCESS_PERSISTENCE_LOCK:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows fallback
            with _portable_store_file_lock(
                directory / ".store.lock.portable",
                timeout=float(timeout),
                read_only=not exclusive,
            ):
                yield
            return
        if exclusive:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        else:
            try:
                fd = os.open(lock_path, os.O_RDONLY)
            except FileNotFoundError:
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_RDONLY, 0o600)
                except PermissionError:  # Read-only legacy index; no writer can race.
                    yield
                    return
        try:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                    fcntl.flock(fd, operation | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Vector store is locked: {lock_path}")
                    time.sleep(min(0.05, remaining))
            try:
                if exclusive:
                    os.ftruncate(fd, 0)
                    os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
                    os.fsync(fd)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextmanager
def _portable_store_file_lock(
    lock_path: Path,
    *,
    timeout: float,
    read_only: bool,
) -> Iterator[None]:
    deadline = time.monotonic() + timeout
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(
                fd,
                json.dumps({"pid": os.getpid(), "acquired_at": time.time()}).encode("ascii"),
            )
            os.fsync(fd)
        except FileExistsError:
            if _remove_abandoned_store_lock(lock_path):
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Vector store is locked: {lock_path}")
            time.sleep(min(0.05, remaining))
        except PermissionError:
            if fd is not None:
                os.close(fd)
                fd = None
                lock_path.unlink(missing_ok=True)
            if read_only:
                # A read-only directory cannot have a concurrent local writer.
                yield
                return
            raise
        except Exception:
            if fd is not None:
                os.close(fd)
                fd = None
                lock_path.unlink(missing_ok=True)
            raise
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        lock_path.unlink(missing_ok=True)


def _remove_abandoned_store_lock(lock_path: Path) -> bool:
    try:
        stat = lock_path.stat()
        payload = json.loads(lock_path.read_text(encoding="ascii"))
    except FileNotFoundError:
        return True
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        try:
            if time.time() - lock_path.stat().st_mtime < 300:
                return False
        except OSError:
            return True
    else:
        pid = payload.get("pid") if isinstance(payload, dict) else None
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            except (PermissionError, OSError):
                return False
            else:
                return False
        elif time.time() - stat.st_mtime < 300:
            return False
    try:
        lock_path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _file_descriptor(path: Path, filename: str) -> dict[str, Any]:
    return {
        "filename": filename,
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _upgrade_legacy_manifest(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Make a valid v1/v2 snapshot recoverable before its first v3 rewrite."""

    if int(manifest.get("schema_version", 1)) >= 3:
        return manifest
    documents = directory / "documents.jsonl"
    embeddings = directory / "embeddings.npy"
    if not documents.is_file() or not embeddings.is_file():
        # No complete prior generation exists; the explicit save is a repair.
        return manifest
    upgraded = dict(manifest)
    upgraded["schema_version"] = 3
    upgraded["documents"] = _file_descriptor(documents, "documents.jsonl")
    upgraded["embeddings"] = _file_descriptor(embeddings, "embeddings.npy")
    temporary = _stage_text(
        directory,
        "manifest-upgrade",
        json.dumps(upgraded, ensure_ascii=False, indent=2, allow_nan=False),
    )
    try:
        os.replace(temporary, directory / "manifest.json")
        _fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return _read_store_manifest(directory)


def _create_snapshot_backups(
    directory: Path,
    manifest: dict[str, Any],
) -> list[tuple[Path, Path]]:
    if int(manifest.get("schema_version", 1)) < 3:
        return []
    candidates: list[tuple[str, Path, str]] = []
    for field, default_name in (("documents", "documents.jsonl"), ("embeddings", "embeddings.npy")):
        descriptor = manifest[field]
        source = directory / default_name
        expected = str(descriptor["sha256"])
        # There is no valid previous generation to preserve when a user is
        # explicitly repairing an already incomplete/corrupt snapshot.
        if not source.exists() or _file_sha256(source) != expected:
            return []
        candidates.append((field, source, expected))
    # The manifest is the commit marker and must be part of the same rollback
    # set as its authenticated data.  It is deliberately appended last so a
    # restoration never publishes it before the old data files are back.
    manifest_path = directory / "manifest.json"
    candidates.append(("manifest", manifest_path, _file_sha256(manifest_path)))
    backups: list[tuple[Path, Path]] = []
    for field, source, expected in candidates:
        backup = _snapshot_backup_path(directory, field, expected)
        if backup.exists() and _file_sha256(backup) != expected:
            backup.unlink()
        if not backup.exists():
            try:
                os.link(source, backup)
            except OSError:  # pragma: no cover - filesystem-specific fallback
                _copy_file_durable(source, backup)
        backups.append((backup, source))
    return backups


def _restore_snapshot_backups(directory: Path, backups: list[tuple[Path, Path]]) -> None:
    # Restore the commit marker last.  This preserves the invariant that every
    # visible manifest references a complete on-disk data generation, including
    # when the failed operation was the directory fsync after manifest replace.
    ordered = sorted(backups, key=lambda item: item[1].name == "manifest.json")
    for backup, destination in ordered:
        if not backup.exists():
            continue
        fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.restore.", dir=directory)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            temporary.unlink()
            try:
                os.link(backup, temporary)
            except OSError:  # pragma: no cover - filesystem-specific fallback
                _copy_file_durable(backup, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    _fsync_directory(directory)


def _copy_file_durable(source: Path, destination: Path) -> None:
    with source.open("rb") as source_file, destination.open("xb") as destination_file:
        shutil.copyfileobj(source_file, destination_file)
        destination_file.flush()
        os.fsync(destination_file.fileno())


def _snapshot_backup_path(directory: Path, field: str, sha256: str) -> Path:
    return directory / f".{field}.{sha256}.snapshot"


def _document_to_dict(doc: Document) -> dict:
    return {"content": doc.content, "metadata": doc.metadata, "doc_id": doc.doc_id, "score": doc.score}


def _document_from_dict(data: dict) -> Document:
    if not isinstance(data, dict):
        raise TypeError("persisted document must be a JSON object")
    if "content" not in data:
        raise ValueError("persisted document is missing content")
    metadata = data.get("metadata", {})
    if metadata is None:
        metadata = {}
    document = Document(
        content=data["content"],
        metadata=metadata,
        doc_id=data.get("doc_id"),
        score=data.get("score"),
    )
    _validate_document(document, label="persisted document")
    return document


def _validate_document(document: Any, *, label: str) -> None:
    if not isinstance(document, Document):
        raise TypeError(f"{label} must be a Document")
    if not isinstance(document.content, str) or not document.content.strip():
        raise ValueError(f"{label}.content must be a non-empty string")
    if not isinstance(document.metadata, dict):
        raise TypeError(f"{label}.metadata must be a dict")
    if document.doc_id is not None and not isinstance(document.doc_id, str):
        raise TypeError(f"{label}.doc_id must be a string or None")
    if document.score is not None:
        if isinstance(document.score, bool) or not isinstance(document.score, (int, float)):
            raise TypeError(f"{label}.score must be a finite number or None")
        if not _is_finite_number(document.score):
            raise ValueError(f"{label}.score must be finite")


def _is_finite_number(value: int | float) -> bool:
    return math.isfinite(float(value))


def _validate_finite_embeddings(value: Any, *, label: str, corruption: bool = False) -> None:
    np = _numpy()
    array = np.asarray(value)
    prefix = "Vector store is corrupted: " if corruption else ""
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{prefix}{label} must contain real numeric values")
    try:
        finite = bool(np.isfinite(array).all())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix}{label} must contain numeric values") from exc
    if not finite:
        raise ValueError(f"{prefix}{label} contains non-finite values")


def _snapshot_data_path(
    directory: Path,
    manifest: dict[str, Any],
    field: str,
    default_filename: str,
) -> Path:
    current = directory / default_filename
    if int(manifest.get("schema_version", 1)) < 3:
        return current
    descriptor = manifest[field]
    expected = str(descriptor["sha256"])
    if current.exists() and _file_sha256(current) == expected:
        return current
    backup = _snapshot_backup_path(directory, field, expected)
    if backup.exists() and _file_sha256(backup) == expected:
        return backup
    # Return the current file so semantic validation can retain specific legacy
    # error messages (dimension/count) before the final checksum failure.
    return current


def _validate_snapshot_integrity(manifest: dict[str, Any], field: str, path: Path) -> None:
    if int(manifest.get("schema_version", 1)) < 3:
        return
    descriptor = manifest[field]
    size = path.stat().st_size
    if size != int(descriptor["size_bytes"]):
        _raise_npy_dimension_mismatch_if_known(manifest, field, path)
        raise ValueError(f"Vector store is corrupted: {field} size mismatch")
    if _file_sha256(path) != descriptor["sha256"]:
        _raise_npy_dimension_mismatch_if_known(manifest, field, path)
        raise ValueError(f"Vector store is corrupted: {field} checksum mismatch")


def _raise_npy_dimension_mismatch_if_known(
    manifest: dict[str, Any],
    field: str,
    path: Path,
) -> None:
    if field != "embeddings" or manifest.get("dimension") is None:
        return
    try:
        np = _numpy()
        with path.open("rb") as file:
            version = np.lib.format.read_magic(file)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(file)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(file)
    except Exception:
        return
    if len(shape) == 2 and int(shape[1]) != int(manifest["dimension"]):
        raise ValueError(
            "Vector store is corrupted: manifest embedding dimension "
            f"{manifest['dimension']} != stored dimension {shape[1]}"
        )


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_documents(directory: Path, documents: Iterable[Document]) -> Path:
    fd, temporary_name = tempfile.mkstemp(prefix=".documents.", suffix=".tmp", dir=directory)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            for document in documents:
                file.write(
                    json.dumps(
                        _document_to_dict(document),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
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
