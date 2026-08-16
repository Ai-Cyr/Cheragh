"""Indexing helpers, including production-safe incremental local indexing."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile
import time
from typing import Any, Iterator, Sequence

from .base import Document, EmbeddingModel, HashingEmbedding
from .ingestion import chunk_documents, load_documents
from .ingestion.pipeline import (
    _combined_exclude_patterns,
    _is_excluded,
    _looks_binary,
    _resolve_contained_candidate,
)
from .vectorstores.memory import MemoryVectorStore


_CONFIG_OVERRIDE_TARGETS = {
    "path": ("ingestion", "path"),
    "chunk_size": ("ingestion", "chunk_size"),
    "chunk_overlap": ("ingestion", "chunk_overlap"),
    "recursive": ("ingestion", "recursive"),
    "exclude_patterns": ("ingestion", "exclude_patterns"),
    "max_file_size_mb": ("ingestion", "max_file_size_mb"),
    "incremental": ("indexing", "incremental"),
    "dry_run": ("indexing", "dry_run"),
    "force": ("indexing", "force"),
    "use_lock": ("indexing", "use_lock"),
    "lock_timeout_seconds": ("indexing", "lock_timeout_seconds"),
}

_INDEXING_OPTIONS_METADATA_KEY = "indexing_options"


@dataclass
class IndexedFile:
    """Manifest entry for one indexed source file."""

    path: str
    sha256: str
    doc_ids: list[str] = field(default_factory=list)
    size_bytes: int = 0
    mtime: float = 0.0
    status: str = "indexed"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexedFile":
        if not isinstance(data, dict):
            raise ValueError("Invalid index manifest: each file entry must be an object")
        path = data.get("path")
        sha256 = data.get("sha256")
        if not isinstance(path, str) or not path:
            raise ValueError("Invalid index manifest: file path must be a non-empty string")
        if not isinstance(sha256, str) or len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise ValueError("Invalid index manifest: file sha256 must be lowercase SHA-256")
        raw_doc_ids = data.get("doc_ids") or []
        if not isinstance(raw_doc_ids, list) or any(not isinstance(doc_id, str) for doc_id in raw_doc_ids):
            raise ValueError("Invalid index manifest: doc_ids must be a list of strings")
        size_bytes = data.get("size_bytes", 0)
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError("Invalid index manifest: size_bytes must be an integer >= 0")
        mtime = data.get("mtime", 0.0)
        if isinstance(mtime, bool) or not isinstance(mtime, (int, float)) or not math.isfinite(float(mtime)):
            raise ValueError("Invalid index manifest: mtime must be finite")
        status = data.get("status", "indexed")
        if not isinstance(status, str) or not status:
            raise ValueError("Invalid index manifest: status must be a non-empty string")
        return cls(
            path=path,
            sha256=sha256,
            doc_ids=list(raw_doc_ids),
            size_bytes=size_bytes,
            mtime=float(mtime),
            status=status,
        )


@dataclass
class IndexManifest:
    """Manifest persisted next to a local vector index."""

    schema_version: int = 3
    files: dict[str, IndexedFile] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "files": {path: entry.__dict__ for path, entry in self.files.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexManifest":
        if not isinstance(data, dict):
            raise ValueError("Invalid index manifest: expected a JSON object")
        schema_version = data.get("schema_version", 3)
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("Invalid index manifest: schema_version must be an integer")
        if schema_version not in {1, 2, 3}:
            raise ValueError(f"Unsupported index manifest schema_version: {schema_version}")
        raw_files = data.get("files") or {}
        if not isinstance(raw_files, dict):
            raise ValueError("Invalid index manifest: files must be an object")
        files: dict[str, IndexedFile] = {}
        for path, entry in raw_files.items():
            if not isinstance(path, str) or not path:
                raise ValueError("Invalid index manifest: file keys must be non-empty strings")
            parsed = IndexedFile.from_dict(entry)
            if parsed.path != path:
                raise ValueError("Invalid index manifest: file key/path mismatch")
            files[path] = parsed
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("Invalid index manifest: metadata must be an object")
        return cls(schema_version=schema_version, files=files, metadata=metadata)


@dataclass
class IndexPlan:
    """Computed work needed to reconcile an index with a source tree."""

    changed_files: list[str]
    unchanged_files: list[str]
    deleted_files: list[str]
    skipped_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_files": self.changed_files,
            "unchanged_files": self.unchanged_files,
            "deleted_files": self.deleted_files,
            "skipped_files": self.skipped_files,
            "changed_count": len(self.changed_files),
            "unchanged_count": len(self.unchanged_files),
            "deleted_count": len(self.deleted_files),
            "skipped_count": len(self.skipped_files),
        }


@dataclass
class IndexOptions:
    """Options for :func:`index_path`."""

    chunk_size: int = 800
    chunk_overlap: int = 120
    recursive: bool = True
    incremental: bool = True
    dry_run: bool = False
    force: bool = False
    include_pdf: bool = True
    include_docx: bool = True
    exclude_patterns: Sequence[str] | None = None
    max_file_size_mb: float | None = 50
    use_lock: bool = True
    lock_timeout_seconds: float = 10.0


def index_from_config(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
    embedding_model: EmbeddingModel | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a local memory index from a validated Cheragh config.

    ``ingestion.path``, ``vectorstore.path`` and an explicit ``output`` are
    resolved relative to the config file.  Explicit output wins over
    ``vectorstore.path``; when neither is set, ``.cheragh_index`` next to the
    config file is used.

    The accepted overrides mirror the ingestion and indexing options consumed
    by :func:`index_path`. Unknown or ``None`` overrides are rejected so a typo
    cannot silently change the indexing contract.

    Only the local :class:`MemoryVectorStore` persistence format is supported.
    Remote or backend-specific vector stores must use their own ingestion API.
    """

    from .config import load_config
    from .config.schema import validate_config

    config_file = Path(config_path).expanduser().resolve()
    config_dir = config_file.parent
    # Match RAGEngine's configuration semantics: resolve exact ${ENV_VAR}
    # references before initial validation and before applying explicit caller
    # overrides. Secret values are never included in resolution errors.
    config = load_config(config_file, validate=False)
    validate_config(config)

    unknown = sorted(set(overrides) - set(_CONFIG_OVERRIDE_TARGETS))
    if unknown:
        supported = ", ".join(sorted(_CONFIG_OVERRIDE_TARGETS))
        raise TypeError(
            f"Unsupported index_from_config override(s): {', '.join(unknown)}. "
            f"Supported overrides: {supported}"
        )
    null_overrides = sorted(key for key, value in overrides.items() if value is None)
    if null_overrides:
        raise ValueError(
            "index_from_config overrides cannot be None: " + ", ".join(null_overrides)
        )

    for key, value in overrides.items():
        section, field_name = _CONFIG_OVERRIDE_TARGETS[key]
        if key == "path" and isinstance(value, Path):
            value = str(value)
        config.setdefault(section, {})[field_name] = value

    # Revalidate after applying overrides so their types and cross-field
    # constraints are held to the same contract as values loaded from disk.
    validated = validate_config(config)
    ingestion = validated.ingestion.model_dump()
    indexing = validated.indexing.model_dump()
    vectorstore = validated.vectorstore.model_dump()

    vectorstore_type = vectorstore.get("type")
    if vectorstore_type not in {None, "memory", "vector"}:
        raise ValueError(
            "index_from_config only writes the local MemoryVectorStore format; "
            f"vectorstore.type={vectorstore_type!r} is not supported"
        )

    source = ingestion.get("path")
    if not source:
        raise ValueError("Config must define ingestion.path for index_from_config")
    source_path = _resolve_config_path(config_dir, source)

    configured_output = output if output is not None else vectorstore.get("path")
    output_path = _resolve_config_path(
        config_dir,
        configured_output if configured_output is not None else ".cheragh_index",
    )

    if embedding_model is None:
        # Imported lazily to keep indexing independent from the engine module at
        # import time while reusing the package's provider factory.
        from .engine import _embedding_from_config

        embedding_model = _embedding_from_config(validated.embedding.model_dump(exclude_none=True))

    return index_path(
        source_path,
        output_path,
        embedding_model=embedding_model,
        chunk_size=ingestion.get("chunk_size", 800),
        chunk_overlap=ingestion.get("chunk_overlap", 120),
        recursive=ingestion.get("recursive", True),
        incremental=indexing.get("incremental", True),
        dry_run=indexing.get("dry_run", False),
        force=indexing.get("force", False),
        exclude_patterns=ingestion.get("exclude_patterns") or None,
        max_file_size_mb=ingestion.get("max_file_size_mb"),
        use_lock=indexing.get("use_lock", True),
        lock_timeout_seconds=indexing.get("lock_timeout_seconds", 10.0),
    )


def index_path(
    path: str | Path,
    output: str | Path,
    embedding_model: EmbeddingModel | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    recursive: bool = True,
    incremental: bool = True,
    include_pdf: bool = True,
    include_docx: bool = True,
    dry_run: bool = False,
    force: bool = False,
    exclude_patterns: Sequence[str] | None = None,
    max_file_size_mb: float | None = 50,
    use_lock: bool = True,
    lock_timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Index a path into a :class:`MemoryVectorStore`.

    The v1.0 implementation is incremental by default. It persists a manifest
    with file hash/mtime/size, keeps unchanged chunks, removes deleted source
    chunks, supports dry-runs, and uses a simple lock file to avoid concurrent
    writers corrupting the local index.
    """

    options = IndexOptions(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        recursive=recursive,
        incremental=incremental,
        dry_run=dry_run,
        force=force,
        include_pdf=include_pdf,
        include_docx=include_docx,
        exclude_patterns=exclude_patterns,
        max_file_size_mb=max_file_size_mb,
        use_lock=use_lock,
        lock_timeout_seconds=lock_timeout_seconds,
    )
    embedder = embedding_model or HashingEmbedding()
    source_root = Path(path)
    source_boundary = (
        source_root.parent if source_root.is_file() else source_root
    ).resolve(strict=True)
    output_path = Path(output)
    scan_excludes = list(options.exclude_patterns or ())
    if source_root.exists() and source_root.is_dir():
        try:
            output_relative = output_path.resolve().relative_to(source_root.resolve())
        except ValueError:
            output_relative = None
        if output_relative is not None:
            if not output_relative.parts:
                raise ValueError("The index output directory cannot be the source directory")
            relative = output_relative.as_posix()
            scan_excludes.extend((relative, f"{relative}/**"))
    # Planning a dry run must not mutate the filesystem, including creating an
    # otherwise empty output directory.
    if not options.dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    with _index_lock(
        output_path,
        enabled=options.use_lock and not options.dry_run,
        timeout=options.lock_timeout_seconds,
    ):
        previous = load_manifest(output_path) if options.incremental else IndexManifest()
        current_entries, skipped = scan_indexable_files(
            source_root,
            recursive=options.recursive,
            include_pdf=options.include_pdf,
            include_docx=options.include_docx,
            exclude_patterns=scan_excludes,
            max_file_size_mb=options.max_file_size_mb,
        )
        embedding_changed = bool(
            previous.files
            and previous.metadata.get("embedding_model")
            and previous.metadata.get("embedding_model") != embedder.get_fingerprint()
        )
        current_indexing_options = _indexing_options_metadata(options)
        previous_indexing_options = previous.metadata.get(_INDEXING_OPTIONS_METADATA_KEY)
        indexing_options_changed = bool(
            previous.files
            and (
                not isinstance(previous_indexing_options, dict)
                or previous_indexing_options != current_indexing_options
            )
        )
        required_store_files = (
            output_path / "documents.jsonl",
            output_path / "embeddings.npy",
            output_path / "manifest.json",
        )
        store_files_available = all(file_path.is_file() for file_path in required_store_files)
        expected_store_manifest_sha256 = previous.metadata.get("vector_store_manifest_sha256")
        store_snapshot_changed = bool(previous.files and not store_files_available)
        if previous.files and store_files_available and isinstance(expected_store_manifest_sha256, str):
            try:
                store_snapshot_changed = (
                    file_sha256(output_path / "manifest.json") != expected_store_manifest_sha256
                )
            except OSError:
                store_snapshot_changed = True
        plan = plan_incremental_update(
            previous,
            current_entries,
            force=(
                options.force
                or not options.incremental
                or embedding_changed
                or indexing_options_changed
                or store_snapshot_changed
            ),
        )
        plan.skipped_files.extend(skipped)

        if options.dry_run:
            return {
                "dry_run": True,
                "indexed_documents": None,
                "output": str(output_path),
                "plan": plan.to_dict(),
                "embedding_changed": embedding_changed,
                "indexing_options_changed": indexing_options_changed,
                "store_snapshot_changed": store_snapshot_changed,
            }

        kept_docs: list[Document] = []
        kept_embeddings = None
        if (
            options.incremental
            and not embedding_changed
            and not indexing_options_changed
            and not store_snapshot_changed
            and store_files_available
        ):
            existing = MemoryVectorStore.load(output_path, embedder)
            dirty_sources = set(plan.changed_files) | set(plan.deleted_files)
            kept_indices = [
                index
                for index, doc in enumerate(existing.documents)
                if _resolved_source(doc) not in dirty_sources
            ]
            kept_docs = [existing.documents[index] for index in kept_indices]
            if existing.embeddings is not None and kept_indices:
                kept_embeddings = existing.embeddings[kept_indices]

        new_docs: list[Document] = []
        for source in plan.changed_files:
            file_path = _resolve_contained_candidate(Path(source), source_boundary)
            if str(file_path) != source:
                raise RuntimeError(f"Source changed during indexing: {source}")
            loaded = load_documents(
                file_path,
                recursive=False,
                include_pdf=options.include_pdf,
                include_docx=options.include_docx,
                exclude_patterns=scan_excludes,
                max_file_size_mb=options.max_file_size_mb,
            )
            chunks = chunk_documents(loaded, chunk_size=options.chunk_size, chunk_overlap=options.chunk_overlap)
            # The planning hash and the bytes consumed by a loader must describe
            # the same generation. Abort before committing anything if a source
            # is edited concurrently; the next run can safely retry it.
            try:
                confirmed_path = _resolve_contained_candidate(file_path, source_boundary)
                if confirmed_path != file_path:
                    raise RuntimeError(f"Source changed during indexing: {file_path}")
                _, indexed_sha256 = _stable_file_fingerprint(confirmed_path)
            except OSError as exc:
                raise RuntimeError(f"Source changed during indexing: {file_path}") from exc
            if indexed_sha256 != current_entries[source].sha256:
                raise RuntimeError(f"Source changed during indexing: {file_path}")
            new_docs.extend(chunks)

        store = MemoryVectorStore(embedder)
        store.documents = kept_docs
        store.embeddings = kept_embeddings
        store.add_documents(new_docs)
        all_docs = store.documents
        store.save(output_path)

        doc_ids_by_source: dict[str, list[str]] = {path: [] for path in current_entries}
        for doc in all_docs:
            source = _resolved_source(doc)
            if source and doc.doc_id:
                doc_ids_by_source.setdefault(source, []).append(doc.doc_id)

        new_manifest = IndexManifest(
            metadata={
                "chunk_size": options.chunk_size,
                "chunk_overlap": options.chunk_overlap,
                "updated_at_unix": time.time(),
                "embedding_model": embedder.get_fingerprint(),
                "incremental": options.incremental,
                "vector_store_manifest_sha256": file_sha256(output_path / "manifest.json"),
                _INDEXING_OPTIONS_METADATA_KEY: current_indexing_options,
            }
        )
        for source, entry in current_entries.items():
            new_manifest.files[source] = IndexedFile(
                path=source,
                sha256=entry.sha256,
                size_bytes=entry.size_bytes,
                mtime=entry.mtime,
                doc_ids=doc_ids_by_source.get(source, []),
                status="indexed",
            )
        save_manifest(output_path, new_manifest)

        return {
            "dry_run": False,
            "indexed_documents": len(all_docs),
            "changed_files": len(plan.changed_files),
            "deleted_files": len(plan.deleted_files),
            "unchanged_files": len(plan.unchanged_files),
            "skipped_files": len(plan.skipped_files),
            "output": str(output_path),
            "plan": plan.to_dict(),
            "embedding_changed": embedding_changed,
            "indexing_options_changed": indexing_options_changed,
            "store_snapshot_changed": store_snapshot_changed,
        }


def scan_indexable_files(
    path: str | Path,
    recursive: bool = True,
    include_pdf: bool = True,
    include_docx: bool = True,
    exclude_patterns: Sequence[str] | None = None,
    max_file_size_mb: float | None = 50,
) -> tuple[dict[str, IndexedFile], list[str]]:
    """Return content fingerprints for files that should be indexed."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(str(source))
    candidates = _candidate_files(source, recursive=recursive, include_pdf=include_pdf, include_docx=include_docx)
    root = source.parent if source.is_file() else source
    resolved_root = root.resolve(strict=True)
    patterns = _combined_exclude_patterns(exclude_patterns)
    max_bytes = None if max_file_size_mb is None else int(max_file_size_mb * 1024 * 1024)
    entries: dict[str, IndexedFile] = {}
    skipped: list[str] = []
    for file_path in candidates:
        resolved_path = _resolve_contained_candidate(file_path, resolved_root)
        resolved = str(resolved_path)
        if _is_excluded(file_path, root, patterns) or _looks_binary(resolved_path):
            skipped.append(resolved)
            continue
        stat = resolved_path.stat()
        if max_bytes is not None and stat.st_size > max_bytes:
            skipped.append(resolved)
            continue
        stat, digest = _stable_file_fingerprint(resolved_path)
        if max_bytes is not None and stat.st_size > max_bytes:
            skipped.append(resolved)
            continue
        entries[resolved] = IndexedFile(
            path=resolved,
            sha256=digest,
            size_bytes=int(stat.st_size),
            mtime=float(stat.st_mtime),
        )
    return entries, skipped


def plan_incremental_update(
    previous: IndexManifest,
    current_files: dict[str, IndexedFile],
    *,
    force: bool = False,
) -> IndexPlan:
    if force:
        changed = sorted(current_files)
        unchanged: list[str] = []
    else:
        changed = sorted(
            path
            for path, entry in current_files.items()
            if path not in previous.files or previous.files[path].sha256 != entry.sha256
        )
        unchanged = sorted(path for path in current_files if path not in set(changed))
    deleted = sorted(path for path in previous.files if path not in current_files)
    return IndexPlan(changed_files=changed, unchanged_files=unchanged, deleted_files=deleted)


def load_manifest(index_path: str | Path) -> IndexManifest:
    manifest_path = Path(index_path) / "index_manifest.json"
    if not manifest_path.exists():
        return IndexManifest()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return IndexManifest.from_dict(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid index manifest: {manifest_path}") from exc


def save_manifest(index_path: str | Path, manifest: IndexManifest) -> None:
    directory = Path(index_path)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "index_manifest.json"
    fd, temporary_name = tempfile.mkstemp(prefix=".index_manifest.", suffix=".tmp", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(manifest.to_dict(), file, ensure_ascii=False, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
        _fsync_directory(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def inspect_index(index_path: str | Path) -> dict[str, Any]:
    p = Path(index_path)
    manifest = load_manifest(p)
    count = 0
    docs_path = p / "documents.jsonl"
    if docs_path.exists():
        with docs_path.open("r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
    total_size = sum(entry.size_bytes for entry in manifest.files.values())
    return {
        "path": str(p),
        "documents": count,
        "files": len(manifest.files),
        "total_source_size_bytes": total_size,
        "schema_version": manifest.schema_version,
        "manifest_available": bool(manifest.files),
        "metadata": manifest.metadata,
    }


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_file_fingerprint(path: Path, *, attempts: int = 3) -> tuple[os.stat_result, str]:
    """Hash one stable on-disk generation or fail rather than index a torn read."""

    for _ in range(attempts):
        before = path.stat()
        digest = file_sha256(path)
        after = path.stat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before == identity_after:
            return after, digest
    raise RuntimeError(f"Source changed while hashing: {path}")


def _candidate_files(path: Path, recursive: bool, include_pdf: bool, include_docx: bool) -> list[Path]:
    if path.is_file():
        return [path]
    globber = path.rglob if recursive else path.glob
    allowed = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".xml", ".html", ".htm"}
    if include_pdf:
        allowed.add(".pdf")
    if include_docx:
        allowed.add(".docx")
    return [child for child in globber("*") if child.is_file() and child.suffix.lower() in allowed]


def _resolved_source(doc: Document) -> str:
    source = doc.metadata.get("source") if doc.metadata else None
    return str(Path(str(source)).resolve()) if source else ""


def _indexing_options_metadata(options: IndexOptions) -> dict[str, Any]:
    """Return the persisted contract that determines chunks and corpus scope."""

    return {
        "chunk_size": options.chunk_size,
        "chunk_overlap": options.chunk_overlap,
        "recursive": options.recursive,
        "include_pdf": options.include_pdf,
        "include_docx": options.include_docx,
        # Exclusion order and duplicates do not affect matching semantics.
        "exclude_patterns": sorted({str(pattern) for pattern in (options.exclude_patterns or ())}),
        "max_file_size_mb": options.max_file_size_mb,
    }


def _resolve_config_path(config_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_dir / path).resolve()


@contextmanager
def _index_lock(index_path: Path, *, enabled: bool, timeout: float = 10.0) -> Iterator[None]:
    if not enabled:
        yield
        return
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("lock timeout must be a finite number >= 0")
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("lock timeout must be a finite number >= 0")
    lock_path = index_path / ".index.lock"
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows fallback
        with _exclusive_create_lock(lock_path, timeout=timeout):
            yield
        return

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Index is locked: {lock_path}")
                time.sleep(min(0.05, remaining))
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"pid": os.getpid(), "acquired_at": time.time()}).encode("ascii"))
        os.fsync(fd)
        try:
            yield
        finally:
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps({"released_at": time.time()}).encode("ascii"))
            os.fsync(fd)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def _exclusive_create_lock(lock_path: Path, *, timeout: float) -> Iterator[None]:
    """Portable fallback for platforms without advisory ``flock`` support."""

    deadline = time.monotonic() + timeout
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(fd, json.dumps({"pid": os.getpid(), "acquired_at": time.time()}).encode("ascii"))
            os.fsync(fd)
        except FileExistsError:
            if _remove_abandoned_lock(lock_path):
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Index is locked: {lock_path}")
            time.sleep(min(0.05, remaining))
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        lock_path.unlink(missing_ok=True)


def _remove_abandoned_lock(lock_path: Path, *, invalid_after_seconds: float = 300.0) -> bool:
    """Best-effort stale owner recovery for the non-POSIX lock fallback."""

    try:
        stat = lock_path.stat()
        payload = json.loads(lock_path.read_text(encoding="ascii"))
    except FileNotFoundError:
        return True
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        try:
            age = max(0.0, time.time() - lock_path.stat().st_mtime)
        except OSError:
            return True
        if age < invalid_after_seconds:
            return False
        try:
            lock_path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False
    pid = payload.get("pid") if isinstance(payload, dict) else None
    acquired_at = payload.get("acquired_at") if isinstance(payload, dict) else None
    if isinstance(payload, dict) and "released_at" in payload:
        try:
            lock_path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(acquired_at, bool)
        or not isinstance(acquired_at, (int, float))
        or not math.isfinite(float(acquired_at))
    ):
        age = max(0.0, time.time() - stat.st_mtime)
        if age < invalid_after_seconds:
            return False
    elif _process_is_alive(pid):
        return False
    try:
        lock_path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # Conservatively retain locks when liveness cannot be determined.
        return True
    return True


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError:  # pragma: no cover - platform/filesystem-specific
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = [
    "IndexedFile",
    "IndexManifest",
    "IndexOptions",
    "IndexPlan",
    "file_sha256",
    "index_from_config",
    "index_path",
    "inspect_index",
    "load_manifest",
    "plan_incremental_update",
    "save_manifest",
    "scan_indexable_files",
]
