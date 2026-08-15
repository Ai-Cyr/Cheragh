"""Indexing helpers, including production-safe incremental local indexing."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import os
import time
from typing import Any, Iterator, Sequence

from .base import Document, EmbeddingModel, HashingEmbedding
from .ingestion import chunk_documents, load_documents
from .ingestion.pipeline import _combined_exclude_patterns, _is_excluded, _looks_binary
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
        return cls(
            path=str(data.get("path", "")),
            sha256=str(data.get("sha256", "")),
            doc_ids=list(data.get("doc_ids") or []),
            size_bytes=int(data.get("size_bytes", 0) or 0),
            mtime=float(data.get("mtime", 0.0) or 0.0),
            status=str(data.get("status", "indexed")),
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
        files = {path: IndexedFile.from_dict(entry) for path, entry in (data.get("files") or {}).items()}
        return cls(schema_version=int(data.get("schema_version", 3)), files=files, metadata=data.get("metadata") or {})


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

    from .config import load_raw_config
    from .config.schema import validate_config

    config_file = Path(config_path).expanduser().resolve()
    config_dir = config_file.parent
    config = load_raw_config(config_file)
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

    with _index_lock(output_path, enabled=options.use_lock and not options.dry_run, timeout=options.lock_timeout_seconds):
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
        plan = plan_incremental_update(
            previous,
            current_entries,
            force=(
                options.force
                or not options.incremental
                or embedding_changed
                or indexing_options_changed
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
            }

        kept_docs: list[Document] = []
        kept_embeddings = None
        if (
            options.incremental
            and not embedding_changed
            and not indexing_options_changed
            and (output_path / "documents.jsonl").exists()
            and (output_path / "embeddings.npy").exists()
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
            file_path = Path(source)
            loaded = load_documents(
                file_path,
                recursive=False,
                include_pdf=options.include_pdf,
                include_docx=options.include_docx,
                exclude_patterns=scan_excludes,
                max_file_size_mb=options.max_file_size_mb,
            )
            chunks = chunk_documents(loaded, chunk_size=options.chunk_size, chunk_overlap=options.chunk_overlap)
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
    patterns = _combined_exclude_patterns(exclude_patterns)
    max_bytes = None if max_file_size_mb is None else int(max_file_size_mb * 1024 * 1024)
    entries: dict[str, IndexedFile] = {}
    skipped: list[str] = []
    for file_path in candidates:
        resolved = str(file_path.resolve())
        if _is_excluded(file_path, root, patterns) or _looks_binary(file_path):
            skipped.append(resolved)
            continue
        stat = file_path.stat()
        if max_bytes is not None and stat.st_size > max_bytes:
            skipped.append(resolved)
            continue
        entries[resolved] = IndexedFile(
            path=resolved,
            sha256=file_sha256(file_path),
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
    return IndexManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))


def save_manifest(index_path: str | Path, manifest: IndexManifest) -> None:
    Path(index_path).mkdir(parents=True, exist_ok=True)
    (Path(index_path) / "index_manifest.json").write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


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
    lock_path = index_path / ".index.lock"
    deadline = time.time() + timeout
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError(f"Index is locked: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


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
