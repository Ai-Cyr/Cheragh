"""Indexing helpers, including production-safe incremental local indexing."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
import hashlib
import json
import os
import time
from typing import Any, Iterator, Sequence

from .base import Document, EmbeddingModel, HashingEmbedding
from .ingestion import chunk_documents, load_documents
from .ingestion.pipeline import DEFAULT_EXCLUDE_PATTERNS, _is_excluded, _looks_binary
from .vectorstores.memory import MemoryVectorStore


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
    output_path.mkdir(parents=True, exist_ok=True)

    with _index_lock(output_path, enabled=options.use_lock and not options.dry_run, timeout=options.lock_timeout_seconds):
        previous = load_manifest(output_path) if options.incremental else IndexManifest()
        current_entries, skipped = scan_indexable_files(
            source_root,
            recursive=options.recursive,
            include_pdf=options.include_pdf,
            include_docx=options.include_docx,
            exclude_patterns=options.exclude_patterns,
            max_file_size_mb=options.max_file_size_mb,
        )
        plan = plan_incremental_update(previous, current_entries, force=options.force or not options.incremental)
        plan.skipped_files.extend(skipped)

        if options.dry_run:
            return {
                "dry_run": True,
                "indexed_documents": None,
                "output": str(output_path),
                "plan": plan.to_dict(),
            }

        kept_docs: list[Document] = []
        if options.incremental and (output_path / "documents.jsonl").exists() and (output_path / "embeddings.npy").exists():
            existing = MemoryVectorStore.load(output_path, embedder)
            dirty_sources = set(plan.changed_files) | set(plan.deleted_files)
            kept_docs = [
                doc
                for doc in existing.documents
                if _resolved_source(doc) not in dirty_sources
            ]

        new_docs: list[Document] = []
        for source in plan.changed_files:
            file_path = Path(source)
            loaded = load_documents(
                file_path,
                recursive=False,
                include_pdf=options.include_pdf,
                include_docx=options.include_docx,
                exclude_patterns=options.exclude_patterns,
                max_file_size_mb=options.max_file_size_mb,
            )
            chunks = chunk_documents(loaded, chunk_size=options.chunk_size, chunk_overlap=options.chunk_overlap)
            new_docs.extend(chunks)

        all_docs = kept_docs + new_docs
        store = MemoryVectorStore(embedder)
        store.add_documents(all_docs)
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
    patterns = tuple(exclude_patterns or DEFAULT_EXCLUDE_PATTERNS)
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
    "index_path",
    "inspect_index",
    "load_manifest",
    "plan_incremental_update",
    "save_manifest",
    "scan_indexable_files",
]
