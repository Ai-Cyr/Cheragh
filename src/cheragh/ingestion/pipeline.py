"""High-level ingestion helpers."""
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Sequence

from ..base import Document
from .chunkers import chunk_documents
from .loaders.docx import load_docx_file
from .loaders.pdf import load_pdf_file
from .loaders.text import load_html_file, load_text_file, supports_html, supports_text

DEFAULT_EXCLUDE_PATTERNS = (
    ".git/**",
    ".hg/**",
    ".svn/**",
    "__pycache__/**",
    ".pytest_cache/**",
    ".mypy_cache/**",
    ".ruff_cache/**",
    ".venv/**",
    "venv/**",
    "env/**",
    "node_modules/**",
    ".cheragh/**",
    ".cheragh_index/**",
    "dist/**",
    "build/**",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dylib",
    "*.dll",
)


def load_documents(
    path: str | Path,
    recursive: bool = True,
    include_pdf: bool = True,
    include_docx: bool = True,
    encoding: str = "utf-8",
    exclude_patterns: Sequence[str] | None = None,
    max_file_size_mb: float | None = 50,
) -> list[Document]:
    """Load documents from a file or directory.

    Supported without optional dependencies: txt, markdown, rst, csv, json,
    jsonl, yaml, xml and simple HTML. PDF and DOCX require extras.

    ``exclude_patterns`` uses shell-style globs relative to the input directory.
    ``max_file_size_mb`` avoids accidentally indexing huge generated files.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    patterns = _combined_exclude_patterns(exclude_patterns)
    root = p.parent if p.is_file() else p
    resolved_root = root.resolve(strict=True)
    files = [p] if p.is_file() else list(_iter_candidate_files(p, recursive=recursive, exclude_patterns=patterns))
    documents: list[Document] = []
    for file_path in files:
        if _is_excluded(file_path, root, patterns):
            continue
        # Resolve every candidate against the canonical source root before any
        # stat/read.  In particular, never follow a file symlink (including one
        # below nested directories) outside the corpus selected by the caller.
        resolved_file = _resolve_contained_candidate(file_path, resolved_root)
        if max_file_size_mb is not None and resolved_file.stat().st_size > max_file_size_mb * 1024 * 1024:
            continue
        if _looks_binary(resolved_file):
            continue
        # Use the canonical path for the loader.  A later replacement of the
        # lexical symlink cannot redirect this read.  Re-resolve immediately
        # before and after the loader to fail closed if a canonical component
        # was concurrently replaced.
        read_path = _resolve_contained_candidate(resolved_file, resolved_root)
        suffix = read_path.suffix.lower()
        loaded: list[Document] = []
        if supports_text(read_path):
            loaded.append(load_text_file(read_path, encoding=encoding))
        elif supports_html(read_path):
            loaded.append(load_html_file(read_path, encoding=encoding))
        elif include_pdf and suffix == ".pdf":
            loaded.extend(load_pdf_file(read_path))
        elif include_docx and suffix == ".docx":
            loaded.append(load_docx_file(read_path))
        confirmed_path = _resolve_contained_candidate(read_path, resolved_root)
        if confirmed_path != read_path:
            raise RuntimeError(f"Source changed during ingestion: {read_path}")
        documents.extend(loaded)
    return [doc for doc in documents if doc.content.strip()]


def ingest_path(
    path: str | Path,
    recursive: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    **loader_kwargs,
) -> list[Document]:
    """Load then chunk a path in one call."""
    docs = load_documents(path, recursive=recursive, **loader_kwargs)
    return chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _iter_candidate_files(
    path: Path,
    recursive: bool = True,
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
) -> Iterable[Path]:
    globber = path.rglob if recursive else path.glob
    for child in globber("*"):
        if child.is_file() and not _is_excluded(child, path, exclude_patterns):
            yield child


def _is_excluded(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.name
    for pattern in patterns:
        normalized = pattern.replace("\\", "/")
        if fnmatch(rel, normalized) or fnmatch(path.name, normalized):
            return True
        if normalized.endswith("/**") and rel.startswith(normalized[:-3].rstrip("/") + "/"):
            return True
    return False


def _looks_binary(path: Path, sample_size: int = 4096) -> bool:
    if path.suffix.lower() in {".pdf", ".docx"}:
        return False
    try:
        sample = path.read_bytes()[:sample_size]
    except OSError:
        return True
    if not sample:
        return False
    return b"\x00" in sample


def _resolve_contained_candidate(path: Path, resolved_root: Path) -> Path:
    """Resolve ``path`` and require it to remain below ``resolved_root``.

    ``resolved_root`` must be the already-canonical directory defining the
    caller's corpus boundary.  ``strict=True`` ensures broken or concurrently
    removed candidates fail instead of being normalized into an unchecked
    path.
    """

    candidate = path.resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to ingest path outside source root: {path}"
        ) from exc
    if not candidate.is_file():
        raise RuntimeError(f"Source is no longer a regular file: {path}")
    return candidate


def _combined_exclude_patterns(exclude_patterns: Sequence[str] | None = None) -> tuple[str, ...]:
    """Keep safety defaults when callers add their own exclusion patterns."""

    return tuple(dict.fromkeys((*DEFAULT_EXCLUDE_PATTERNS, *(exclude_patterns or ()))))
