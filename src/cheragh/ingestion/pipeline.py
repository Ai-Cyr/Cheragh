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

    patterns = tuple(exclude_patterns or DEFAULT_EXCLUDE_PATTERNS)
    root = p.parent if p.is_file() else p
    files = [p] if p.is_file() else list(_iter_candidate_files(p, recursive=recursive, exclude_patterns=patterns))
    documents: list[Document] = []
    for file_path in files:
        if _is_excluded(file_path, root, patterns):
            continue
        if max_file_size_mb is not None and file_path.stat().st_size > max_file_size_mb * 1024 * 1024:
            continue
        if _looks_binary(file_path):
            continue
        suffix = file_path.suffix.lower()
        if supports_text(file_path):
            documents.append(load_text_file(file_path, encoding=encoding))
        elif supports_html(file_path):
            documents.append(load_html_file(file_path, encoding=encoding))
        elif include_pdf and suffix == ".pdf":
            documents.extend(load_pdf_file(file_path))
        elif include_docx and suffix == ".docx":
            documents.append(load_docx_file(file_path))
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
