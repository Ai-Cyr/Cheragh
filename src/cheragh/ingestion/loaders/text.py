"""Plain-text and lightweight markup document loaders."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import hashlib
import html
import re

from ...base import Document

_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".xml"}
_HTML_EXTENSIONS = {".html", ".htm"}


def load_text_file(path: str | Path, encoding: str = "utf-8") -> Document:
    """Load a UTF-8 compatible text file as a :class:`Document`."""
    p = Path(path)
    content = p.read_text(encoding=encoding, errors="replace")
    return Document(content=content, doc_id=_stable_file_doc_id(p), metadata=_file_metadata(p))


def load_html_file(path: str | Path, encoding: str = "utf-8") -> Document:
    """Load a simple HTML file by stripping tags without heavyweight dependencies."""
    p = Path(path)
    raw = p.read_text(encoding=encoding, errors="replace")
    text = html_to_text(raw)
    return Document(content=text, doc_id=_stable_file_doc_id(p), metadata=_file_metadata(p))


def supports_text(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _TEXT_EXTENSIONS


def supports_html(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _HTML_EXTENSIONS


def html_to_text(raw_html: str) -> str:
    """Convert HTML to rough text using the standard library only."""
    without_scripts = re.sub(r"<\s*(script|style).*?>.*?<\s*/\s*\1\s*>", " ", raw_html, flags=re.I | re.S)
    with_breaks = re.sub(r"<\s*(br|p|div|li|h[1-6]|tr)\b[^>]*>", "\n", without_scripts, flags=re.I)
    no_tags = re.sub(r"<[^>]+>", " ", with_breaks)
    decoded = html.unescape(no_tags)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", decoded)).strip()


def iter_supported_text_files(path: str | Path, recursive: bool = True) -> Iterable[Path]:
    p = Path(path)
    if p.is_file():
        if supports_text(p) or supports_html(p):
            yield p
        return
    globber = p.rglob if recursive else p.glob
    for child in globber("*"):
        if child.is_file() and (supports_text(child) or supports_html(child)):
            yield child


def _stable_file_doc_id(path: Path) -> str:
    resolved = str(path.resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
    return f"file-{digest}"


def _file_metadata(path: Path) -> dict:
    stat = path.stat()
    return {
        "source": str(path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "file_sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
