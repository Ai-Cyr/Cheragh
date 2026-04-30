"""Optional PDF loader."""
from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any

from ...base import Document
from .text import _file_metadata


def load_pdf_file(path: str | Path, pages: bool = True) -> list[Document]:
    """Load a PDF with page-level metadata.

    Requires the optional dependency ``pypdf``:
    ``pip install cheragh[pdf]``.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PDF loading requires pypdf. Install with: pip install cheragh[pdf]") from exc

    p = Path(path)
    reader = PdfReader(str(p))
    base = _stable_file_doc_id(p)
    file_hash = _file_sha256(p)
    page_count = len(reader.pages)
    pdf_metadata = _safe_pdf_metadata(reader)
    docs: list[Document] = []
    page_texts: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        metadata = {
            **_file_metadata(p),
            "page": index,
            "page_count": page_count,
            "file_sha256": file_hash,
            "char_count": len(text),
            **pdf_metadata,
        }
        if pages:
            docs.append(Document(content=text, doc_id=f"{base}-p{index}", metadata=metadata))
        else:
            page_texts.append(f"\n\n--- page {index} ---\n{text}")
    if not pages and page_texts:
        docs.append(
            Document(
                content="".join(page_texts).strip(),
                doc_id=base,
                metadata={**_file_metadata(p), "page_count": page_count, **pdf_metadata},
            )
        )
    return docs


def _safe_pdf_metadata(reader: Any) -> dict[str, str]:
    raw = getattr(reader, "metadata", None) or {}
    output: dict[str, str] = {}
    for key, value in dict(raw).items():
        clean_key = str(key).lstrip("/").lower()
        if clean_key in {"title", "author", "subject", "creator", "producer"} and value:
            output[f"pdf_{clean_key}"] = str(value)
    return output


def _stable_file_doc_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"file-{digest}"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
