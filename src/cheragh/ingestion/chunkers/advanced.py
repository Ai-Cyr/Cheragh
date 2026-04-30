"""Advanced structure-, semantic- and layout-aware chunkers."""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
import io
import math
import re
from typing import Iterable, Sequence

from ...base import Document, EmbeddingModel
from .recursive import RecursiveTextChunker
from .structured import HTMLSectionChunker, MarkdownHeaderChunker, _split_sentences


def _numpy():
    import numpy as np

    return np


@dataclass
class SemanticChunker:
    """Split documents where adjacent sentence meaning changes.

    The chunker embeds each sentence, computes adjacent cosine similarities and
    starts a new chunk when the similarity falls below ``breakpoint_threshold``.
    It also enforces ``max_chunk_size`` as a safety bound.
    """

    embedding_model: EmbeddingModel
    breakpoint_threshold: float = 0.75
    max_chunk_size: int = 1200
    min_chunk_size: int = 40
    min_sentences: int = 1
    fallback_chunker: RecursiveTextChunker | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.breakpoint_threshold <= 1:
            raise ValueError("breakpoint_threshold must be between 0 and 1")
        if self.max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be > 0")
        if self.min_sentences <= 0:
            raise ValueError("min_sentences must be > 0")
        if self.fallback_chunker is None:
            self.fallback_chunker = RecursiveTextChunker(
                chunk_size=self.max_chunk_size,
                chunk_overlap=max(0, min(160, self.max_chunk_size // 8)),
                min_chunk_size=self.min_chunk_size,
            )

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in documents:
            base_id = doc.doc_id or f"doc-{len(chunks)}"
            specs = self.split_text(doc.content)
            for idx, spec in enumerate(specs):
                content = str(spec["content"]).strip()
                if len(content) > self.max_chunk_size and self.fallback_chunker is not None:
                    parent = Document(content=content, metadata=dict(doc.metadata), doc_id=f"{base_id}#semantic-{idx}")
                    for fallback_doc in self.fallback_chunker.split_documents([parent]):
                        fallback_doc.metadata.update(
                            {
                                "chunk_method": "semantic+recursive",
                                "semantic_breakpoint_threshold": self.breakpoint_threshold,
                                "sentence_start": spec.get("sentence_start"),
                                "sentence_end": spec.get("sentence_end"),
                            }
                        )
                        chunks.append(fallback_doc)
                    continue
                if len(content) < self.min_chunk_size:
                    continue
                chunks.append(
                    Document(
                        content=content,
                        metadata={
                            **doc.metadata,
                            "chunk_index": idx,
                            "parent_doc_id": base_id,
                            "chunk_method": "semantic",
                            "semantic_breakpoint_threshold": self.breakpoint_threshold,
                            "sentence_start": spec.get("sentence_start"),
                            "sentence_end": spec.get("sentence_end"),
                            "avg_adjacent_similarity": spec.get("avg_adjacent_similarity"),
                        },
                        doc_id=f"{base_id}#semantic-{idx}",
                    )
                )
        return chunks

    def split_text(self, text: str) -> list[dict[str, object]]:
        sentences = _split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [{"content": sentences[0], "sentence_start": 0, "sentence_end": 0, "avg_adjacent_similarity": None}]

        embeddings = self.embedding_model.embed_documents(sentences)
        similarities = _adjacent_cosine_similarities(embeddings)
        chunks: list[dict[str, object]] = []
        start = 0
        current_sentences: list[str] = []
        current_len = 0
        current_sims: list[float] = []

        for idx, sentence in enumerate(sentences):
            current_sentences.append(sentence)
            current_len += len(sentence) + 1
            if idx > start:
                current_sims.append(float(similarities[idx - 1]))

            semantic_break = idx < len(similarities) and float(similarities[idx]) < self.breakpoint_threshold
            size_break = current_len >= self.max_chunk_size
            enough = len(current_sentences) >= self.min_sentences and current_len >= self.min_chunk_size
            if idx < len(sentences) - 1 and enough and (semantic_break or size_break):
                chunks.append(
                    {
                        "content": " ".join(current_sentences).strip(),
                        "sentence_start": start,
                        "sentence_end": idx,
                        "avg_adjacent_similarity": _safe_mean(current_sims),
                    }
                )
                start = idx + 1
                current_sentences = []
                current_len = 0
                current_sims = []

        if current_sentences:
            chunks.append(
                {
                    "content": " ".join(current_sentences).strip(),
                    "sentence_start": start,
                    "sentence_end": len(sentences) - 1,
                    "avg_adjacent_similarity": _safe_mean(current_sims),
                }
            )
        return chunks


@dataclass
class CodeChunker:
    """Split source-code-like documents around functions, classes and SQL statements."""

    language: str | None = None
    max_chunk_size: int = 1600
    chunk_overlap: int = 120
    min_chunk_size: int = 20

    def __post_init__(self) -> None:
        if self.max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be > 0")

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        output: list[Document] = []
        fallback = RecursiveTextChunker(
            chunk_size=self.max_chunk_size,
            chunk_overlap=max(0, min(self.chunk_overlap, self.max_chunk_size - 1)),
            min_chunk_size=self.min_chunk_size,
        )
        for doc in documents:
            base_id = doc.doc_id or f"doc-{len(output)}"
            language = self.language or _detect_language(doc)
            blocks = self.split_text(doc.content, language=language)
            for idx, block in enumerate(blocks):
                block_doc = Document(
                    content=block["content"],
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "parent_doc_id": base_id,
                        "chunk_method": "code",
                        "code_language": language,
                        "symbol_name": block.get("symbol_name"),
                        "start_line": block.get("start_line"),
                        "end_line": block.get("end_line"),
                    },
                    doc_id=f"{base_id}#code-{idx}",
                )
                if len(block_doc.content) > self.max_chunk_size:
                    output.extend(fallback.split_documents([block_doc]))
                elif len(block_doc.content.strip()) >= self.min_chunk_size:
                    output.append(block_doc)
        return output

    def split_text(self, text: str, language: str | None = None) -> list[dict[str, object]]:
        language = (language or "text").lower()
        if language in {"python", "py"}:
            return _split_python_code(text)
        if language in {"javascript", "typescript", "js", "ts", "tsx", "jsx"}:
            return _split_js_code(text)
        if language in {"sql", "postgres", "mysql", "sqlite"}:
            return _split_sql_code(text)
        return _split_generic_code(text)


@dataclass
class TableChunker:
    """Extract Markdown/CSV-like tables and split them by row groups."""

    rows_per_chunk: int = 20
    include_header: bool = True
    min_rows: int = 2
    delimiter_candidates: tuple[str, ...] = ("|", ",", "\t", ";")

    def __post_init__(self) -> None:
        if self.rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be > 0")

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        output: list[Document] = []
        for doc in documents:
            base_id = doc.doc_id or f"doc-{len(output)}"
            tables = self.extract_tables(doc.content)
            for table_idx, table in enumerate(tables):
                rows = table["rows"]
                if len(rows) < self.min_rows:
                    continue
                header = rows[:1]
                data_rows = rows[1:] if len(rows) > 1 else rows
                for chunk_idx, start in enumerate(range(0, len(data_rows), self.rows_per_chunk)):
                    part_rows = data_rows[start : start + self.rows_per_chunk]
                    rendered_rows = (header + part_rows) if self.include_header and header and data_rows is not rows else part_rows
                    content = _render_table(rendered_rows, table["delimiter"])
                    output.append(
                        Document(
                            content=content,
                            metadata={
                                **doc.metadata,
                                "chunk_index": chunk_idx,
                                "parent_doc_id": base_id,
                                "chunk_method": "table",
                                "table_index": table_idx,
                                "row_start": start + (1 if self.include_header and header else 0),
                                "row_end": start + len(part_rows),
                                "column_count": table["column_count"],
                                "table_format": table["format"],
                            },
                            doc_id=f"{base_id}#table-{table_idx}-{chunk_idx}",
                        )
                    )
        return output

    def extract_tables(self, text: str) -> list[dict[str, object]]:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        tables: list[dict[str, object]] = []
        i = 0
        while i < len(lines):
            best_delim = self._line_delimiter(lines[i])
            if best_delim is None:
                i += 1
                continue
            start = i
            block: list[str] = []
            while i < len(lines) and self._line_delimiter(lines[i]) == best_delim:
                if lines[i].strip():
                    block.append(lines[i])
                i += 1
            parsed = _parse_table_block(block, best_delim)
            if parsed and len(parsed["rows"]) >= self.min_rows:
                tables.append(parsed)
            if i == start:
                i += 1
        return tables

    def _line_delimiter(self, line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None
        for delim in self.delimiter_candidates:
            if delim == "|" and stripped.count("|") >= 2:
                return delim
            if delim != "|" and stripped.count(delim) >= 1:
                return delim
        return None


@dataclass
class PDFLayoutChunker:
    """Chunk page-level PDF documents into layout-inspired paragraph blocks.

    This chunker works with the metadata produced by ``load_pdf_file`` and also
    preserves optional layout metadata such as ``bbox`` or ``layout_block_id`` if
    upstream parsers provide it.
    """

    max_chunk_size: int = 1000
    min_chunk_size: int = 30
    merge_small_blocks: bool = True

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        output: list[Document] = []
        fallback = RecursiveTextChunker(chunk_size=self.max_chunk_size, chunk_overlap=80, min_chunk_size=self.min_chunk_size)
        for doc in documents:
            base_id = doc.doc_id or f"doc-{len(output)}"
            blocks = self.split_page(doc.content)
            if self.merge_small_blocks:
                blocks = _merge_small_layout_blocks(blocks, min_chars=self.min_chunk_size)
            for idx, block in enumerate(blocks):
                block_doc = Document(
                    content=block["content"],
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "parent_doc_id": base_id,
                        "chunk_method": "pdf-layout",
                        "layout_block_type": block["layout_block_type"],
                        "layout_block_index": idx,
                        "page": doc.metadata.get("page"),
                        "bbox": doc.metadata.get("bbox"),
                    },
                    doc_id=f"{base_id}#layout-{idx}",
                )
                if len(block_doc.content) > self.max_chunk_size:
                    output.extend(fallback.split_documents([block_doc]))
                elif len(block_doc.content.strip()) >= self.min_chunk_size:
                    output.append(block_doc)
        return output

    def split_page(self, text: str) -> list[dict[str, str]]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.replace("\r\n", "\n")) if p.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        return [{"content": p, "layout_block_type": _classify_layout_block(p)} for p in paragraphs]


@dataclass
class HierarchicalChunker:
    """Create parent section chunks plus child chunks with hierarchy metadata."""

    chunk_size: int = 900
    chunk_overlap: int = 120
    include_parent_sections: bool = True
    detect_format: bool = True
    min_chunk_size: int = 30

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        output: list[Document] = []
        fallback = RecursiveTextChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=max(0, min(self.chunk_overlap, self.chunk_size - 1)),
            min_chunk_size=self.min_chunk_size,
        )
        for doc in documents:
            base_id = doc.doc_id or f"doc-{len(output)}"
            sections = self._sections(doc)
            for section_idx, section in enumerate(sections):
                section_id = f"{base_id}#hsec-{section_idx}"
                section_meta = {
                    **doc.metadata,
                    "chunk_method": "hierarchical",
                    "hierarchy_level": section.get("level", 0),
                    "section": section.get("title", "document"),
                    "section_path": section.get("path", section.get("title", "document")),
                    "parent_doc_id": base_id,
                }
                section_content = str(section["content"]).strip()
                if self.include_parent_sections and len(section_content) >= self.min_chunk_size:
                    output.append(
                        Document(
                            content=section_content,
                            metadata={**section_meta, "chunk_role": "parent_section", "chunk_index": section_idx},
                            doc_id=section_id,
                        )
                    )
                children = fallback.split_documents([Document(section_content, metadata=section_meta, doc_id=section_id)])
                for child_idx, child in enumerate(children):
                    child.metadata.update(
                        {
                            "chunk_role": "child_chunk",
                            "parent_section_id": section_id,
                            "chunk_index": child_idx,
                        }
                    )
                    child.doc_id = f"{section_id}#child-{child_idx}"
                    output.append(child)
        return output

    def _sections(self, doc: Document) -> list[dict[str, object]]:
        source = str(doc.metadata.get("source") or doc.metadata.get("filename") or "").lower()
        text = doc.content
        is_html = "<h1" in text.lower() or "<html" in text.lower() or source.endswith((".html", ".htm"))
        if self.detect_format and is_html:
            return _with_section_paths(HTMLSectionChunker(chunk_size=10**9, min_chunk_size=1).split_html(text))
        if self.detect_format and ("# " in text or re.search(r"^#{1,6}\s+", text, flags=re.M) or source.endswith((".md", ".markdown"))):
            return _with_section_paths(MarkdownHeaderChunker(chunk_size=10**9, min_chunk_size=1).split_text(text))
        return [{"title": "document", "level": 0, "path": "document", "content": text}]


def _adjacent_cosine_similarities(embeddings):
    np = _numpy()
    embeddings = np.asarray(embeddings, dtype=float)
    if len(embeddings) < 2:
        return np.array([], dtype=float)
    left = embeddings[:-1]
    right = embeddings[1:]
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = left_norm * right_norm
    sims = np.divide((left * right).sum(axis=1), denom, out=np.zeros_like(denom, dtype=float), where=denom > 1e-12)
    return np.clip(sims, -1.0, 1.0)


def _safe_mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _detect_language(doc: Document) -> str:
    language = doc.metadata.get("language") or doc.metadata.get("code_language")
    if language:
        return str(language).lower()
    source = str(doc.metadata.get("source") or doc.metadata.get("filename") or "").lower()
    suffix_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".sql": "sql",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
    }
    for suffix, lang in suffix_map.items():
        if source.endswith(suffix):
            return lang
    return "text"


def _split_python_code(text: str) -> list[dict[str, object]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    starts: list[tuple[int, str | None]] = []
    for idx, line in enumerate(lines):
        match = re.match(r"^(?:async\s+def|def|class)\s+([A-Za-z_]\w*)", line)
        if match:
            start = idx
            while start > 0 and lines[start - 1].lstrip().startswith("@"):
                start -= 1
            starts.append((start, match.group(1)))
    return _blocks_from_starts(lines, starts)


def _split_js_code(text: str) -> list[dict[str, object]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    pattern = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function\s+([A-Za-z_$][\w$]*)|class\s+([A-Za-z_$][\w$]*)|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(?[^=]*=>)")
    starts: list[tuple[int, str | None]] = []
    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            starts.append((idx, next((g for g in match.groups() if g), None)))
    return _blocks_from_starts(lines, starts)


def _split_sql_code(text: str) -> list[dict[str, object]]:
    statements = [s.strip() for s in re.split(r";\s*(?:\n|$)", text) if s.strip()]
    blocks = []
    current_line = 1
    for statement in statements:
        line_count = statement.count("\n") + 1
        symbol = _first_sql_keyword(statement)
        blocks.append({"content": statement + ";", "symbol_name": symbol, "start_line": current_line, "end_line": current_line + line_count - 1})
        current_line += line_count
    return blocks or _split_generic_code(text)


def _split_generic_code(text: str) -> list[dict[str, object]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[dict[str, object]] = []
    current: list[str] = []
    start_line = 1
    for idx, line in enumerate(lines, start=1):
        if current and not line.strip():
            content = "\n".join(current).strip()
            if content:
                blocks.append({"content": content, "symbol_name": None, "start_line": start_line, "end_line": idx - 1})
            current = []
            start_line = idx + 1
        else:
            if not current:
                start_line = idx
            current.append(line)
    if current:
        content = "\n".join(current).strip()
        if content:
            blocks.append({"content": content, "symbol_name": None, "start_line": start_line, "end_line": len(lines)})
    return blocks or [{"content": text, "symbol_name": None, "start_line": 1, "end_line": len(lines)}]


def _blocks_from_starts(lines: list[str], starts: list[tuple[int, str | None]]) -> list[dict[str, object]]:
    if not starts:
        return _split_generic_code("\n".join(lines))
    blocks: list[dict[str, object]] = []
    if starts[0][0] > 0:
        preamble = "\n".join(lines[: starts[0][0]]).strip()
        if preamble:
            blocks.append({"content": preamble, "symbol_name": "module_preamble", "start_line": 1, "end_line": starts[0][0]})
    for idx, (start, symbol) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        if content:
            blocks.append({"content": content, "symbol_name": symbol, "start_line": start + 1, "end_line": end})
    return blocks


def _first_sql_keyword(statement: str) -> str | None:
    match = re.match(r"\s*(select|insert|update|delete|create|alter|drop|with|merge)\b", statement, flags=re.I)
    return match.group(1).upper() if match else None


def _parse_table_block(block: list[str], delim: str) -> dict[str, object] | None:
    if len(block) < 2:
        return None
    rows: list[list[str]] = []
    if delim == "|":
        for line in block:
            stripped = line.strip().strip("|")
            cells = [cell.strip() for cell in stripped.split("|")]
            if _is_markdown_separator_row(cells):
                continue
            rows.append(cells)
        fmt = "markdown"
    else:
        try:
            rows = [[cell.strip() for cell in row] for row in csv.reader(io.StringIO("\n".join(block)), delimiter=delim)]
        except csv.Error:
            return None
        fmt = "csv" if delim == "," else "delimited"
    rows = [row for row in rows if any(cell for cell in row)]
    if len(rows) < 2:
        return None
    common_cols = _mode([len(row) for row in rows])
    normalized = [row + [""] * (common_cols - len(row)) if len(row) < common_cols else row[:common_cols] for row in rows]
    return {"rows": normalized, "delimiter": delim, "column_count": common_cols, "format": fmt}


def _is_markdown_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells if cell.strip())


def _render_table(rows: list[list[str]], delim: str) -> str:
    if delim == "|":
        return "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(delim.join(row) for row in rows)


def _mode(values: list[int]) -> int:
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=lambda value: (counts[value], value))


def _classify_layout_block(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if "|" in stripped and stripped.count("|") >= 2:
        return "table"
    if len(stripped) < 90 and (stripped.isupper() or re.match(r"^\d+(?:\.\d+)*\s+\S+", stripped)):
        return "heading"
    if re.match(r"^(figure|table|chart|image)\s+\d+", stripped, flags=re.I):
        return "caption"
    return "paragraph"


def _merge_small_layout_blocks(blocks: list[dict[str, str]], min_chars: int) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    buffer: list[str] = []
    buffer_type = "paragraph"
    for block in blocks:
        content = block["content"]
        block_type = block["layout_block_type"]
        if len(content) < min_chars and block_type == "paragraph":
            buffer.append(content)
            continue
        if buffer:
            merged.append({"content": "\n".join(buffer), "layout_block_type": buffer_type})
            buffer = []
        merged.append(block)
    if buffer:
        merged.append({"content": "\n".join(buffer), "layout_block_type": buffer_type})
    return merged


def _with_section_paths(sections: list[dict[str, object]]) -> list[dict[str, object]]:
    stack: list[tuple[int, str]] = []
    output: list[dict[str, object]] = []
    for section in sections:
        level = int(section.get("level", 0) or 0)
        title = str(section.get("title", "document"))
        while stack and stack[-1][0] >= level and level > 0:
            stack.pop()
        if level > 0:
            stack.append((level, title))
        path = " > ".join(item[1] for item in stack) if stack else title
        output.append({**section, "path": path})
    return output
