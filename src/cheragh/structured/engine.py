"""SQL and structured-data RAG.

The implementation intentionally avoids mandatory external dependencies. It can
query SQLite directly, or materialize Python records / CSV files into an in-memory
SQLite database. LLM-based SQL generation is supported, but a deterministic
rule-based generator is included for tests, demos and constrained environments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from ..base import Document, ExtractiveLLMClient, LLMClient
from ..tracing import RAGTrace


@dataclass
class Source:
    """Source returned with a structured RAG answer."""

    doc_id: str | None
    score: float | None
    preview: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Lightweight response compatible with the high-level RAG response shape."""

    query: str
    answer: str
    sources: list[Source]
    retrieved_documents: list[Document]
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    grounded_score: float = 1.0
    trace: RAGTrace | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "query": self.query,
            "answer": self.answer,
            "sources": [source.__dict__ for source in self.sources],
            "metadata": self.metadata,
            "warnings": self.warnings,
            "grounded_score": self.grounded_score,
        }
        if self.trace is not None:
            data["trace"] = self.trace.to_dict(include_prompt=False)
        return data


@dataclass
class TableSchema:
    """Schema information for a SQL table."""

    name: str
    columns: list[str]
    column_types: dict[str, str] = field(default_factory=dict)
    row_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": self.columns,
            "column_types": self.column_types,
            "row_count": self.row_count,
        }


@dataclass
class SQLGenerationResult:
    """Generated SQL plus metadata about how it was produced."""

    query: str
    sql: str
    method: str = "rule_based"
    confidence: float = 0.5
    warnings: list[str] = field(default_factory=list)


@dataclass
class SQLExecutionResult:
    """Result of a read-only SQL execution."""

    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False

    def to_markdown(self, max_rows: int = 20) -> str:
        rows = self.rows[:max_rows]
        if not self.columns:
            return ""
        header = "| " + " | ".join(self.columns) + " |"
        sep = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = ["| " + " | ".join(_cell(row.get(col)) for col in self.columns) + " |" for row in rows]
        suffix = [f"\nRésultat tronqué à {max_rows} lignes."] if self.truncated or self.row_count > len(rows) else []
        return "\n".join([header, sep, *body, *suffix])


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


class SQLRAGEngine:
    """Question-answering engine over a read-only SQL database.

    Parameters
    ----------
    connection:
        Existing SQLite connection. If omitted, ``database`` is opened.
    database:
        Path to a SQLite database. Use ``":memory:"`` for an in-memory database.
    llm_client:
        Optional LLM used to generate and/or synthesize. Without it, the engine
        uses deterministic SQL heuristics and a compact tabular answer.
    table_allowlist:
        Optional list of tables the engine may query.
    max_rows:
        Maximum rows returned by generated SQL.
    """

    def __init__(
        self,
        connection: sqlite3.Connection | None = None,
        database: str | Path | None = None,
        llm_client: LLMClient | None = None,
        table_allowlist: Sequence[str] | None = None,
        max_rows: int = 50,
        read_only: bool = True,
        trace_enabled: bool = True,
        max_sql_steps: int = 100_000,
    ):
        if connection is None:
            db = ":memory:" if database is None else str(database)
            if read_only and db != ":memory:":
                db_path = Path(db).expanduser().resolve()
                connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            else:
                connection = sqlite3.connect(db)
        connection.row_factory = sqlite3.Row
        self.connection = connection
        self.llm_client = llm_client or ExtractiveLLMClient()
        self.table_allowlist = set(table_allowlist or [])
        self.max_rows = max_rows
        self.read_only = read_only
        self.trace_enabled = trace_enabled
        self.max_sql_steps = max(1_000, int(max_sql_steps))
        if self.read_only:
            self._enable_sqlite_query_only()

    @classmethod
    def from_records(
        cls,
        table_name: str,
        records: Iterable[Mapping[str, Any]],
        llm_client: LLMClient | None = None,
        max_rows: int = 50,
    ) -> "SQLRAGEngine":
        engine = cls(
            database=":memory:",
            llm_client=llm_client,
            table_allowlist=[table_name],
            max_rows=max_rows,
            read_only=False,
        )
        engine.add_table(table_name, records)
        engine.enable_read_only()
        return engine

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        table_name: str | None = None,
        llm_client: LLMClient | None = None,
        max_rows: int = 50,
        encoding: str = "utf-8",
    ) -> "SQLRAGEngine":
        p = Path(path)
        name = table_name or _safe_identifier(p.stem)
        with p.open("r", encoding=encoding, newline="") as f:
            records = list(csv.DictReader(f))
        return cls.from_records(name, records, llm_client=llm_client, max_rows=max_rows)

    def add_table(self, table_name: str, records: Iterable[Mapping[str, Any]]) -> None:
        if self.read_only:
            raise RuntimeError("Cannot add tables after SQLRAGEngine has been made read-only")
        rows = [dict(record) for record in records]
        safe_table = _safe_identifier(table_name)
        if not rows:
            self.connection.execute(f'CREATE TABLE IF NOT EXISTS "{safe_table}" (id INTEGER)')
            self.connection.commit()
            self.table_allowlist.add(safe_table)
            return
        columns = list(dict.fromkeys(key for row in rows for key in row.keys()))
        safe_columns = [_safe_identifier(col) for col in columns]
        col_defs = ", ".join(
            f'"{col}" {_infer_sql_type([row.get(src) for row in rows])}'
            for src, col in zip(columns, safe_columns)
        )
        self.connection.execute(f'DROP TABLE IF EXISTS "{safe_table}"')
        self.connection.execute(f'CREATE TABLE "{safe_table}" ({col_defs})')
        placeholders = ", ".join("?" for _ in safe_columns)
        col_list = ", ".join(f'"{col}"' for col in safe_columns)
        for row in rows:
            self.connection.execute(
                f'INSERT INTO "{safe_table}" ({col_list}) VALUES ({placeholders})',
                [_coerce_value(row.get(col)) for col in columns],
            )
        self.connection.commit()
        self.table_allowlist.add(safe_table)

    def schema(self) -> list[TableSchema]:
        tables = []
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        for row in self.connection.execute(query):
            name = row[0]
            if self.table_allowlist and name not in self.table_allowlist:
                continue
            pragma_rows = list(self.connection.execute(f'PRAGMA table_info("{name}")'))
            columns = [item[1] for item in pragma_rows]
            types = {item[1]: item[2] for item in pragma_rows}
            count = self.connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            tables.append(TableSchema(name=name, columns=columns, column_types=types, row_count=int(count)))
        return tables

    def generate_sql(self, question: str, use_llm: bool = False) -> SQLGenerationResult:
        warnings: list[str] = []
        if use_llm:
            prompt = self._sql_prompt(question)
            raw = self.llm_client.generate(prompt)
            sql = _extract_sql(raw)
            try:
                sql = self.validate_sql(sql)
                return SQLGenerationResult(question, sql, method="llm", confidence=0.65)
            except ValueError as exc:
                warnings.append(f"llm_sql_rejected:{exc}")
        sql, confidence = self._rule_based_sql(question)
        return SQLGenerationResult(question, sql, method="rule_based", confidence=confidence, warnings=warnings)

    def validate_sql(self, sql: str) -> str:
        cleaned = sql.strip().rstrip(";")
        if not cleaned:
            raise ValueError("empty SQL")
        if self.read_only and not re.match(r"^\s*(select|with)\b", cleaned, flags=re.I):
            raise ValueError("only SELECT/WITH statements are allowed")
        forbidden = re.compile(
            r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum)\b",
            re.I,
        )
        if forbidden.search(cleaned):
            raise ValueError("write or administrative SQL is not allowed")
        if ";" in cleaned:
            raise ValueError("multiple SQL statements are not allowed")
        if self.table_allowlist:
            referenced = _referenced_tables(cleaned)
            unknown = [name for name in referenced if name not in self.table_allowlist]
            if unknown:
                raise ValueError(f"SQL references unavailable tables: {unknown}")
        if not re.search(r"\blimit\b", cleaned, flags=re.I):
            cleaned = f"{cleaned} LIMIT {self.max_rows}"
        return cleaned

    def execute_sql(self, sql: str) -> SQLExecutionResult:
        valid_sql = self.validate_sql(sql)
        cursor = self._execute_readonly(valid_sql)
        rows_raw = cursor.fetchmany(self.max_rows + 1)
        columns = [desc[0] for desc in cursor.description or []]
        truncated = len(rows_raw) > self.max_rows
        rows = [dict(row) for row in rows_raw[: self.max_rows]]
        return SQLExecutionResult(sql=valid_sql, columns=columns, rows=rows, row_count=len(rows), truncated=truncated)

    def enable_read_only(self) -> None:
        """Permanently switch this engine to SQLite read-only/query-only mode."""

        self.read_only = True
        self._enable_sqlite_query_only()

    def _enable_sqlite_query_only(self) -> None:
        try:
            self.connection.execute("PRAGMA query_only = ON")
        except sqlite3.DatabaseError:
            # Connections opened with mode=ro may reject pragmas depending on the
            # driver/build. The read-only URI and authorizer still enforce safety.
            pass

    def _execute_readonly(self, sql: str):
        if not self.read_only:
            return self.connection.execute(sql)
        self._enable_sqlite_query_only()
        self.connection.set_authorizer(_sqlite_readonly_authorizer)
        steps = {"count": 0}

        def progress_handler() -> int:
            steps["count"] += 1
            return 1 if steps["count"] > self.max_sql_steps else 0

        self.connection.set_progress_handler(progress_handler, 1000)
        try:
            return self.connection.execute(sql)
        finally:
            self.connection.set_progress_handler(None, 0)
            self.connection.set_authorizer(None)

    def ask(self, question: str, use_llm_sql: bool = False, synthesize: bool = True, **kwargs: Any) -> RAGResponse:
        trace = RAGTrace() if self.trace_enabled else None
        step = trace.start_step("sql_generation") if trace else None
        generated = self.generate_sql(question, use_llm=use_llm_sql)
        if step:
            step.finish(method=generated.method, confidence=generated.confidence)
        exec_step = trace.start_step("sql_execution", sql=generated.sql) if trace else None
        result = self.execute_sql(generated.sql)
        if exec_step:
            exec_step.finish(row_count=result.row_count, truncated=result.truncated)
        answer = self._synthesize_answer(question, generated, result) if synthesize else result.to_markdown()
        doc = Document(
            content=result.to_markdown(),
            metadata={"sql": result.sql, "row_count": result.row_count, "structured": True},
            doc_id="sql-result",
            score=1.0,
        )
        return RAGResponse(
            query=question,
            answer=answer,
            sources=[Source(doc.doc_id, doc.score, doc.content[:240], dict(doc.metadata))],
            retrieved_documents=[doc],
            prompt=self._sql_prompt(question),
            metadata={
                "architecture": "sql_rag",
                "sql": result.sql,
                "sql_generation_method": generated.method,
                "sql_confidence": generated.confidence,
                "schema": [table.to_dict() for table in self.schema()],
                "execution": {"row_count": result.row_count, "truncated": result.truncated},
            },
            warnings=generated.warnings,
            grounded_score=1.0,
            trace=trace,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        response = self.ask(query, synthesize=False)
        return response.retrieved_documents[:top_k]

    def _rule_based_sql(self, question: str) -> tuple[str, float]:
        schemas = self.schema()
        if not schemas:
            raise ValueError("No tables available")
        q = question.lower()
        table = _best_table(q, schemas)
        columns = table.columns
        numeric_cols = [col for col in columns if _is_numeric_type(table.column_types.get(col, ""))]
        selected_col = _best_column(q, columns) or (numeric_cols[0] if numeric_cols else columns[0])
        where = _simple_where_clause(question, table)
        if any(word in q for word in ["combien", "nombre", "count", "how many"]):
            return f'SELECT COUNT(*) AS count FROM "{table.name}"{where}', 0.75
        if any(word in q for word in ["total", "somme", "sum", "revenu total", "montant total"]):
            col = _best_numeric_column(q, table) or (numeric_cols[0] if numeric_cols else selected_col)
            return f'SELECT SUM("{col}") AS total_{col} FROM "{table.name}"{where}', 0.72
        if any(word in q for word in ["moyenne", "average", "avg"]):
            col = _best_numeric_column(q, table) or (numeric_cols[0] if numeric_cols else selected_col)
            return f'SELECT AVG("{col}") AS average_{col} FROM "{table.name}"{where}', 0.72
        if any(word in q for word in ["max", "maximum", "plus grand", "plus gros", "le plus", "highest", "top"]):
            col = _best_numeric_column(q, table) or (numeric_cols[0] if numeric_cols else selected_col)
            return f'SELECT * FROM "{table.name}"{where} ORDER BY "{col}" DESC', 0.68
        if any(word in q for word in ["min", "minimum", "plus petit", "lowest"]):
            col = _best_numeric_column(q, table) or (numeric_cols[0] if numeric_cols else selected_col)
            return f'SELECT * FROM "{table.name}"{where} ORDER BY "{col}" ASC', 0.68
        return f'SELECT * FROM "{table.name}"{where}', 0.55

    def _sql_prompt(self, question: str) -> str:
        schema_lines = []
        for table in self.schema():
            cols = ", ".join(f"{col} {table.column_types.get(col, '')}".strip() for col in table.columns)
            schema_lines.append(f"- {table.name}({cols})")
        return (
            "Tu génères uniquement une requête SQL SQLite en lecture seule.\n"
            "Schéma disponible:\n"
            + "\n".join(schema_lines)
            + f"\nQuestion: {question}\nSQL:"
        )

    def _synthesize_answer(self, question: str, generation: SQLGenerationResult, result: SQLExecutionResult) -> str:
        table = result.to_markdown(max_rows=min(10, self.max_rows))
        if not result.rows:
            return f"Aucun résultat trouvé.\n\nSQL exécuté: `{result.sql}`"
        # If using the default extractive client, avoid a verbose prompt echo and return the table directly.
        if isinstance(self.llm_client, ExtractiveLLMClient):
            return f"Voici le résultat structuré.\n\n{table}\n\nSQL exécuté: `{result.sql}`"
        prompt = (
            "Réponds en français à partir du résultat SQL ci-dessous. "
            "Ne mentionne pas d'information absente du tableau.\n\n"
            f"Question: {question}\nSQL: {generation.sql}\nRésultat:\n{table}\n\nRéponse:"
        )
        return self.llm_client.generate(prompt)


class StructuredRAG:
    """Facade for RAG over one or more structured tables.

    It materializes Python records or CSV files into SQLite and delegates query
    answering to :class:`SQLRAGEngine`.
    """

    def __init__(self, sql_engine: SQLRAGEngine):
        self.sql_engine = sql_engine

    @classmethod
    def from_tables(
        cls,
        tables: Mapping[str, Iterable[Mapping[str, Any]]],
        llm_client: LLMClient | None = None,
        max_rows: int = 50,
    ) -> "StructuredRAG":
        engine = SQLRAGEngine(database=":memory:", llm_client=llm_client, max_rows=max_rows, read_only=False)
        for name, records in tables.items():
            engine.add_table(name, records)
        engine.enable_read_only()
        return cls(engine)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        table_name: str = "data",
        llm_client: LLMClient | None = None,
        max_rows: int = 50,
    ) -> "StructuredRAG":
        return cls(SQLRAGEngine.from_records(table_name, records, llm_client=llm_client, max_rows=max_rows))

    @classmethod
    def from_csv(cls, path: str | Path, table_name: str | None = None, **kwargs: Any) -> "StructuredRAG":
        return cls(SQLRAGEngine.from_csv(path, table_name=table_name, **kwargs))

    @classmethod
    def from_sqlite(cls, database: str | Path, **kwargs: Any) -> "StructuredRAG":
        return cls(SQLRAGEngine(database=database, **kwargs))

    def ask(self, question: str, **kwargs: Any) -> RAGResponse:
        response = self.sql_engine.ask(question, **kwargs)
        response.metadata["architecture"] = "structured_rag"
        return response

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.sql_engine.retrieve(query, top_k=top_k)

    def schema(self) -> list[TableSchema]:
        return self.sql_engine.schema()


def _sqlite_readonly_authorizer(action, arg1, arg2, dbname, source):
    denied = {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_PRAGMA,
        sqlite3.SQLITE_TRANSACTION,
    }
    if action in denied:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _safe_identifier(name: str) -> str:
    cleaned = re.sub(r"\W+", "_", str(name).strip()).strip("_")
    if not cleaned:
        cleaned = "data"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned.lower()


def _coerce_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _infer_sql_type(values: Sequence[Any]) -> str:
    non_empty = [v for v in values if v not in (None, "")]
    if not non_empty:
        return "TEXT"
    try:
        for v in non_empty:
            int(v)
        return "INTEGER"
    except (TypeError, ValueError):
        pass
    try:
        for v in non_empty:
            float(v)
        return "REAL"
    except (TypeError, ValueError):
        return "TEXT"


def _is_numeric_type(sql_type: str) -> bool:
    return any(part in sql_type.upper() for part in ["INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE"])


def _extract_sql(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    for line in text.splitlines():
        if re.match(r"\s*(select|with)\b", line, flags=re.I):
            return line.strip()
    return text.strip()


def _referenced_tables(sql: str) -> list[str]:
    names: list[str] = []
    pattern = r"\b(?:from|join)\s+(?:[\"`\[]?)([A-Za-z_][\w]*)"
    for match in re.finditer(pattern, sql, flags=re.I):
        names.append(match.group(1))
    return names


def _best_table(question_lower: str, schemas: list[TableSchema]) -> TableSchema:
    def score(table: TableSchema) -> int:
        tokens = {table.name.lower(), *[col.lower() for col in table.columns]}
        return sum(1 for token in tokens if token and token in question_lower)

    return max(schemas, key=score)


def _best_column(question_lower: str, columns: list[str]) -> str | None:
    scored = [(col, 1 if col.lower() in question_lower else 0) for col in columns]
    best = max(scored, key=lambda item: item[1], default=(None, 0))
    return best[0] if best[1] > 0 else None


def _best_numeric_column(question_lower: str, table: TableSchema) -> str | None:
    aliases = {
        "revenu": ["revenue", "revenu", "amount", "montant", "sales", "vente", "ventes", "ca"],
        "prix": ["price", "prix", "cost", "cout", "coût"],
        "quantite": ["quantity", "quantite", "quantité", "qty", "volume"],
        "score": ["score", "rating", "note"],
    }
    numeric = [col for col in table.columns if _is_numeric_type(table.column_types.get(col, ""))]
    for col in numeric:
        col_lower = col.lower()
        if col_lower in question_lower:
            return col
        for words in aliases.values():
            if col_lower in words and any(word in question_lower for word in words):
                return col
    return None


def _simple_where_clause(question: str, table: TableSchema) -> str:
    # conservative equality filters for quoted literals or exact value mentions in text columns
    q = question.lower()
    clauses: list[str] = []
    for col in table.columns:
        col_lower = col.lower()
        if col_lower not in q:
            continue
        match = re.search(rf"{re.escape(col_lower)}\s*(?:=|est|equals|vaut)\s*['\"]?([\wÀ-ÿ .-]+)['\"]?", q)
        if match:
            value = match.group(1).strip().strip(".,;:!?'")
            clauses.append(f'"{col}" = \'{value.replace("'", "''")}\'')
    return " WHERE " + " AND ".join(clauses) if clauses else ""
