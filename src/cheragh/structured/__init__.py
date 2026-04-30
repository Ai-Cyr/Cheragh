"""Structured-data RAG engines.

This module keeps structured RAG dependency-light by using SQLite as the common
execution layer. CSV files, in-memory records and existing SQLite databases can
all be queried through the same :class:`SQLRAGEngine` interface.
"""
from .engine import (
    SQLExecutionResult,
    SQLGenerationResult,
    SQLRAGEngine,
    StructuredRAG,
    TableSchema,
)

__all__ = [
    "SQLRAGEngine",
    "StructuredRAG",
    "TableSchema",
    "SQLGenerationResult",
    "SQLExecutionResult",
]
