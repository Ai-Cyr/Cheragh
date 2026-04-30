"""Opinionated pipeline presets."""
from __future__ import annotations

from typing import Iterable

from ..base import Document, EmbeddingModel, HashingEmbedding, LLMClient
from ..engine import RAGEngine
from ..tokenization import RetrievalTokenizer


def simple_rag(
    documents: Iterable[Document],
    embedding_model: EmbeddingModel | None = None,
    llm_client: LLMClient | None = None,
    top_k: int = 5,
) -> RAGEngine:
    """Hybrid retrieval + generation."""
    return RAGEngine.from_documents(
        documents,
        embedding_model=embedding_model or HashingEmbedding(),
        llm_client=llm_client,
        retriever_type="hybrid",
        alpha=0.45,
        top_k=top_k,
    )


def vector_rag(
    documents: Iterable[Document],
    embedding_model: EmbeddingModel | None = None,
    llm_client: LLMClient | None = None,
    top_k: int = 5,
) -> RAGEngine:
    """Dense vector retrieval + generation."""
    return RAGEngine.from_documents(
        documents,
        embedding_model=embedding_model or HashingEmbedding(),
        llm_client=llm_client,
        retriever_type="vector",
        top_k=top_k,
    )


def strict_rag(
    documents: Iterable[Document],
    embedding_model: EmbeddingModel | None = None,
    llm_client: LLMClient | None = None,
    top_k: int = 5,
    min_score: float = 0.05,
) -> RAGEngine:
    """RAG with basic no-context fallback and citation warnings."""
    return RAGEngine.from_documents(
        documents,
        embedding_model=embedding_model or HashingEmbedding(),
        llm_client=llm_client,
        retriever_type="hybrid",
        top_k=top_k,
        strict_grounding=True,
        min_score=min_score,
    )


def production_hybrid_rag(
    documents: Iterable[Document],
    embedding_model: EmbeddingModel,
    llm_client: LLMClient | None = None,
    top_k: int = 6,
    first_stage_top_k: int = 40,
    filters: dict | None = None,
) -> RAGEngine:
    """Production-oriented hybrid preset.

    Uses stronger lexical tokenization, hybrid retrieval, reranking, compression,
    strict grounding and tracing. Supply a real embedding model for production;
    the function intentionally requires one instead of silently using hashing.
    """

    return RAGEngine.from_documents(
        documents,
        embedding_model=embedding_model,
        llm_client=llm_client,
        retriever_type="hybrid",
        alpha=0.55,
        top_k=top_k,
        tokenizer=RetrievalTokenizer(ngram_range=(1, 2)),
        filters=filters,
        reranker="keyword",
        first_stage_top_k=max(first_stage_top_k, top_k),
        compressor="default",
        query_transformer="multi-query",
        strict_grounding=True,
        require_citations=True,
        flag_unsourced_sentences=True,
        trace_enabled=True,
        min_score=0.03,
    )


__all__ = ["simple_rag", "vector_rag", "strict_rag", "production_hybrid_rag"]
