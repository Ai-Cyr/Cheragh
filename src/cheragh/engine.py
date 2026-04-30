"""High-level RAG engine API."""
from __future__ import annotations

from pathlib import Path
import asyncio
from typing import Any, Iterable, Iterator

from .base import BaseRetriever, Document, EmbeddingModel, ExtractiveLLMClient, HashingEmbedding, LLMClient, OpenAILLMClient
from .citations import citation_location, extract_citations, validate_citations
from .hybrid_search import HybridSearchRetriever
from .schema import RAGResponse, Source
from .ingestion import chunk_documents, ingest_path
from .pipeline import DEFAULT_ANSWER_PROMPT_FR, AdvancedRAGPipeline
from .reranking import BaseReranker, RerankingRetriever, build_reranker
from .compression import ContextCompressor, CompressionPipeline, ExtractiveContextCompressor, RedundancyFilter
from .query import QueryTransformer, build_query_transformer
from .tracing import RAGTrace, append_trace_jsonl
from .cache import (
    CacheBackend,
    CachedEmbeddingModel,
    CachedLLMClient,
    CachedReranker,
    CachedRetriever,
    build_cache_backend,
)


class RAGEngine:
    """Simple façade for common RAG use cases.

    The engine wraps a retriever and an LLM client, handles prompt formatting,
    returns structured sources, and can enforce basic grounding rules.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_client: LLMClient | None = None,
        answer_prompt: str = DEFAULT_ANSWER_PROMPT_FR,
        top_k: int = 5,
        strict_grounding: bool = False,
        min_score: float | None = None,
        require_citations: bool | None = None,
        flag_unsourced_sentences: bool = False,
        compressor: ContextCompressor | str | None = None,
        query_transformer: QueryTransformer | str | None = None,
        trace_enabled: bool = True,
        cache_backend: CacheBackend | None = None,
        cache_config: dict[str, Any] | None = None,
        trace_export_path: str | Path | None = None,
        trace_include_prompt: bool = False,
        trace_pricing: dict[str, Any] | None = None,
    ):
        self.retriever = retriever
        self.llm_client = llm_client or ExtractiveLLMClient()
        self.answer_prompt = answer_prompt
        self.top_k = top_k
        self.strict_grounding = strict_grounding
        self.min_score = min_score
        self.require_citations = strict_grounding if require_citations is None else require_citations
        self.flag_unsourced_sentences = flag_unsourced_sentences
        self.compressor = _build_compressor(compressor)
        self.query_transformer = build_query_transformer(query_transformer) if isinstance(query_transformer, str) or query_transformer is None else query_transformer
        self.trace_enabled = trace_enabled
        self.cache_backend = cache_backend
        self.cache_config = cache_config or {}
        self.trace_export_path = Path(trace_export_path) if trace_export_path else None
        self.trace_include_prompt = trace_include_prompt
        self.trace_pricing = trace_pricing or {}

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        embedding_model: EmbeddingModel | None = None,
        llm_client: LLMClient | None = None,
        retriever_type: str = "hybrid",
        alpha: float = 0.5,
        top_k: int = 5,
        chunk: bool = False,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        vectorstore_path: str | Path | None = None,
        reranker: BaseReranker | str | None = None,
        reranker_model: str | None = None,
        first_stage_top_k: int = 30,
        compressor: ContextCompressor | str | None = None,
        query_transformer: QueryTransformer | str | None = None,
        **kwargs: Any,
    ) -> "RAGEngine":
        cache_backend = kwargs.get("cache_backend") or build_cache_backend(_cache_config_from_kwargs(kwargs))
        cache_cfg = _normalize_cache_config(kwargs.get("cache_config") or _cache_config_from_kwargs(kwargs))
        embedder = embedding_model or HashingEmbedding()
        if cache_backend is not None and cache_cfg.get("cache_embeddings", True):
            embedder = CachedEmbeddingModel(embedder, cache_backend, ttl=cache_cfg.get("ttl"), namespace=cache_cfg.get("embedding_namespace", "embeddings"))
        docs = list(documents)
        if chunk:
            docs = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        retriever = _build_retriever(
            docs=docs,
            embedder=embedder,
            retriever_type=retriever_type,
            alpha=alpha,
            vectorstore_path=vectorstore_path,
            kwargs=kwargs,
        )
        if reranker:
            reranker_obj = build_reranker(provider=reranker, model=reranker_model) if isinstance(reranker, str) else reranker
            if cache_backend is not None and cache_cfg.get("cache_reranking", True):
                reranker_obj = CachedReranker(reranker_obj, cache_backend, ttl=cache_cfg.get("ttl"), namespace=cache_cfg.get("reranking_namespace", "reranking"))
            retriever = RerankingRetriever(
                retriever,
                first_stage_top_k=max(first_stage_top_k, top_k),
                reranker=reranker_obj,
            )
        if cache_backend is not None and cache_cfg.get("cache_retrieval", True):
            retriever = CachedRetriever(retriever, cache_backend, ttl=cache_cfg.get("ttl"), namespace=cache_cfg.get("retrieval_namespace", "retrieval"))
        effective_llm = llm_client
        if cache_backend is not None and effective_llm is not None and cache_cfg.get("cache_llm", True):
            effective_llm = CachedLLMClient(effective_llm, cache_backend, ttl=cache_cfg.get("ttl"), namespace=cache_cfg.get("llm_namespace", "llm"))
        return cls(
            retriever=retriever,
            llm_client=effective_llm,
            top_k=top_k,
            compressor=compressor or kwargs.get("compressor"),
            query_transformer=query_transformer or kwargs.get("query_transformer"),
            cache_backend=cache_backend,
            cache_config=cache_cfg,
            trace_export_path=kwargs.get("trace_export_path"),
            trace_include_prompt=bool(kwargs.get("trace_include_prompt", False)),
            trace_pricing=kwargs.get("trace_pricing"),
            **_engine_kwargs(kwargs),
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        embedding_model: EmbeddingModel | None = None,
        llm_client: LLMClient | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        **kwargs: Any,
    ) -> "RAGEngine":
        chunks = ingest_path(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return cls.from_documents(chunks, embedding_model=embedding_model, llm_client=llm_client, **kwargs)

    @classmethod
    def from_config(cls, config_path: str | Path, documents: Iterable[Document] | None = None) -> "RAGEngine":
        from .config import load_config

        config = load_config(config_path)
        embedder = _embedding_from_config(config.get("embedding", {}))
        llm = _llm_from_config(config.get("generation", {}))
        retriever_cfg = config.get("retriever", {})
        ingestion_cfg = config.get("ingestion", {})
        vectorstore_cfg = config.get("vectorstore", {})
        reranker_cfg = config.get("reranker", {})
        compression_cfg = config.get("compression", {})
        query_cfg = config.get("query", {})
        observability_cfg = config.get("observability", {}) or {}
        cache_cfg = _normalize_cache_config(config.get("cache", {
            "enabled": bool(config.get("cache_backend")),
            "backend": config.get("cache_backend"),
            "path": config.get("cache_path"),
            "ttl": config.get("cache_ttl"),
        }))
        cache_backend = build_cache_backend(cache_cfg) if cache_cfg.get("enabled", False) else None

        if documents is None:
            source_path = ingestion_cfg.get("path")
            if not source_path:
                raise ValueError("Config must define ingestion.path when documents are not provided")
            docs = ingest_path(
                source_path,
                chunk_size=int(ingestion_cfg.get("chunk_size", 800)),
                chunk_overlap=int(ingestion_cfg.get("chunk_overlap", 120)),
                recursive=bool(ingestion_cfg.get("recursive", True)),
                exclude_patterns=ingestion_cfg.get("exclude_patterns") or None,
                max_file_size_mb=ingestion_cfg.get("max_file_size_mb", 50),
            )
        else:
            docs = list(documents)

        reranker: str | None = None
        reranker_model: str | None = None
        if bool(reranker_cfg.get("enabled", False)):
            reranker = str(reranker_cfg.get("provider", "cross-encoder"))
            reranker_model = reranker_cfg.get("model")

        return cls.from_documents(
            docs,
            embedding_model=embedder,
            llm_client=llm,
            retriever_type=str(retriever_cfg.get("type", vectorstore_cfg.get("type", "hybrid"))),
            alpha=float(retriever_cfg.get("alpha", 0.5)),
            top_k=int(retriever_cfg.get("top_k", 5)),
            strict_grounding=bool(config.get("strict_grounding", False)),
            min_score=config.get("min_score"),
            require_citations=config.get("require_citations"),
            flag_unsourced_sentences=bool(config.get("flag_unsourced_sentences", False)),
            vectorstore_path=vectorstore_cfg.get("path"),
            collection_name=vectorstore_cfg.get("collection_name", "cheragh"),
            qdrant_url=vectorstore_cfg.get("url"),
            qdrant_api_key=vectorstore_cfg.get("api_key"),
            filters=retriever_cfg.get("filters") or None,
            tokenizer_config=retriever_cfg.get("tokenizer") or None,
            reranker=reranker,
            reranker_model=reranker_model,
            first_stage_top_k=int(reranker_cfg.get("first_stage_top_k", 30)),
            compressor=str(compression_cfg.get("type", "default")) if bool(compression_cfg.get("enabled", False)) else None,
            query_transformer=str(query_cfg.get("transform", query_cfg.get("type", "multi-query"))) if bool(query_cfg.get("enabled", False)) else None,
            trace_enabled=bool(config.get("trace_enabled", True)),
            cache_backend=cache_backend,
            cache_config=cache_cfg,
            trace_export_path=observability_cfg.get("trace_export_path"),
            trace_include_prompt=bool(observability_cfg.get("trace_include_prompt", False)),
            trace_pricing=observability_cfg.get("pricing"),
        )

    def ask(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> RAGResponse:
        trace = RAGTrace(query=query) if self.trace_enabled else None
        effective_top_k = top_k or self.top_k

        query_variants = self._query_variants(query, trace)
        step = trace.start_step("retrieval", query_count=len(query_variants), top_k=effective_top_k) if trace else None
        docs = self._retrieve_variants(query_variants, effective_top_k=effective_top_k, trace=trace)
        if step:
            step.finish(document_count=len(docs))

        warnings: list[str] = []
        if self.min_score is not None:
            before = len(docs)
            docs = [doc for doc in docs if doc.score is None or doc.score >= self.min_score]
            if trace:
                trace.warnings.append(f"min_score_filtered:{before - len(docs)}")

        if self.compressor is not None and docs:
            step = trace.start_step("compression", compressor=self.compressor.__class__.__name__) if trace else None
            before_chars = sum(len(doc.content) for doc in docs)
            docs = self.compressor.compress(query, docs)
            after_chars = sum(len(doc.content) for doc in docs)
            if trace:
                trace.compression = {"before_chars": before_chars, "after_chars": after_chars, "document_count": len(docs)}
            if step:
                step.finish(before_chars=before_chars, after_chars=after_chars, document_count=len(docs))

        if self.strict_grounding and not docs:
            return RAGResponse(
                query=query,
                answer="Je ne sais pas : aucun extrait suffisamment pertinent n'a été trouvé.",
                sources=[],
                retrieved_documents=[],
                prompt="",
                metadata={"strict_grounding": True},
                warnings=["no_relevant_documents"],
                grounded_score=0.0,
                trace=self._finalize_trace(trace, answer="", prompt=""),
            )

        context = AdvancedRAGPipeline._format_context(docs)
        prompt = self.answer_prompt.format(context=context, query=query)
        if trace:
            trace.prompt = prompt
        step = trace.start_step("generation", prompt_chars=len(prompt)) if trace else None
        answer = self.llm_client.generate(prompt, **generate_kwargs)
        if step:
            step.finish(answer_chars=len(answer))
        if trace:
            trace.record_generation(prompt=prompt, answer=answer, model=getattr(self.llm_client, "model", None), pricing=self.trace_pricing)
        validation = validate_citations(
            answer,
            docs,
            require_citations=self.require_citations,
            flag_unsourced_sentences=self.flag_unsourced_sentences,
        )
        warnings.extend(validation.warnings)
        if self.strict_grounding and validation.unknown_citations:
            warnings.append("strict_grounding_unknown_citations")
        if trace:
            trace.warnings.extend(warnings)

        return RAGResponse(
            query=query,
            answer=answer,
            sources=[Source(doc.doc_id, doc.score, doc.content[:240], dict(doc.metadata), citation_location(doc)) for doc in docs],
            retrieved_documents=docs,
            prompt=prompt,
            metadata={"top_k": effective_top_k, "strict_grounding": self.strict_grounding, "cache": self.cache_backend.stats().to_dict() if self.cache_backend else None},
            citations=validation.citations,
            warnings=warnings,
            grounded_score=validation.grounded_score,
            unsourced_claims=validation.unsourced_claims,
            citation_validation=validation,
            trace=self._finalize_trace(trace, answer=answer, prompt=prompt),
        )

    def _finalize_trace(self, trace: RAGTrace | None, *, answer: str, prompt: str) -> RAGTrace | None:
        if trace is None:
            return None
        trace.finish(answer_chars=len(answer), prompt_chars=len(prompt))
        if self.trace_export_path is not None:
            append_trace_jsonl(self.trace_export_path, trace, include_prompt=self.trace_include_prompt)
        return trace

    async def aask(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> RAGResponse:
        """Async wrapper for frameworks that need awaitable execution."""
        return await asyncio.to_thread(self.ask, query, top_k, **generate_kwargs)

    def stream(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> Iterator[str]:
        effective_top_k = top_k or self.top_k
        docs = self._retrieve_variants(self._query_variants(query, None), effective_top_k=effective_top_k, trace=None)
        if self.compressor is not None and docs:
            docs = self.compressor.compress(query, docs)
        context = AdvancedRAGPipeline._format_context(docs)
        prompt = self.answer_prompt.format(context=context, query=query)
        yield from self.llm_client.stream(prompt, **generate_kwargs)

    async def astream(self, query: str, top_k: int | None = None, **generate_kwargs: Any):
        """Async streaming wrapper. Yields chunks from the synchronous stream."""
        for chunk in self.stream(query, top_k=top_k, **generate_kwargs):
            yield chunk

    def _query_variants(self, query: str, trace: RAGTrace | None) -> list[str]:
        if self.query_transformer is None:
            variants = [query]
        else:
            step = trace.start_step("query_transform", transformer=self.query_transformer.__class__.__name__) if trace else None
            variants = self.query_transformer.transform(query)
            if step:
                step.finish(query_count=len(variants))
        if trace:
            trace.query_variants = variants
        return variants

    def _retrieve_variants(self, queries: list[str], effective_top_k: int, trace: RAGTrace | None) -> list[Document]:
        merged: dict[str, Document] = {}
        for variant in queries:
            docs = self.retriever.retrieve(variant, top_k=effective_top_k)
            if trace:
                trace.add_retrieval(variant, docs)
            for doc in docs:
                key = doc.doc_id or doc.content
                previous = merged.get(key)
                if previous is None or ((doc.score or 0.0) > (previous.score or 0.0)):
                    merged[key] = doc
        ordered = sorted(merged.values(), key=lambda doc: (doc.score is not None, doc.score or 0.0), reverse=True)
        return ordered[:effective_top_k]


def _build_retriever(
    docs: list[Document],
    embedder: EmbeddingModel,
    retriever_type: str,
    alpha: float,
    vectorstore_path: str | Path | None,
    kwargs: dict[str, Any],
) -> BaseRetriever:
    rt = retriever_type.lower().replace("_", "-")
    filters = kwargs.get("filters")
    tokenizer = _tokenizer_from_config(kwargs.get("tokenizer") or kwargs.get("tokenizer_config"))
    if rt == "hybrid":
        return HybridSearchRetriever(docs, embedder, alpha=alpha, cache_path=kwargs.get("cache_path"), filters=filters, tokenizer=tokenizer)
    if rt in {"vector", "memory"}:
        from .vectorstores.memory import MemoryVectorStore

        store = MemoryVectorStore(embedder)
        store.add_documents(docs)
        if vectorstore_path:
            store.save(vectorstore_path)
        return store.as_retriever(filters=filters)
    if rt == "faiss":
        from .vectorstores.faiss import FaissVectorStore

        store = FaissVectorStore(embedder, normalize=bool(kwargs.get("normalize", True)))
        store.add_documents(docs)
        if vectorstore_path:
            store.save(vectorstore_path)
        return store.as_retriever(filters=filters)
    if rt == "chroma":
        from .vectorstores.chroma import ChromaVectorStore

        store = ChromaVectorStore(
            embedder,
            collection_name=str(kwargs.get("collection_name", "cheragh")),
            path=vectorstore_path,
        )
        store.add_documents(docs)
        return store.as_retriever(filters=filters)
    if rt == "qdrant":
        from .vectorstores.qdrant import QdrantVectorStore

        store = QdrantVectorStore(
            embedder,
            collection_name=str(kwargs.get("collection_name", "cheragh")),
            path=vectorstore_path,
            url=kwargs.get("qdrant_url"),
            api_key=kwargs.get("qdrant_api_key"),
        )
        store.add_documents(docs)
        return store.as_retriever(filters=filters)
    raise ValueError(f"Unknown retriever_type: {retriever_type}")


def _engine_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = {"answer_prompt", "strict_grounding", "min_score", "require_citations", "flag_unsourced_sentences", "trace_enabled"}
    return {key: kwargs[key] for key in allowed if key in kwargs}




def _cache_config_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "cache_config" in kwargs and isinstance(kwargs.get("cache_config"), dict):
        return dict(kwargs["cache_config"])
    cfg: dict[str, Any] = {}
    mapping = {
        "cache_backend_name": "backend",
        "cache_backend_type": "backend",
        "cache_path": "path",
        "cache_ttl": "ttl",
        "cache_namespace": "namespace",
        "cache_enabled": "enabled",
    }
    if "cache_backend" in kwargs and isinstance(kwargs.get("cache_backend"), str):
        cfg["backend"] = kwargs["cache_backend"]
    for source, target in mapping.items():
        if source in kwargs and kwargs[source] is not None:
            cfg[target] = kwargs[source]
    for key in ["cache_embeddings", "cache_retrieval", "cache_reranking", "cache_llm"]:
        if key in kwargs:
            cfg[key] = kwargs[key]
    return cfg


def _normalize_cache_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    if not cfg:
        return {"enabled": False}
    if "backend" not in cfg and "type" in cfg:
        cfg["backend"] = cfg["type"]
    if cfg.get("backend") is None and cfg.get("enabled") is True:
        cfg["backend"] = "memory"
    if isinstance(cfg.get("enabled"), str):
        cfg["enabled"] = str(cfg["enabled"]).lower() not in {"0", "false", "no", "off"}
    elif "enabled" not in cfg:
        cfg["enabled"] = bool(cfg.get("backend") or cfg.get("path") or cfg.get("ttl"))
    if cfg.get("ttl") not in {None, ""}:
        cfg["ttl"] = float(cfg["ttl"])
    for key in ["cache_embeddings", "cache_retrieval", "cache_reranking", "cache_llm"]:
        if key not in cfg:
            cfg[key] = True
    return cfg

def _build_compressor(compressor: ContextCompressor | str | None) -> ContextCompressor | None:
    if compressor is None or compressor is False:
        return None
    if isinstance(compressor, ContextCompressor):
        return compressor
    name = str(compressor).lower().replace("_", "-")
    if name in {"extractive", "sentence", "sentences"}:
        return ExtractiveContextCompressor()
    if name in {"redundancy", "dedupe", "redundancy-filter"}:
        return RedundancyFilter()
    if name in {"default", "pipeline"}:
        return CompressionPipeline([RedundancyFilter(), ExtractiveContextCompressor()])
    raise ValueError(f"Unsupported compressor: {compressor}")

def _embedding_from_config(config: dict[str, Any]) -> EmbeddingModel:
    provider = str(config.get("provider", "hashing")).lower().replace("_", "-")
    if provider in {"hashing", "local-hash"}:
        return HashingEmbedding(dimension=int(config.get("dimension", 384)))
    if provider in {"sentence-transformers", "sentence-transformer"}:
        from .base import SentenceTransformerEmbedding

        return SentenceTransformerEmbedding(model_name=str(config.get("model", "sentence-transformers/all-MiniLM-L6-v2")))
    if provider == "openai":
        from .embeddings import OpenAIEmbedding

        return OpenAIEmbedding(model=str(config.get("model", "text-embedding-3-small")), api_key=config.get("api_key"))
    if provider in {"azure-openai", "azure"}:
        from .embeddings import AzureOpenAIEmbedding

        return AzureOpenAIEmbedding(
            model=str(config["model"]),
            api_key=config.get("api_key"),
            azure_endpoint=config.get("azure_endpoint"),
            api_version=str(config.get("api_version", "2024-02-01")),
        )
    if provider == "cohere":
        from .embeddings import CohereEmbedding

        return CohereEmbedding(model=str(config.get("model", "embed-multilingual-v3.0")), api_key=config.get("api_key"))
    if provider == "voyage":
        from .embeddings import VoyageEmbedding

        return VoyageEmbedding(model=str(config.get("model", "voyage-multilingual-2")), api_key=config.get("api_key"))
    raise ValueError(f"Unsupported embedding provider: {provider}")


def _llm_from_config(config: dict[str, Any]) -> LLMClient:
    provider = str(config.get("provider", "extractive")).lower().replace("_", "-")
    if provider in {"extractive", "none", "local"}:
        return ExtractiveLLMClient()
    if provider in {"openai", "openai-chat"}:
        return OpenAILLMClient(model=str(config.get("model", "gpt-4o-mini")), api_key=config.get("api_key"))
    if provider in {"azure-openai", "azure"}:
        from .llms import AzureOpenAIChatClient

        return AzureOpenAIChatClient(
            model=str(config["model"]),
            api_key=config.get("api_key"),
            azure_endpoint=config.get("azure_endpoint"),
            api_version=str(config.get("api_version", "2024-02-01")),
        )
    if provider == "anthropic":
        from .llms import AnthropicClient

        return AnthropicClient(model=str(config.get("model", "claude-3-5-sonnet-latest")), api_key=config.get("api_key"))
    if provider == "ollama":
        from .llms import OllamaClient

        return OllamaClient(model=str(config.get("model", "llama3.1")), base_url=str(config.get("base_url", "http://localhost:11434")))
    if provider == "litellm":
        from .llms import LiteLLMClient

        return LiteLLMClient(model=str(config["model"]))
    raise ValueError(f"Unsupported generation provider: {provider}")


# Backward-compatible private alias used in older tests/users.
def _extract_citations(answer: str) -> list[str]:
    return extract_citations(answer)


def _tokenizer_from_config(config: Any):
    if config is None:
        return None
    if hasattr(config, "tokenize"):
        return config
    if not isinstance(config, dict):
        raise TypeError("tokenizer config must be a mapping or RetrievalTokenizer instance")
    if not config:
        return None
    from .tokenization import DEFAULT_STOPWORDS, RetrievalTokenizer

    cfg = dict(config)
    if "stopwords" in cfg and cfg["stopwords"] is not None:
        cfg["stopwords"] = frozenset(str(item) for item in cfg["stopwords"])
    elif cfg.pop("use_default_stopwords", True) is False:
        cfg["stopwords"] = frozenset()
    else:
        cfg.setdefault("stopwords", DEFAULT_STOPWORDS)
    if "ngram_range" in cfg and isinstance(cfg["ngram_range"], list):
        cfg["ngram_range"] = tuple(cfg["ngram_range"])
    return RetrievalTokenizer(**cfg)
