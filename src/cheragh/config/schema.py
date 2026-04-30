"""Pydantic models for validated RAG configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("Config validation requires pydantic>=2. Install with: pip install cheragh[config]") from exc


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class IngestionConfig(StrictBaseModel):
    path: str | None = None
    chunk_size: int = Field(default=800, ge=1, le=1_000_000)
    chunk_overlap: int = Field(default=120, ge=0, le=1_000_000)
    recursive: bool = True
    max_file_size_mb: float | None = Field(default=None, gt=0)
    exclude_patterns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_overlap(self) -> "IngestionConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("ingestion.chunk_overlap must be smaller than ingestion.chunk_size")
        return self


class EmbeddingConfig(StrictBaseModel):
    provider: str = "hashing"
    model: str | None = None
    dimension: int = Field(default=384, ge=1, le=100_000)
    api_key: str | None = None
    azure_endpoint: str | None = None
    api_version: str | None = None

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return value.lower().replace("_", "-")


class RetrieverConfig(StrictBaseModel):
    type: str = "hybrid"
    top_k: int = Field(default=5, ge=1, le=1_000)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    filters: dict[str, Any] = Field(default_factory=dict)
    tokenizer: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        return value.lower().replace("_", "-")


class VectorStoreConfig(StrictBaseModel):
    type: str | None = None
    path: str | None = None
    collection_name: str = "cheragh"
    url: str | None = None
    api_key: str | None = None
    normalize: bool = True

    @field_validator("type")
    @classmethod
    def normalize_type(cls, value: str | None) -> str | None:
        return value.lower().replace("_", "-") if value else None


class RerankerConfig(StrictBaseModel):
    enabled: bool = False
    provider: str = "cross-encoder"
    model: str | None = None
    first_stage_top_k: int = Field(default=30, ge=1, le=10_000)


class ToggleTypeConfig(StrictBaseModel):
    enabled: bool = False
    type: str = "default"
    transform: str | None = None


class GenerationConfig(StrictBaseModel):
    provider: str = "extractive"
    model: str | None = None
    api_key: str | None = None
    azure_endpoint: str | None = None
    api_version: str | None = None
    base_url: str | None = None

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return value.lower().replace("_", "-")


class CacheConfig(StrictBaseModel):
    enabled: bool = False
    backend: Literal["memory", "sqlite", "sqlite3", "redis", "in-memory", "mem"] = "memory"
    type: str | None = None
    path: str | None = None
    cache_path: str | None = None
    ttl: float | None = Field(default=None, gt=0)
    default_ttl: float | None = Field(default=None, gt=0)
    namespace: str = "default"
    serializer: Literal["json", "pickle", "signed-pickle"] = "json"
    secret_key: str | None = None
    hmac_key: str | None = None
    allow_pickle: bool | None = None
    allow_unsigned_pickle: bool = False
    redis_url: str | None = None
    url: str | None = None
    key_prefix: str = "cheragh"
    cache_embeddings: bool = True
    cache_retrieval: bool = True
    cache_reranking: bool = True
    cache_llm: bool = True

    @model_validator(mode="before")
    @classmethod
    def normalize_backend_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if data.get("backend") is None and data.get("type") is not None:
                data["backend"] = data["type"]
        return data

    @model_validator(mode="after")
    def validate_pickle_safety(self) -> "CacheConfig":
        backend = self.backend.replace("_", "-")
        if self.serializer == "signed-pickle" and not (self.secret_key or self.hmac_key):
            raise ValueError("cache.serializer='signed-pickle' requires cache.secret_key or cache.hmac_key")
        if backend in {"sqlite", "sqlite3", "redis"} and self.serializer == "pickle":
            effective_allow_pickle = True if self.allow_pickle is None else self.allow_pickle
            if not effective_allow_pickle:
                raise ValueError("cache.serializer='pickle' requires cache.allow_pickle=true")
            if not (self.secret_key or self.hmac_key or self.allow_unsigned_pickle):
                raise ValueError(
                    "persistent pickle cache requires cache.secret_key "
                    "or cache.allow_unsigned_pickle=true"
                )
        return self


class ObservabilityConfig(StrictBaseModel):
    enabled: bool = True
    trace_export_path: str | None = None
    trace_include_prompt: bool = False
    pricing: dict[str, float | str] = Field(default_factory=dict)


class IndexingConfig(StrictBaseModel):
    incremental: bool = True
    dry_run: bool = False
    use_lock: bool = True
    lock_timeout_seconds: float = Field(default=10.0, ge=0)
    force: bool = False


class RAGConfig(StrictBaseModel):
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    compression: ToggleTypeConfig = Field(default_factory=ToggleTypeConfig)
    query: ToggleTypeConfig = Field(default_factory=lambda: ToggleTypeConfig(type="multi-query"))
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    strict_grounding: bool = False
    require_citations: bool | None = None
    flag_unsourced_sentences: bool = False
    trace_enabled: bool = True
    min_score: float | None = None
    answer_prompt: str | None = None
    cache_backend: str | None = None
    cache_path: str | None = None
    cache_ttl: float | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_cache(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        cache = dict(data.get("cache") or {})
        if data.get("cache_backend") is not None and "backend" not in cache:
            cache["enabled"] = True
            cache["backend"] = data.get("cache_backend")
        if data.get("cache_path") is not None and "path" not in cache:
            cache["path"] = data.get("cache_path")
        if data.get("cache_ttl") is not None and "ttl" not in cache:
            cache["ttl"] = data.get("cache_ttl")
        if cache:
            data["cache"] = cache
        return data

    @model_validator(mode="after")
    def validate_retriever_vectorstore(self) -> "RAGConfig":
        retriever_type = self.retriever.type
        vector_type = self.vectorstore.type
        if vector_type and retriever_type in {"hybrid", "memory", "vector"}:
            # Backward-compatible convention: vectorstore.type overrides retriever.type in RAGEngine.from_config.
            return self
        supported = {"hybrid", "memory", "vector", "faiss", "chroma", "qdrant"}
        effective_type = vector_type or retriever_type
        if effective_type not in supported:
            raise ValueError(f"Unsupported retriever/vectorstore type: {effective_type}")
        return self

    def to_legacy_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


def validate_config(data: dict[str, Any]) -> RAGConfig:
    """Validate raw config data and return a typed config model."""

    try:
        return RAGConfig.model_validate(data)
    except ValidationError:
        raise


def load_and_validate_config(path: str | Path) -> RAGConfig:
    from .loader import load_raw_config

    return validate_config(load_raw_config(path))
