"""Pydantic models for validated RAG configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("Config validation requires pydantic>=2. Install with: pip install cheragh[config]") from exc


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)


_RETRIEVER_TYPES = {"bm25", "hybrid", "memory", "vector", "faiss", "chroma", "qdrant"}
_VECTORSTORE_TYPES = {"hybrid", "memory", "vector", "faiss", "chroma", "qdrant"}
_EMBEDDING_PROVIDERS = {
    "hashing",
    "local-hash",
    "sentence-transformers",
    "sentence-transformer",
    "openai",
    "azure-openai",
    "azure",
    "cohere",
    "voyage",
}
_GENERATION_PROVIDERS = {
    "extractive",
    "none",
    "local",
    "openai",
    "openai-chat",
    "azure-openai",
    "azure",
    "anthropic",
    "ollama",
    "litellm",
}
_RERANKER_PROVIDERS = {
    "cross-encoder",
    "crossencoder",
    "sentence-transformers",
    "keyword",
    "keyword-overlap",
    "local",
    "cohere",
}
_COMPRESSION_TYPES = {
    "default",
    "pipeline",
    "extractive",
    "sentence",
    "sentences",
    "redundancy",
    "dedupe",
    "redundancy-filter",
}
_QUERY_TRANSFORM_TYPES = {
    "identity",
    "none",
    "multi-query",
    "multiquery",
    "multi",
    "step-back",
    "stepback",
}


def _normalize_choice(value: str) -> str:
    return value.lower().replace("_", "-")


def _validated_choice(value: str, *, supported: set[str], label: str) -> str:
    normalized = _normalize_choice(value)
    if normalized not in supported:
        raise ValueError(f"Unsupported {label}: {normalized}")
    return normalized


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
        return _validated_choice(value, supported=_EMBEDDING_PROVIDERS, label="embedding provider")

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "EmbeddingConfig":
        if self.provider in {"azure-openai", "azure"} and not self.model:
            raise ValueError("embedding.model is required for the Azure OpenAI provider")
        return self


class RetrieverConfig(StrictBaseModel):
    type: str = "hybrid"
    top_k: int = Field(default=5, ge=1, le=1_000)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_k1: float = Field(default=1.5, gt=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    filters: dict[str, Any] = Field(default_factory=dict)
    tokenizer: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        return _validated_choice(value, supported=_RETRIEVER_TYPES, label="retriever type")

    @field_validator("tokenizer")
    @classmethod
    def validate_tokenizer_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "lowercase",
            "strip_accents",
            "keep_hyphenated",
            "stopwords",
            "ngram_range",
            "min_token_length",
            "use_default_stopwords",
        }
        normalized = dict(value)
        if "normalize_accents" in normalized:
            alias_value = normalized.pop("normalize_accents")
            if "strip_accents" in normalized and normalized["strip_accents"] != alias_value:
                raise ValueError("retriever.tokenizer normalize_accents conflicts with strip_accents")
            normalized.setdefault("strip_accents", alias_value)
        unknown = set(normalized) - allowed
        if unknown:
            raise ValueError(f"Unsupported retriever.tokenizer options: {', '.join(sorted(unknown))}")

        for key in ("lowercase", "strip_accents", "keep_hyphenated", "use_default_stopwords"):
            if key in normalized and not isinstance(normalized[key], bool):
                raise ValueError(f"retriever.tokenizer.{key} must be a boolean")

        if "min_token_length" in normalized:
            minimum = normalized["min_token_length"]
            if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
                raise ValueError("retriever.tokenizer.min_token_length must be an integer >= 1")

        if "ngram_range" in normalized:
            ngram_range = normalized["ngram_range"]
            if not isinstance(ngram_range, (list, tuple)) or len(ngram_range) != 2:
                raise ValueError("retriever.tokenizer.ngram_range must contain exactly two integers")
            minimum, maximum = ngram_range
            if (
                isinstance(minimum, bool)
                or isinstance(maximum, bool)
                or not isinstance(minimum, int)
                or not isinstance(maximum, int)
                or minimum < 1
                or minimum > maximum
            ):
                raise ValueError("retriever.tokenizer.ngram_range must be like [1, 2]")
            normalized["ngram_range"] = [minimum, maximum]

        if "stopwords" in normalized:
            stopwords = normalized["stopwords"]
            if stopwords is None:
                raise ValueError("retriever.tokenizer.stopwords cannot be null; use [] to disable them")
            if isinstance(stopwords, (str, bytes)) or not isinstance(stopwords, (list, tuple, set, frozenset)):
                raise ValueError("retriever.tokenizer.stopwords must be a collection of strings")
            if not all(isinstance(item, str) for item in stopwords):
                raise ValueError("retriever.tokenizer.stopwords must contain only strings")
            if "use_default_stopwords" in normalized:
                raise ValueError(
                    "retriever.tokenizer.stopwords and use_default_stopwords are mutually exclusive"
                )
            normalized["stopwords"] = list(stopwords)
        return normalized


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
        if value is None:
            return None
        return _validated_choice(value, supported=_VECTORSTORE_TYPES, label="vectorstore type")


class RerankerConfig(StrictBaseModel):
    enabled: bool = False
    provider: str = "cross-encoder"
    model: str | None = None
    first_stage_top_k: int = Field(default=30, ge=1, le=10_000)

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return _validated_choice(value, supported=_RERANKER_PROVIDERS, label="reranker provider")


class ToggleTypeConfig(StrictBaseModel):
    enabled: bool = False
    type: str = "default"
    transform: str | None = None


class CompressionConfig(ToggleTypeConfig):
    @field_validator("type")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        return _validated_choice(value, supported=_COMPRESSION_TYPES, label="compression type")

    @model_validator(mode="after")
    def validate_transform(self) -> "CompressionConfig":
        if self.transform is not None:
            raise ValueError("compression.transform is unsupported; use compression.type")
        return self


class QueryConfig(ToggleTypeConfig):
    type: str = "multi-query"

    @field_validator("type")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        return _validated_choice(value, supported=_QUERY_TRANSFORM_TYPES, label="query transform type")

    @field_validator("transform")
    @classmethod
    def normalize_transform(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validated_choice(value, supported=_QUERY_TRANSFORM_TYPES, label="query transform")

    @model_validator(mode="after")
    def validate_selector_aliases(self) -> "QueryConfig":
        if (
            "type" in self.model_fields_set
            and "transform" in self.model_fields_set
            and self.transform is not None
            and self.type != self.transform
        ):
            raise ValueError("query.type and query.transform conflict; configure only one selector")
        return self


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
        return _validated_choice(value, supported=_GENERATION_PROVIDERS, label="generation provider")

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "GenerationConfig":
        if self.provider in {"azure-openai", "azure", "litellm"} and not self.model:
            raise ValueError(f"generation.model is required for the {self.provider} provider")
        return self


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
            for key in ("backend", "type"):
                if isinstance(data.get(key), str):
                    data[key] = data[key].lower().replace("_", "-")
            if data.get("backend") is None and data.get("type") is not None:
                data["backend"] = data["type"]
        return data

    @model_validator(mode="after")
    def validate_pickle_safety(self) -> "CacheConfig":
        backend = self.backend.replace("_", "-")
        aliases = {"mem": "memory", "in-memory": "memory", "sqlite3": "sqlite"}
        if self.type is not None:
            configured_type = aliases.get(self.type, self.type)
            configured_backend = aliases.get(backend, backend)
            if configured_type != configured_backend:
                raise ValueError("cache.backend and cache.type conflict; configure only one selector")
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
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
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
        retriever_type_was_explicit = "type" in self.retriever.model_fields_set
        vector_type_was_explicit = "type" in self.vectorstore.model_fields_set and vector_type is not None
        aliases = {"vector": "memory"}
        if (
            retriever_type_was_explicit
            and vector_type_was_explicit
            and aliases.get(retriever_type, retriever_type) != aliases.get(vector_type, vector_type)
        ):
            raise ValueError(
                "retriever.type and vectorstore.type conflict; configure one selector or use matching values"
            )
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
