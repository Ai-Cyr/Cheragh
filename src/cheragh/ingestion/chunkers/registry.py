"""Shared chunking strategies for Python, configuration, CLI and HTTP clients."""
from __future__ import annotations

import math
from typing import Any, Iterable, Protocol

from ...base import Document, EmbeddingModel, HashingEmbedding, LLMClient


class DocumentChunker(Protocol):
    def split_documents(self, documents: Iterable[Document]) -> list[Document]: ...


_STRATEGY_DETAILS = (
    ("recursive", "Récursif", "characters",
     "Privilégie les paragraphes, les lignes puis les espaces. Taille et chevauchement en caractères."),
    ("sentence", "Phrases", "characters",
     "Regroupe les phrases sous une limite de caractères. Les frontières linguistiques sont heuristiques."),
    ("paragraph", "Paragraphes", "characters",
     "Privilégie les paragraphes séparés par des lignes vides ; découpe les blocs trop longs."),
    ("token", "Tokens approximatifs", "tokens",
     "Compte les mots séparés par des espaces, et non les tokens exacts du modèle. Taille et chevauchement en mots."),
    ("semantic", "Semantic chunking", "characters",
     "Détecte les changements de sujet par similarité des embeddings, avec contexte de phrases et seuil adaptatif optionnel."),
    ("late", "Late chunking", "characters",
     "Encode le document entier avant de regrouper les vecteurs de ses tokens par passage. Nécessite un encodeur compatible."),
    ("contextual", "Contextual chunking", "characters",
     "Un LLM situe chaque passage dans le document avant son embedding. Les citations conservent le texte original."),
)

CHUNKING_STRATEGIES: tuple[str, ...] = tuple(item[0] for item in _STRATEGY_DETAILS)


def validate_chunking_options(strategy: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate serializable strategy options without echoing input values."""
    if not isinstance(strategy, str) or strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"chunking_strategy must be one of: {', '.join(CHUNKING_STRATEGIES)}")
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError("chunking_options must be a mapping")
    allowed = {
        "semantic": {
            "breakpoint_threshold", "breakpoint_percentile", "buffer_size", "min_chunk_size", "min_sentences",
            "embedding_batch_size", "max_buffer_chars", "max_sentences",
        },
        "contextual": {"max_document_chars", "max_context_chars", "embedding_batch_size", "max_chunks_per_document"},
    }.get(strategy, set())
    if any(key not in allowed for key in options):
        raise ValueError(f"Unsupported chunking_options for {strategy}")
    integer_bounds = {
        "buffer_size": (0, 32), "min_chunk_size": (0, 1_000_000), "min_sentences": (1, 10_000),
        "max_document_chars": (1, 1_000_000), "max_context_chars": (1, 10_000),
        "embedding_batch_size": (1, 1_024), "max_chunks_per_document": (1, 10_000),
        "max_buffer_chars": (1, 1_000_000), "max_sentences": (1, 100_000),
    }
    for key, value in options.items():
        if key in integer_bounds:
            minimum, maximum = integer_bounds[key]
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"chunking_options.{key} must be an integer between {minimum} and {maximum}")
        elif key == "breakpoint_percentile" and value is None:
            continue
        elif not _finite_number(value):
            raise ValueError(f"chunking_options.{key} must be a finite number")
        elif key == "breakpoint_threshold" and not 0 <= value <= 1:
            raise ValueError("chunking_options.breakpoint_threshold must be between 0 and 1")
        elif key == "breakpoint_percentile" and not 0 < value < 100:
            raise ValueError("chunking_options.breakpoint_percentile must be between 0 and 100")
    return dict(options)


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def list_chunking_strategies() -> list[dict[str, str]]:
    """Return independent descriptions of the configurable strategies."""
    return [
        dict(zip(("id", "label", "unit", "description"), details))
        for details in _STRATEGY_DETAILS
    ]


def build_chunker(
    chunking_strategy: str = "recursive",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    *,
    embedding_model: EmbeddingModel | None = None,
    llm_client: LLMClient | None = None,
    chunking_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DocumentChunker:
    """Construct a chunker without loading a model or external dependency.

    Size and overlap use characters, except for ``token`` (whitespace-delimited
    words unless a fast tokenizer is explicitly provided through Python).
    Strategy-specific keyword arguments are passed to the selected class and
    rejected by that class if unsupported. The recursive default is unchanged.
    """
    validated = validate_chunking_options(chunking_strategy, chunking_options)
    options = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, **validated, **kwargs}
    if chunking_strategy == "recursive":
        from .recursive import RecursiveTextChunker
        return RecursiveTextChunker(**options)
    if chunking_strategy == "token":
        from .token import TokenTextChunker
        return TokenTextChunker(**options)
    if chunking_strategy in {"sentence", "paragraph"}:
        from .boundaries import ParagraphChunker, SentenceTextChunker
        return (SentenceTextChunker if chunking_strategy == "sentence" else ParagraphChunker)(**options)
    if embedding_model is None:
        raise ValueError(f"{chunking_strategy} chunking requires an explicit embedding_model")
    # Known cache wrappers preserve query/document embedding behavior but do
    # not expose the underlying encoder's optional token extension.
    from ...cache import CachedEmbeddingModel

    source_model = embedding_model
    seen_models: set[int] = set()
    while isinstance(source_model, CachedEmbeddingModel):
        if id(source_model) in seen_models:
            raise ValueError("Cyclic embedding cache wrappers are not supported")
        seen_models.add(id(source_model))
        source_model = source_model.model
    if chunking_strategy == "semantic":
        if isinstance(source_model, HashingEmbedding):
            raise ValueError("Semantic chunking requires semantic embeddings, not the hashing baseline")
        from .advanced import SemanticChunker
        from .boundaries import SentenceTextChunker

        semantic_options = {**validated, **kwargs}
        fallback = SentenceTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return SemanticChunker(
            embedding_model=embedding_model, max_chunk_size=chunk_size,
            fallback_chunker=fallback, **semantic_options,
        )
    if chunking_strategy == "late":
        from .late import LateChunker
        return LateChunker(embedding_model=source_model, **options)
    if llm_client is None:
        raise ValueError("Contextual chunking requires an explicit llm_client")
    from .contextual import ContextualChunker
    return ContextualChunker(embedding_model=embedding_model, llm_client=llm_client, **options)
