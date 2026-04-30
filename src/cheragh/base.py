"""Core abstractions for the :mod:`cheragh` package.

The package deliberately keeps integrations optional: a retriever only needs an
object implementing :class:`EmbeddingModel`, and a generation pipeline only needs
an object implementing :class:`LLMClient`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


def _numpy():
    import numpy as np

    return np


@dataclass
class Document:
    """A source document or chunk returned by a retriever."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    score: Optional[float] = None

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"Document(id={self.doc_id}, score={self.score}, content='{preview}...')"


class EmbeddingModel(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Return an array shaped ``(n_documents, dimension)``."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Return an array shaped ``(dimension,)``."""

    def get_fingerprint(self) -> str:
        """Stable identifier used for cache invalidation."""
        return self.__class__.__name__


class LLMClient(ABC):
    """Interface for text generation clients."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt."""

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream generated text.

        Providers can override this method. The default implementation yields
        the full result once, so every ``LLMClient`` is stream-compatible.
        """
        yield self.generate(prompt, **kwargs)


class BaseRetriever(ABC):
    """Common interface for all retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Return up to ``top_k`` relevant documents for ``query``."""


class HashingEmbedding(EmbeddingModel):
    """Small dependency-free embedding model for tests, demos and fallbacks.

    It builds normalized hashed bag-of-words vectors. This is not a semantic
    embedding model, but it makes the package runnable without downloading a
    transformer model. For production RAG, prefer ``SentenceTransformerEmbedding``
    or your own embedding provider.
    """

    def __init__(self, dimension: int = 384, ngram_range: tuple[int, int] = (1, 2)):
        if dimension <= 0:
            raise ValueError("dimension must be > 0")
        if ngram_range[0] <= 0 or ngram_range[0] > ngram_range[1]:
            raise ValueError("ngram_range must be like (1, 2)")
        self.dimension = dimension
        self.ngram_range = ngram_range

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        np = _numpy()
        return np.vstack([self.embed_query(text) for text in texts]) if texts else np.zeros((0, self.dimension))

    def embed_query(self, text: str) -> np.ndarray:
        np = _numpy()
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = _tokenize(text)
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for gram in _ngrams(tokens, n):
                digest = hashlib.blake2b(" ".join(gram).encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[bucket] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def get_fingerprint(self) -> str:
        return f"HashingEmbedding::{self.dimension}::{self.ngram_range}"


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embeddings backed by ``sentence-transformers``."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **model_kwargs: Any):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "SentenceTransformerEmbedding requires the optional dependency "
                "'sentence-transformers'. Install with: pip install cheragh[local]"
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **model_kwargs)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        np = _numpy()
        return np.asarray(self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True))

    def embed_query(self, text: str) -> np.ndarray:
        np = _numpy()
        return np.asarray(self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0])

    def get_fingerprint(self) -> str:
        return f"SentenceTransformer::{self.model_name}"


class CallableLLMClient(LLMClient):
    """Wrap any Python callable as an LLM client."""

    def __init__(self, generate_fn: Callable[..., str]):
        self.generate_fn = generate_fn

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return str(self.generate_fn(prompt, **kwargs))

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        result = self.generate_fn(prompt, **kwargs)
        if isinstance(result, str):
            yield result
            return
        try:
            for chunk in result:
                yield str(chunk)
        except TypeError:
            yield str(result)


class OpenAILLMClient(LLMClient):
    """OpenAI chat-completions client."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, **client_kwargs: Any):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "OpenAILLMClient requires the optional dependency 'openai'. "
                "Install with: pip install cheragh[openai]"
            ) from exc
        self.client = OpenAI(api_key=api_key, **client_kwargs) if api_key else OpenAI(**client_kwargs)
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def stream(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        for event in stream:  # pragma: no cover - provider integration
            chunk = event.choices[0].delta.content
            if chunk:
                yield chunk


class StaticLLMClient(LLMClient):
    """Deterministic LLM client useful for tests."""

    def __init__(self, response: str = ""):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        return self.response


class ExtractiveLLMClient(LLMClient):
    """Dependency-free fallback that returns the most relevant retrieved context."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        context_lines = [
            line for line in lines
            if not line.lower().startswith(("question", "réponse", "reponse", "answer"))
        ]
        return " ".join(context_lines[:4])[:1200] or "Aucun contexte exploitable fourni."


def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Cosine-like similarity for already-normalized vectors."""
    np = _numpy()
    if query_vec.ndim == 1:
        query_vec = query_vec[np.newaxis, :]
    if doc_matrix is None or len(doc_matrix) == 0:
        return np.array([])
    return (doc_matrix @ query_vec.T).flatten()


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to ``[0, 1]`` while handling flat vectors."""
    np = _numpy()
    if len(scores) == 0:
        return scores
    s_min, s_max = float(scores.min()), float(scores.max())
    if s_max - s_min < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - s_min) / (s_max - s_min)


def _tokenize(text: str) -> list[str]:
    from .tokenization import tokenize

    return tokenize(text)


def _ngrams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
    from .tokenization import ngrams

    return ngrams(tokens, n)
