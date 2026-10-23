"""Embedding provider integrations.

All third-party SDKs are optional. Classes only import provider packages when
instantiated, so the base package stays lightweight.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from .base import EmbeddingModel, _numpy

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


def _normalize(matrix: np.ndarray) -> np.ndarray:
    np = _numpy()
    if matrix.size == 0:
        return matrix
    if matrix.ndim == 1:
        norm = np.linalg.norm(matrix.astype(np.float64))
        return matrix / norm if norm > 0 else matrix
    # Finite float32 coordinates may overflow when their squares are summed.
    # Accumulate in float64 so normalization cannot silently erase a vector.
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _provider_vectors(value: Any, *, rows: int, normalize: bool = True) -> np.ndarray:
    """Reject incomplete or malformed provider batches before indexing/caching."""
    from .vectorstores._validation import embedding_matrix

    matrix = embedding_matrix(value, rows=rows)
    return _normalize(matrix) if normalize else matrix


class OpenAIEmbedding(EmbeddingModel):
    """Embedding model backed by the OpenAI embeddings API.

    Parameters
    ----------
    model:
        Embedding model name.
    api_key:
        Optional API key. When omitted, the OpenAI SDK reads environment
        variables such as ``OPENAI_API_KEY``.
    client:
        Optional preconfigured client, useful for tests or custom transports.
    normalize:
        Normalize vectors to unit length. Keep enabled for cosine/IP retrieval.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        client: Any | None = None,
        normalize: bool = True,
        **client_kwargs: Any,
    ):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "OpenAIEmbedding requires the optional dependency 'openai'. "
                    "Install with: pip install cheragh[openai]"
                ) from exc
            client = OpenAI(api_key=api_key, **client_kwargs) if api_key else OpenAI(**client_kwargs)
        self.client = client
        self.model = model
        self.normalize = normalize

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        np = _numpy()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        response = self.client.embeddings.create(model=self.model, input=texts)
        data = list(response.data)
        indices = [getattr(item, "index", None) for item in data]
        if (
            any(type(index) is not int for index in indices)
            or sorted(indices) != list(range(len(texts)))
        ):
            raise ValueError("Embedding response must contain each input index exactly once")
        data.sort(key=lambda item: item.index)
        return _provider_vectors(
            [item.embedding for item in data], rows=len(texts), normalize=self.normalize,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def get_fingerprint(self) -> str:
        return f"OpenAIEmbedding::{self.model}::normalize={self.normalize}"


class AzureOpenAIEmbedding(OpenAIEmbedding):
    """Embedding model backed by Azure OpenAI.

    The ``model`` argument should match the Azure deployment name used for the
    embeddings endpoint.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-02-01",
        client: Any | None = None,
        normalize: bool = True,
        **client_kwargs: Any,
    ):
        if client is None:
            try:
                from openai import AzureOpenAI
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "AzureOpenAIEmbedding requires the optional dependency 'openai'. "
                    "Install with: pip install cheragh[openai]"
                ) from exc
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                **client_kwargs,
            )
        super().__init__(model=model, client=client, normalize=normalize)
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version

    def get_fingerprint(self) -> str:
        return f"AzureOpenAIEmbedding::{self.model}::{self.api_version}::normalize={self.normalize}"


class CohereEmbedding(EmbeddingModel):
    """Embedding model backed by Cohere Embed."""

    def __init__(
        self,
        model: str = "embed-multilingual-v3.0",
        api_key: Optional[str] = None,
        client: Any | None = None,
        **client_kwargs: Any,
    ):
        if client is None:
            try:
                import cohere
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("CohereEmbedding requires: pip install cheragh[cohere]") from exc
            client = cohere.Client(api_key, **client_kwargs)
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        np = _numpy()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        response = self.client.embed(texts=texts, model=self.model, input_type="search_document")
        return _provider_vectors(response.embeddings, rows=len(texts))

    def embed_query(self, text: str) -> np.ndarray:
        response = self.client.embed(texts=[text], model=self.model, input_type="search_query")
        return _provider_vectors(response.embeddings, rows=1)[0]

    def get_fingerprint(self) -> str:
        return f"CohereEmbedding::{self.model}"


class VoyageEmbedding(EmbeddingModel):
    """Embedding model backed by Voyage AI."""

    def __init__(
        self,
        model: str = "voyage-multilingual-2",
        api_key: Optional[str] = None,
        client: Any | None = None,
        **client_kwargs: Any,
    ):
        if client is None:
            try:
                import voyageai
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("VoyageEmbedding requires: pip install cheragh[voyage]") from exc
            client = voyageai.Client(api_key=api_key, **client_kwargs)
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        np = _numpy()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        response = self.client.embed(texts, model=self.model, input_type="document")
        return _provider_vectors(response.embeddings, rows=len(texts))

    def embed_query(self, text: str) -> np.ndarray:
        response = self.client.embed([text], model=self.model, input_type="query")
        return _provider_vectors(response.embeddings, rows=1)[0]

    def get_fingerprint(self) -> str:
        return f"VoyageEmbedding::{self.model}"
