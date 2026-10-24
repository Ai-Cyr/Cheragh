"""Pool chunk vectors after encoding their complete source document once."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from numbers import Integral
import threading
from typing import TYPE_CHECKING, Any, Iterable

from ...base import Document, EmbeddingModel, _numpy, _snapshot_documents, _validate_top_k
from .boundaries import SentenceTextChunker

if TYPE_CHECKING:
    import numpy as np


def _token_output(text: str, output: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate the contextual-token extension contract and take owned copies."""
    np = _numpy()
    if not isinstance(output, Mapping) or "embeddings" not in output or "offsets" not in output:
        raise ValueError("encode_tokens must return embeddings and offsets")
    vectors = np.asarray(output["embeddings"])
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("Token embeddings must be a non-empty (tokens, dimension) matrix")
    if (
        not np.issubdtype(vectors.dtype, np.number)
        or np.issubdtype(vectors.dtype, np.bool_)
        or np.issubdtype(vectors.dtype, np.complexfloating)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Token embeddings must contain finite real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        vectors = vectors.astype(float, copy=True)
    if not np.isfinite(vectors).all():
        raise ValueError("Token embeddings must remain finite after numeric conversion")
    raw_offsets = output["offsets"]
    if (
        not isinstance(raw_offsets, (list, tuple, np.ndarray))
        or (isinstance(raw_offsets, np.ndarray) and raw_offsets.ndim != 2)
        or len(raw_offsets) != vectors.shape[0]
    ):
        raise ValueError("Token offsets must have one pair per embedding")
    offsets: list[tuple[int, int]] = []
    previous_start = previous_end = 0
    for pair in raw_offsets:
        if (
            not isinstance(pair, (list, tuple, np.ndarray))
            or (isinstance(pair, np.ndarray) and pair.ndim != 1)
            or len(pair) != 2
        ):
            raise ValueError("Token offsets must be character start/end pairs")
        start, end = pair
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in pair):
            raise ValueError("Token offsets must be integer character indices")
        start, end = int(start), int(end)
        if not 0 <= start < end <= len(text):
            raise ValueError("Token offsets must be non-empty spans inside the source document")
        if start < previous_start or end < previous_end:
            raise ValueError("Token offsets must follow source order")
        offsets.append((start, end))
        previous_start, previous_end = start, end
    return vectors, np.asarray(offsets, dtype=np.int64)


def _normalized_mean(vectors: np.ndarray) -> np.ndarray:
    """Mean pooling with stable scaling before L2 normalization."""
    np = _numpy()
    scale = np.max(np.abs(vectors))
    if scale == 0:
        return np.zeros(vectors.shape[1], dtype=float)
    mean = np.mean(vectors / scale, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


@dataclass
class LateChunker:
    """Encode full document context before producing each chunk's mean vector.

    The embedding model must implement ``EmbeddingModel`` and additionally
    ``encode_tokens(text) -> {"embeddings": matrix, "offsets": pairs}``. It must
    encode the complete input without truncation and omit special/padding
    tokens. Offsets are exclusive-end character spans in that same input.

    Sentence-aware character boundaries are used for text excerpts. Each chunk
    pools the contextual vectors of tokens intersecting its source span, then
    L2-normalizes the mean. An oversized sentence may split a token across two
    excerpts; its contextual vector then contributes to both. This is late
    pooling, not independently encoding each chunk, nor ColBERT late interaction.

    Use ``chunk_and_embed`` and store the returned vectors with
    ``MemoryVectorStore.add_embeddings``. Calling ``split_documents`` is an
    error because discarding these vectors would silently remove the technique.
    """

    embedding_model: EmbeddingModel
    chunk_size: int = 800
    chunk_overlap: int = 120

    def __post_init__(self) -> None:
        SentenceTextChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        if not isinstance(self.embedding_model, EmbeddingModel) or not callable(
            getattr(self.embedding_model, "encode_tokens", None)
        ):
            raise TypeError("LateChunker requires an EmbeddingModel with encode_tokens(text)")

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        raise ValueError(
            "LateChunker requires chunk_and_embed(documents); preserve its contextual vectors "
            "with MemoryVectorStore.add_embeddings(chunks, embeddings)"
        )

    def chunk_and_embed(self, documents: Iterable[Document]) -> tuple[list[Document], np.ndarray]:
        np = _numpy()
        materialized = list(documents)
        if any(not isinstance(doc, Document) or not isinstance(doc.content, str) for doc in materialized):
            raise TypeError("documents must contain Document objects with string content")
        snapshots = _snapshot_documents(materialized)
        splitter = SentenceTextChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        result: list[Document] = []
        pooled: list[Any] = []
        dimension = getattr(self.embedding_model, "dimension", None)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            dimension = None
        for document_number, doc in enumerate(snapshots):
            spans = splitter.split_text_with_offsets(doc.content)
            if not spans:
                continue
            # Exactly one contextual forward for this entire source document.
            vectors, offsets = _token_output(doc.content, getattr(self.embedding_model, "encode_tokens")(doc.content))
            if dimension is not None and vectors.shape[1] != dimension:
                raise ValueError("Contextual embedding dimension changed or differs from the query model")
            dimension = vectors.shape[1]
            base_id = doc.doc_id if doc.doc_id is not None else f"doc-{document_number}"
            for index, span in enumerate(spans):
                first = int(np.searchsorted(offsets[:, 1], span.source_char_start, side="right"))
                last = int(np.searchsorted(offsets[:, 0], span.source_char_end, side="left"))
                if first >= last:
                    raise ValueError(
                        "A chunk has no contextual tokens; the token encoder must encode the complete "
                        "source without truncation and provide matching character offsets"
                    )
                metadata = _snapshot_documents([doc])[0].metadata
                metadata.update({
                    "chunker": "late", "chunk_method": "late", "chunk_index": index,
                    "parent_doc_id": base_id,
                    "source_char_start": span.source_char_start,
                    "source_char_end": span.source_char_end,
                    "token_start": first, "token_end": last,
                    "embedding_context": "document",
                })
                result.append(Document(span.text, metadata=metadata, doc_id=f"{base_id}#late-{index}"))
                pooled.append(_normalized_mean(vectors[first:last]))
        matrix = np.vstack(pooled) if pooled else np.zeros((0, dimension or 0), dtype=float)
        return result, matrix


class TransformersTokenEmbedding(EmbeddingModel):
    """Lazy Hugging Face encoder exposing contextual token vectors and offsets.

    Requires a fast tokenizer and an encoder model with ``last_hidden_state``.
    Special/padding tokens are excluded from pooling; queries and standalone
    document embeddings use the same normalized token mean as ``LateChunker``.
    ``max_tokens`` includes special tokens and is further limited by declared
    tokenizer/model capacities. Overlong inputs raise before a model forward;
    this adapter never requests truncation or automatically executes remote code.

    The default MiniLM encoder has a short context window. Select an appropriate
    long-context encoder and explicitly set its token budget for long documents.
    Inject both ``tokenizer`` and ``model`` for local/offline model instances.
    Their identity must correspond to ``model_name`` and ``revision`` for
    persistent indexes and caches to remain compatible.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        revision: str | None = None,
        device: str = "cpu",
        max_tokens: int = 512,
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("revision must be a non-empty string or None")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        _validate_top_k(max_tokens, name="max_tokens")
        if (tokenizer is None) != (model is None):
            raise ValueError("Inject both tokenizer and model, or neither")
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.model = model
        self.dimension = _model_dimension(model)
        self._lock = threading.RLock()
        self._initialized = False

    def _initialize(self) -> None:
        if self._initialized:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "TransformersTokenEmbedding requires torch and transformers. "
                "Install with: pip install cheragh[late-chunking]"
            ) from exc
        tokenizer, model = self.tokenizer, self.model
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.revision, use_fast=True, trust_remote_code=False,
            )
            model = AutoModel.from_pretrained(
                self.model_name, revision=self.revision, trust_remote_code=False,
            )
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Late chunking requires a fast tokenizer with source character offsets")
        if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
            raise ValueError("Use an encoder model exposing last_hidden_state, not an encoder-decoder model")
        model.to(self.device)
        model.eval()
        self.tokenizer, self.model = tokenizer, model
        self.dimension = _model_dimension(model)
        self._initialized = True

    def encode_tokens(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        with self._lock:
            self._initialize()
            import torch

            encoded = self.tokenizer(
                text, add_special_tokens=True, truncation=False, padding=False,
                return_offsets_mapping=True, return_special_tokens_mask=True,
                return_attention_mask=True, return_tensors="pt",
            )
            ids = encoded.get("input_ids")
            if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
                raise ValueError("Tokenizer must return a single untruncated sequence")
            token_count = ids.shape[1]
            limits = [self.max_tokens]
            for limit in (
                getattr(self.tokenizer, "model_max_length", None),
                getattr(getattr(self.model, "config", None), "max_position_embeddings", None),
            ):
                if isinstance(limit, Integral) and not isinstance(limit, bool):
                    numeric_limit = int(limit)
                    if 0 < numeric_limit < 1_000_000_000:
                        limits.append(numeric_limit)
            if token_count > min(limits):
                raise ValueError(
                    f"Full input has {token_count} tokens but the encoder limit is {min(limits)}; "
                    "use a suitable long-context encoder or explicitly split source documents first. "
                    "Late chunking never truncates inputs."
                )
            offsets = encoded.get("offset_mapping")
            special = encoded.get("special_tokens_mask")
            attention = encoded.get("attention_mask")
            if (
                not isinstance(offsets, torch.Tensor) or tuple(offsets.shape) != (1, token_count, 2)
                or not isinstance(special, torch.Tensor) or tuple(special.shape) != (1, token_count)
                or not isinstance(attention, torch.Tensor) or tuple(attention.shape) != (1, token_count)
            ):
                raise ValueError("Fast tokenizer must return aligned offsets, attention and special-token masks")
            model_inputs = {
                key: value.to(self.device) for key, value in encoded.items()
                if key in self.tokenizer.model_input_names
            }
            self.model.eval()
            with torch.inference_mode():
                output = self.model(**model_inputs)
            hidden = getattr(output, "last_hidden_state", None)
            if (
                not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                or hidden.shape[0] != 1 or hidden.shape[1] != token_count
            ):
                raise ValueError("Encoder must return one last_hidden_state vector per complete input token")
            raw_offsets = offsets[0].detach().cpu()
            valid = attention[0].detach().cpu().bool() & ~special[0].detach().cpu().bool()
            valid &= raw_offsets[:, 1] > raw_offsets[:, 0]
            contextual = hidden[0].detach().float().cpu()[valid].numpy()
            token_vectors, source_offsets = _token_output(text, {
                "embeddings": contextual, "offsets": raw_offsets[valid].numpy(),
            })
            if self.dimension is not None and token_vectors.shape[1] != self.dimension:
                raise ValueError("Encoder token dimension differs from its model configuration")
            self.dimension = token_vectors.shape[1]
            return {"embeddings": token_vectors, "offsets": source_offsets.tolist()}

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        np = _numpy()
        vectors = [self.embed_query(text) for text in texts]
        return np.vstack(vectors) if vectors else np.zeros((0, self.dimension or 0), dtype=float)

    def embed_query(self, text: str) -> np.ndarray:
        return _normalized_mean(self.encode_tokens(text)["embeddings"])

    def get_fingerprint(self) -> str:
        return "TransformersTokenEmbedding::" + json.dumps({
            "model": self.model_name, "revision": self.revision, "device": self.device,
            "max_tokens": self.max_tokens, "pooling": "content-token-mean-l2-v1",
        }, sort_keys=True, separators=(",", ":"))


def _model_dimension(model: Any) -> int | None:
    config = getattr(model, "config", None)
    for name in ("hidden_size", "d_model", "dim"):
        dimension = getattr(config, name, None)
        if isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0:
            return dimension
    return None
