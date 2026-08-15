"""Learned sparse and late-interaction retrieval adapters.

The classes in this module deliberately use duck-typed encoders. Applications
can inject an already-loaded model, while tests and lightweight deployments can
provide small dependency-free encoders. Optional model libraries are imported
only when their adapters are instantiated.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import math
from typing import Any

from ..base import BaseRetriever, Document, _numpy, _validate_top_k


SparseVector = dict[Hashable, float]


class LearnedSparseRetriever(BaseRetriever):
    """Exact in-memory retriever for learned sparse vectors such as SPLADE.

    Parameters
    ----------
    documents:
        Documents to index. Documents and their metadata are copied so result
        scores never mutate caller-owned objects.
    encoder:
        An injected encoder exposing ``encode_document(s)`` and
        ``encode_query(ies)``, or a shared ``encode`` method. Each method may
        return dense arrays, SciPy sparse matrices, PyTorch sparse tensors, or
        dictionaries mapping dimensions to weights.
    model_name:
        A Sentence Transformers ``SparseEncoder`` model. It is loaded only when
        ``encoder`` is omitted.
    batch_size:
        Maximum number of inputs passed to the encoder in one call.

    Notes
    -----
    This implementation performs exact sparse dot products. It is intended for
    small and medium in-memory indexes, evaluation, and as a reference adapter.
    Large corpora should persist the same sparse vectors in a search engine
    with an inverted index.
    """

    DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"

    def __init__(
        self,
        documents: Iterable[Document],
        encoder: Any | None = None,
        *,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        model_kwargs: Mapping[str, Any] | None = None,
    ):
        _validate_batch_size(batch_size)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        self.batch_size = batch_size
        self.model_name = model_name
        self.encoder = encoder if encoder is not None else _load_sparse_encoder(model_name, model_kwargs)
        _validate_encoder(self.encoder)

        self.documents: list[Document] = []
        self._document_vectors: list[SparseVector] = []
        self._dimension: int | None = None
        self.add_documents(documents)

    def add_documents(self, documents: Iterable[Document]) -> None:
        """Encode and append documents to the in-memory index."""
        new_documents = _copy_documents(documents)
        if not new_documents:
            return
        vectors, dimension = _encode_sparse(
            self.encoder,
            [document.content for document in new_documents],
            role="document",
            batch_size=self.batch_size,
        )
        self._dimension = _merge_dimension(self._dimension, dimension, "sparse encoder")
        self.documents.extend(new_documents)
        self._document_vectors.extend(vectors)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Return documents ranked by the exact learned-sparse dot product."""
        _validate_query(query)
        _validate_top_k(top_k)
        if not self.documents:
            return []

        query_vectors, query_dimension = _encode_sparse(
            self.encoder,
            [query],
            role="query",
            batch_size=self.batch_size,
        )
        _merge_dimension(self._dimension, query_dimension, "sparse query encoder")
        query_vector = query_vectors[0]
        scores = [_sparse_dot(query_vector, vector) for vector in self._document_vectors]
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]
        return [_scored_copy(self.documents[index], scores[index], "learned_sparse") for index in order]


# SPLADE is the best-known learned sparse retrieval family. Keeping an alias
# avoids duplicating behavior while making the intended technique discoverable.
SPLADERetriever = LearnedSparseRetriever


class ColBERTRetriever(BaseRetriever):
    """Exact in-memory late-interaction retriever using ColBERT MaxSim.

    The injected ``token_encoder`` may expose role-specific
    ``encode_document(s)``/``encode_query(ies)`` methods or a shared ``encode``
    method. It must produce one ``(tokens, dimension)`` matrix per input. A
    ``(batch, tokens, dimension)`` tensor, a list of matrices, and an
    ``(embeddings, attention_mask)`` pair are supported.

    Scores use the canonical ColBERT operation: for every query token, take the
    maximum dot product over document tokens, then sum those maxima. Token
    vectors are L2-normalized by default.
    """

    def __init__(
        self,
        documents: Iterable[Document],
        token_encoder: Any | None = None,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 16,
        normalize: bool = True,
        model_kwargs: Mapping[str, Any] | None = None,
    ):
        _validate_batch_size(batch_size)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool")

        self.batch_size = batch_size
        self.model_name = model_name
        self.normalize = normalize
        self.token_encoder = (
            token_encoder
            if token_encoder is not None
            else SentenceTransformerTokenEncoder(model_name, model_kwargs=model_kwargs)
        )
        _validate_encoder(self.token_encoder)

        self.documents: list[Document] = []
        self._document_vectors: list[Any] = []
        self._dimension: int | None = None
        self.add_documents(documents)

    def add_documents(self, documents: Iterable[Document]) -> None:
        """Encode and append documents to the late-interaction index."""
        new_documents = _copy_documents(documents)
        if not new_documents:
            return
        vectors, dimension = _encode_token_vectors(
            self.token_encoder,
            [document.content for document in new_documents],
            role="document",
            batch_size=self.batch_size,
            normalize=self.normalize,
        )
        self._dimension = _merge_dimension(self._dimension, dimension, "token encoder")
        self.documents.extend(new_documents)
        self._document_vectors.extend(vectors)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Return documents ranked by exact in-memory MaxSim scores."""
        _validate_query(query)
        _validate_top_k(top_k)
        if not self.documents:
            return []

        query_vectors, query_dimension = _encode_token_vectors(
            self.token_encoder,
            [query],
            role="query",
            batch_size=self.batch_size,
            normalize=self.normalize,
        )
        _merge_dimension(self._dimension, query_dimension, "query token encoder")
        query_vector = query_vectors[0]
        scores = [_maxsim(query_vector, document_vector) for document_vector in self._document_vectors]
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]
        return [_scored_copy(self.documents[index], scores[index], "colbert_maxsim") for index in order]


class SentenceTransformerTokenEncoder:
    """Optional token-level adapter around a ``SentenceTransformer`` model.

    This adapter exposes the token embeddings produced before sentence pooling,
    together with the attention mask. It makes late-interaction experiments
    available through the existing ``cheragh[local]`` dependency. For faithful
    ColBERT quality, inject a token encoder backed by a ColBERT-trained model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        model: Any | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ):
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "ColBERTRetriever without an injected token_encoder requires "
                    "'sentence-transformers'. Install with: pip install cheragh[local]"
                ) from exc
            model = SentenceTransformer(model_name, **dict(model_kwargs or {}))
        if not callable(getattr(model, "tokenize", None)) or not callable(getattr(model, "forward", None)):
            raise TypeError("model must expose callable tokenize() and forward() methods")
        self.model_name = model_name
        self.model = model

    def encode_documents(self, texts: Sequence[str]) -> tuple[Any, Any]:
        return self._encode(texts, role="document")

    def encode_queries(self, texts: Sequence[str]) -> tuple[Any, Any]:
        return self._encode(texts, role="query")

    def _encode(self, texts: Sequence[str], *, role: str) -> tuple[Any, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - installed with sentence-transformers
            raise ImportError("SentenceTransformerTokenEncoder requires PyTorch") from exc

        prepared = self._apply_prompt(list(texts), role)
        features = self.model.tokenize(prepared)
        device = getattr(self.model, "device", None)
        if device is not None:
            features = {
                key: value.to(device) if callable(getattr(value, "to", None)) else value
                for key, value in features.items()
            }
        with torch.inference_mode():
            output = self.model.forward(features)
        if "token_embeddings" not in output:
            raise ValueError("SentenceTransformer model did not return token_embeddings")
        mask = output.get("attention_mask", features.get("attention_mask"))
        if mask is None:
            raise ValueError("SentenceTransformer model did not return an attention mask")
        return output["token_embeddings"], mask

    def _apply_prompt(self, texts: list[str], role: str) -> list[str]:
        prompts = getattr(self.model, "prompts", None)
        if not isinstance(prompts, Mapping):
            return texts
        names = ("query",) if role == "query" else ("document", "passage", "corpus")
        prompt = next((prompts[name] for name in names if isinstance(prompts.get(name), str)), "")
        return [prompt + text for text in texts] if prompt else texts


def _load_sparse_encoder(model_name: str, model_kwargs: Mapping[str, Any] | None) -> Any:
    try:
        from sentence_transformers import SparseEncoder
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "LearnedSparseRetriever without an injected encoder requires a recent "
            "'sentence-transformers' with SparseEncoder support."
        ) from exc
    return SparseEncoder(model_name, **dict(model_kwargs or {}))


def _copy_documents(documents: Iterable[Document]) -> list[Document]:
    try:
        materialized = list(documents)
    except TypeError as exc:
        raise TypeError("documents must be an iterable of Document objects") from exc
    copied: list[Document] = []
    for index, document in enumerate(materialized):
        if not isinstance(document, Document):
            raise TypeError(f"documents[{index}] must be a Document")
        if not isinstance(document.content, str):
            raise TypeError(f"documents[{index}].content must be a string")
        if document.metadata is not None and not isinstance(document.metadata, Mapping):
            raise TypeError(f"documents[{index}].metadata must be a mapping")
        copied.append(
            Document(
                content=document.content,
                metadata=dict(document.metadata or {}),
                doc_id=document.doc_id,
                score=document.score,
            )
        )
    return copied


def _validate_encoder(encoder: Any) -> None:
    names = (
        "encode",
        "encode_document",
        "encode_documents",
        "encode_query",
        "encode_queries",
    )
    if not callable(encoder) and not any(callable(getattr(encoder, name, None)) for name in names):
        raise TypeError("encoder must be callable or expose an encode method")


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")


def _validate_query(query: str) -> None:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query must not be empty")


def _encoder_method(encoder: Any, role: str) -> Callable[[Sequence[str]], Any]:
    names = (
        ("encode_documents", "encode_document", "encode")
        if role == "document"
        else ("encode_queries", "encode_query", "encode")
    )
    for name in names:
        method = getattr(encoder, name, None)
        if callable(method):
            return method
    if callable(encoder):
        return encoder
    raise TypeError(f"encoder does not provide a callable {role} encoding method")


def _encode_sparse(
    encoder: Any,
    texts: list[str],
    *,
    role: str,
    batch_size: int,
) -> tuple[list[SparseVector], int | None]:
    method = _encoder_method(encoder, role)
    vectors: list[SparseVector] = []
    dimension: int | None = None
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_vectors, batch_dimension = _sparse_rows(method(batch), expected_count=len(batch))
        dimension = _merge_dimension(dimension, batch_dimension, f"{role} sparse encoder batches")
        vectors.extend(batch_vectors)
    return vectors, dimension


def _sparse_rows(value: Any, *, expected_count: int) -> tuple[list[SparseVector], int | None]:
    if isinstance(value, Mapping):
        if expected_count != 1:
            raise ValueError(f"encoder returned 1 sparse vector for a batch of {expected_count}")
        return [_validated_sparse_mapping(value)], None

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == expected_count and all(isinstance(row, Mapping) for row in value):
            return [_validated_sparse_mapping(row) for row in value], None

    # Handle SciPy-like CSR/CSC matrices without importing SciPy.
    if callable(getattr(value, "tocsr", None)) and hasattr(value, "shape"):
        matrix = value.tocsr()
        if callable(getattr(matrix, "sum_duplicates", None)):
            matrix.sum_duplicates()
        if len(matrix.shape) != 2 or matrix.shape[0] != expected_count:
            raise ValueError(f"sparse encoder output must have {expected_count} rows")
        rows: list[SparseVector] = []
        for row_index in range(expected_count):
            start, end = int(matrix.indptr[row_index]), int(matrix.indptr[row_index + 1])
            csr_row: SparseVector = {
                int(matrix.indices[position]): float(matrix.data[position])
                for position in range(start, end)
                if float(matrix.data[position]) != 0.0
            }
            rows.append(_validated_sparse_mapping(csr_row))
        return rows, int(matrix.shape[1])

    # PyTorch sparse layouts cannot be converted directly with ``numpy()``.
    layout = str(getattr(value, "layout", ""))
    if "sparse" in layout and callable(getattr(value, "to_sparse_coo", None)):
        sparse = value.to_sparse_coo().coalesce()
        shape = tuple(int(size) for size in sparse.shape)
        indices = _as_numpy(sparse.indices())
        values = _as_numpy(sparse.values())
        if len(shape) == 1:
            if expected_count != 1:
                raise ValueError(f"encoder returned 1 sparse vector for a batch of {expected_count}")
            coo_row: SparseVector = {
                int(indices[0, offset]): float(values[offset]) for offset in range(len(values))
            }
            return [_validated_sparse_mapping(coo_row)], shape[0]
        if len(shape) != 2 or shape[0] != expected_count:
            raise ValueError(f"sparse encoder output must have shape ({expected_count}, dimensions)")
        rows = [dict() for _ in range(expected_count)]
        for offset in range(len(values)):
            rows[int(indices[0, offset])][int(indices[1, offset])] = float(values[offset])
        return [_validated_sparse_mapping(row) for row in rows], shape[1]

    array = _as_numpy(value)
    if array.ndim == 1:
        if expected_count != 1:
            raise ValueError(f"encoder returned 1 sparse vector for a batch of {expected_count}")
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] != expected_count:
        raise ValueError(f"sparse encoder output must have shape ({expected_count}, dimensions)")
    if not bool(_numpy().isfinite(array).all()):
        raise ValueError("sparse encoder output contains non-finite values")
    rows = []
    for row in array:
        nonzero = _numpy().flatnonzero(row)
        rows.append({int(index): float(row[index]) for index in nonzero})
    return rows, int(array.shape[1])


def _validated_sparse_mapping(value: Mapping[Hashable, Any]) -> SparseVector:
    row: SparseVector = {}
    for key, weight in value.items():
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError("sparse vector dimensions must be hashable") from exc
        try:
            number = float(weight)
        except (TypeError, ValueError) as exc:
            raise TypeError("sparse vector weights must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError("sparse encoder output contains non-finite values")
        if number != 0.0:
            row[key] = number
    return row


def _sparse_dot(left: SparseVector, right: SparseVector) -> float:
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    return float(sum(weight * larger.get(dimension, 0.0) for dimension, weight in smaller.items()))


def _encode_token_vectors(
    encoder: Any,
    texts: list[str],
    *,
    role: str,
    batch_size: int,
    normalize: bool,
) -> tuple[list[Any], int]:
    method = _encoder_method(encoder, role)
    vectors: list[Any] = []
    dimension: int | None = None
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_vectors = _token_rows(method(batch), expected_count=len(batch), normalize=normalize)
        batch_dimension = int(batch_vectors[0].shape[1])
        dimension = _merge_dimension(dimension, batch_dimension, f"{role} token encoder batches")
        vectors.extend(batch_vectors)
    if dimension is None:  # pragma: no cover - callers skip empty input
        raise ValueError("token encoder did not return any vectors")
    return vectors, dimension


def _token_rows(value: Any, *, expected_count: int, normalize: bool) -> list[Any]:
    embeddings, mask = _token_embeddings_and_mask(value)
    np = _numpy()

    if isinstance(embeddings, Sequence) and not isinstance(embeddings, (str, bytes)):
        if len(embeddings) == expected_count and not hasattr(embeddings, "shape"):
            rows = [_as_numpy(row) for row in embeddings]
        else:
            array = _as_numpy(embeddings)
            rows = _split_token_array(array, expected_count)
    else:
        array = _as_numpy(embeddings)
        rows = _split_token_array(array, expected_count)

    if mask is not None:
        mask_array = _as_numpy(mask)
        if mask_array.ndim == 1 and expected_count == 1:
            mask_array = mask_array.reshape(1, -1)
        if mask_array.ndim != 2 or mask_array.shape[0] != expected_count:
            raise ValueError(f"attention mask must have shape ({expected_count}, tokens)")
        if any(mask_array[index].shape[0] != rows[index].shape[0] for index in range(expected_count)):
            raise ValueError("attention mask token count does not match token embeddings")
        rows = [row[mask_array[index].astype(bool)] for index, row in enumerate(rows)]

    dimension: int | None = None
    validated: list[Any] = []
    for index, row in enumerate(rows):
        row = np.asarray(row, dtype=float)
        if row.ndim != 2:
            raise ValueError(f"token encoder output {index} must have shape (tokens, dimensions)")
        if row.shape[0] == 0 or row.shape[1] == 0:
            raise ValueError(f"token encoder output {index} must contain at least one token and dimension")
        if not bool(np.isfinite(row).all()):
            raise ValueError("token encoder output contains non-finite values")
        dimension = _merge_dimension(dimension, int(row.shape[1]), "token vectors")
        if normalize:
            norms = np.linalg.norm(row, axis=1, keepdims=True)
            row = np.divide(row, norms, out=np.zeros_like(row), where=norms > 0)
        validated.append(row)
    if len(validated) != expected_count:
        raise ValueError(f"token encoder returned {len(validated)} vectors for a batch of {expected_count}")
    return validated


def _token_embeddings_and_mask(value: Any) -> tuple[Any, Any | None]:
    if isinstance(value, Mapping) and "token_embeddings" in value:
        return value["token_embeddings"], value.get("attention_mask")
    if isinstance(value, tuple) and len(value) == 2:
        embeddings, mask = value
        try:
            embedding_array = _as_numpy(embeddings)
            mask_array = _as_numpy(mask)
        except (TypeError, ValueError):
            return value, None
        is_single_mask = (
            embedding_array.ndim == 2
            and mask_array.ndim == 1
            and mask_array.shape[0] == embedding_array.shape[0]
        )
        is_batch_mask = (
            embedding_array.ndim == 3
            and mask_array.ndim == 2
            and mask_array.shape == embedding_array.shape[:2]
        )
        if is_single_mask or is_batch_mask:
            return embeddings, mask
    return value, None


def _split_token_array(array: Any, expected_count: int) -> list[Any]:
    if array.ndim == 2:
        if expected_count != 1:
            raise ValueError(
                f"token encoder returned one token matrix for a batch of {expected_count}; "
                "return a list or a 3D batch tensor"
            )
        return [array]
    if array.ndim != 3 or array.shape[0] != expected_count:
        raise ValueError(f"token encoder output must have shape ({expected_count}, tokens, dimensions)")
    return [array[index] for index in range(expected_count)]


def _as_numpy(value: Any) -> Any:
    detached = value.detach() if callable(getattr(value, "detach", None)) else value
    on_cpu = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
    raw = on_cpu.numpy() if callable(getattr(on_cpu, "numpy", None)) else on_cpu
    try:
        return _numpy().asarray(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError("encoder output must be numeric and array-like") from exc


def _merge_dimension(current: int | None, incoming: int | None, label: str) -> int | None:
    if incoming is None:
        return current
    if current is not None and current != incoming:
        raise ValueError(f"{label} returned inconsistent dimensions: {current} and {incoming}")
    return incoming


def _maxsim(query_vectors: Any, document_vectors: Any) -> float:
    similarities = query_vectors @ document_vectors.T
    return float(similarities.max(axis=1).sum())


def _scored_copy(document: Document, score: float, retrieval_method: str) -> Document:
    metadata = dict(document.metadata or {})
    metadata["retrieval_method"] = retrieval_method
    return Document(
        content=document.content,
        metadata=metadata,
        doc_id=document.doc_id,
        score=float(score),
    )


__all__ = [
    "ColBERTRetriever",
    "LearnedSparseRetriever",
    "SPLADERetriever",
    "SentenceTransformerTokenEncoder",
]
