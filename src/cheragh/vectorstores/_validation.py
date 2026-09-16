"""Validate provider vectors before mutating an external index."""

import numpy as np

from .memory import _validate_finite_embeddings


def embedding_matrix(value, *, rows: int) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0:
        raise ValueError(f"Expected {rows} non-empty embedding rows, received shape {matrix.shape}")
    _validate_finite_embeddings(matrix, label="document embeddings")
    with np.errstate(over="ignore"):
        matrix = matrix.astype(np.float32, copy=True)
    _validate_finite_embeddings(matrix, label="float32 document embeddings")
    return matrix


def query_vector(value, *, dimension: int | None = None) -> np.ndarray:
    vector = np.asarray(value)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"Expected a non-empty query vector, received shape {vector.shape}")
    if dimension is not None and vector.size != dimension:
        raise ValueError(f"Embedding dimension mismatch: expected {dimension}, got {vector.size}")
    return embedding_matrix(vector.reshape(1, -1), rows=1)[0]
