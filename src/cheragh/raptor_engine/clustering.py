"""Optional RAPTOR global/local UMAP and soft Gaussian-mixture clustering.

Implements the clustering mechanism in Sarthi et al. (2024), section 3:
https://arxiv.org/abs/2401.18059 . This is an independent implementation, with
bounded optimization, deterministic seeds and explicit coverage guarantees.
It does not reproduce the paper's encoders, summarizers or evaluation results.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any
import warnings

import numpy as np

from ..base import _validate_non_negative_int, _validate_top_k


@dataclass(frozen=True)
class RAPTORClusteringConfig:
    """Controls for the optional ``umap_gmm`` construction mode.

    ``membership_threshold`` applies to posterior component probabilities at
    both stages. A point may belong to several clusters; if none pass, its
    highest-probability component is retained so no document disappears.
    BIC considers 1 through ``min(max_clusters, sample_count - 1)`` components.
    ``max_cluster_points`` bounds work and memory; partition larger corpora
    explicitly instead of silently sampling away their content.
    """

    reduction_dimension: int = 10
    membership_threshold: float = 0.1
    max_clusters: int = 50
    random_state: int = 224
    local_neighbors: int = 10
    max_iter: int = 100
    n_init: int = 1
    reg_covar: float = 1e-6
    max_cluster_points: int = 10_000

    def __post_init__(self) -> None:
        for name in (
            "reduction_dimension", "max_clusters", "local_neighbors", "max_iter",
            "n_init", "max_cluster_points",
        ):
            _validate_top_k(getattr(self, name), name=name)
        if self.local_neighbors < 2:
            raise ValueError("local_neighbors must be >= 2")
        _validate_non_negative_int(self.random_state, name="random_state")
        if self.random_state > 2**32 - 1:
            raise ValueError("random_state must fit an unsigned 32-bit integer")
        for name in ("membership_threshold", "reg_covar"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.membership_threshold < 1:
            raise ValueError("membership_threshold must be in [0, 1)")
        if self.reg_covar <= 0:
            raise ValueError("reg_covar must be > 0")


class UMAPGMMClusterer:
    """Cluster embedding rows into overlapping lists of original row indices.

    Scientific dependencies are imported only when nontrivial clustering is
    requested. Install ``cheragh[raptor]``. Empty, tiny and all-identical inputs
    have a deterministic single-cluster solution without fitting a model.
    """

    def __init__(self, config: RAPTORClusteringConfig | None = None):
        if config is not None and not isinstance(config, RAPTORClusteringConfig):
            raise TypeError("config must be a RAPTORClusteringConfig or None")
        self.config = config or RAPTORClusteringConfig()

    def cluster(self, embeddings: Any) -> list[list[int]]:
        matrix = self._validate_matrix(embeddings)
        size = len(matrix)
        if not size:
            return []
        if size < 3 or np.all(matrix == matrix[0]):
            return [list(range(size))]
        umap_class, mixture_class, convergence_warning = self._load_dependencies()
        reduced = self._reduce(matrix, umap_class, local=False)
        global_memberships = self._soft_cluster(reduced, mixture_class, convergence_warning)
        groups: list[list[int]] = []
        # Map row indices directly. Equality matching embedding values confuses
        # distinct documents whose embeddings happen to be identical.
        global_labels = sorted({label for row in global_memberships for label in row})
        for label in global_labels:
            indices = [i for i, row in enumerate(global_memberships) if label in row]
            subset = matrix[indices]
            if len(subset) <= self.config.reduction_dimension + 1 or np.all(subset == subset[0]):
                groups.append(indices)
                continue
            local_reduced = self._reduce(subset, umap_class, local=True)
            memberships = self._soft_cluster(local_reduced, mixture_class, convergence_warning)
            for local_label in sorted({label for row in memberships for label in row}):
                groups.append([indices[i] for i, row in enumerate(memberships) if local_label in row])
        # Soft assignments can yield the same set through different global
        # paths. Summarize that set once while preserving genuine overlap.
        return [list(group) for group in dict.fromkeys(tuple(group) for group in groups)]

    def _validate_matrix(self, embeddings: Any) -> np.ndarray:
        try:
            matrix = np.asarray(embeddings, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("RAPTOR embeddings must be a rectangular numeric matrix") from exc
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("RAPTOR embeddings must have shape (documents, dimensions > 0)")
        if len(matrix) > self.config.max_cluster_points:
            raise ValueError("RAPTOR input exceeds max_cluster_points; partition the corpus explicitly")
        if not np.isfinite(matrix).all():
            raise ValueError("RAPTOR embeddings must contain only finite values")
        return matrix

    @staticmethod
    def _load_dependencies() -> tuple[Any, Any, type[Warning]]:
        try:
            from umap import UMAP
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.mixture import GaussianMixture
        except ImportError as exc:
            raise ImportError(
                "RAPTOR umap_gmm clustering requires optional dependencies; install cheragh[raptor]"
            ) from exc
        return UMAP, GaussianMixture, ConvergenceWarning

    def _reduce(self, matrix: np.ndarray, umap_class: Any, *, local: bool) -> np.ndarray:
        neighbors = self.config.local_neighbors if local else int(math.sqrt(len(matrix) - 1))
        reducer = umap_class(
            n_components=min(self.config.reduction_dimension, len(matrix) - 2, matrix.shape[1]),
            n_neighbors=max(2, min(neighbors, len(matrix) - 1)),
            metric="cosine",
            random_state=self.config.random_state,
            transform_seed=self.config.random_state,
            n_jobs=1,
            init="random",  # Spectral initialization is undefined for tiny groups.
        )
        reduced = np.asarray(reducer.fit_transform(matrix), dtype=np.float64)
        if reduced.ndim != 2 or reduced.shape[0] != len(matrix) or not np.isfinite(reduced).all():
            raise ValueError("UMAP returned invalid or non-finite RAPTOR embeddings")
        return reduced

    def _soft_cluster(
        self, matrix: np.ndarray, mixture_class: Any, convergence_warning: type[Warning],
    ) -> list[list[int]]:
        unique_count = len(np.unique(matrix, axis=0))
        if len(matrix) < 2 or unique_count < 2:
            return [[0] for _ in matrix]
        best_model: Any = None
        best_bic = math.inf
        max_components = min(self.config.max_clusters, len(matrix) - 1, unique_count)
        for components in range(1, max_components + 1):
            model = mixture_class(
                n_components=components,
                covariance_type="full",
                random_state=self.config.random_state,
                max_iter=self.config.max_iter,
                n_init=self.config.n_init,
                reg_covar=self.config.reg_covar,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", convergence_warning)
                    model.fit(matrix)
                if not model.converged_:
                    continue
                bic = float(model.bic(matrix))
            except (ValueError, np.linalg.LinAlgError):
                # A degenerate covariance for one candidate should not discard
                # valid simpler models. If every candidate fails, fail clearly.
                continue
            if math.isfinite(bic) and bic < best_bic:
                best_bic, best_model = bic, model
        if best_model is None:
            raise ValueError("No finite, converged RAPTOR Gaussian mixture; inspect embeddings or increase max_iter")
        probabilities = np.asarray(best_model.predict_proba(matrix), dtype=np.float64)
        if (
            probabilities.shape != (len(matrix), best_model.n_components)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0)
        ):
            raise ValueError("Gaussian mixture returned invalid membership probabilities")
        return [
            np.flatnonzero(row > self.config.membership_threshold).tolist() or [int(np.argmax(row))]
            for row in probabilities
        ]
