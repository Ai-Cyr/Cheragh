"""Visual document retrieval with ColPali-compatible late interaction.

The retriever in this module implements the page-image multi-vector boundary
introduced by ColPali while keeping model inference injectable.  The bundled
adapter loads ``colpali-engine`` only when instantiated; deterministic tests and
proprietary providers can instead use :class:`CallableVisualLateInteractionEncoder`.

Scoring is exact and in memory: for every query vector, take its maximum dot
product over the page vectors and sum those maxima.  Large corpora should use a
multi-vector index or a candidate-generation stage before this exact reranker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..base import BaseRetriever, Document, _numpy, _validate_top_k
from ..filters import metadata_matches
from .retrieval import Modality, MultimodalDocument


class VisualLateInteractionEncoder(ABC):
    """Encode text queries and document pages into variable-length vectors."""

    @abstractmethod
    def embed_pages(self, pages: Sequence[MultimodalDocument]) -> Sequence[Any]:
        """Return one ``(patches, dimension)`` matrix per page."""

    @abstractmethod
    def embed_queries(self, queries: Sequence[str]) -> Sequence[Any]:
        """Return one ``(tokens, dimension)`` matrix per query."""

    def get_fingerprint(self) -> str:
        return self.__class__.__name__


class CallableVisualLateInteractionEncoder(VisualLateInteractionEncoder):
    """Adapt callables that already produce ColPali-style multi-vectors."""

    def __init__(
        self,
        page_encoder: Callable[[Sequence[MultimodalDocument]], Sequence[Any]],
        query_encoder: Callable[[Sequence[str]], Sequence[Any]],
        *,
        fingerprint: str = "callable-visual-late-interaction",
    ):
        self.page_encoder = page_encoder
        self.query_encoder = query_encoder
        self.fingerprint = fingerprint

    def embed_pages(self, pages: Sequence[MultimodalDocument]) -> Sequence[Any]:
        return self.page_encoder(pages)

    def embed_queries(self, queries: Sequence[str]) -> Sequence[Any]:
        return self.query_encoder(queries)

    def get_fingerprint(self) -> str:
        return self.fingerprint


class ColPaliEngineAdapter(VisualLateInteractionEncoder):
    """Optional adapter around the official ``colpali-engine`` API.

    By default the adapter loads ``ColPali`` and ``ColPaliProcessor``.  A model
    and processor may be injected for another ColVision family supported by the
    official package, such as ColQwen or ColSmol.
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        *,
        model: Any | None = None,
        processor: Any | None = None,
        device: str | None = None,
        torch_dtype: Any | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ColPaliEngineAdapter requires colpali-engine, torch and Pillow. "
                "Install with: pip install cheragh[colpali]"
            ) from exc

        if model is None or processor is None:
            try:
                from colpali_engine.models import ColPali, ColPaliProcessor
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "ColPaliEngineAdapter requires colpali-engine. "
                    "Install with: pip install cheragh[colpali]"
                ) from exc
            load_kwargs = dict(model_kwargs or {})
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype
            if device is not None:
                load_kwargs.setdefault("device_map", device)
            model = model or ColPali.from_pretrained(model_name, **load_kwargs).eval()
            processor = processor or ColPaliProcessor.from_pretrained(model_name)

        self._torch = torch
        self.model_name = model_name
        self.model = model
        self.processor = processor
        self.device = device or _model_device(model)

    def embed_pages(self, pages: Sequence[MultimodalDocument]) -> Sequence[Any]:
        images = [_load_page_image(page) for page in pages]
        if not images:
            return []
        batch = self.processor.process_images(images)
        embeddings = self._forward(batch)
        return _split_embeddings(embeddings)

    def embed_queries(self, queries: Sequence[str]) -> Sequence[Any]:
        if not queries:
            return []
        if any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("ColPali queries must be non-empty strings")
        batch = self.processor.process_queries(list(queries))
        embeddings = self._forward(batch)
        return _split_embeddings(embeddings)

    def get_fingerprint(self) -> str:
        return f"colpali-engine::{self.model_name}"

    def _forward(self, batch: Any) -> Any:
        if hasattr(batch, "to"):
            batch = batch.to(self.device)
        elif isinstance(batch, dict):
            batch = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
        with self._torch.no_grad():
            return self.model(**batch)


class ColPaliRetriever(BaseRetriever):
    """Exact page-image MaxSim retrieval with filters and patch provenance."""

    def __init__(
        self,
        pages: Iterable[MultimodalDocument],
        encoder: VisualLateInteractionEncoder,
        *,
        normalize_vectors: bool = True,
        normalize_by_query_tokens: bool = False,
    ):
        self.encoder = encoder
        self.normalize_vectors = bool(normalize_vectors)
        self.normalize_by_query_tokens = bool(normalize_by_query_tokens)
        self.pages: list[MultimodalDocument] = []
        self.page_embeddings: list[Any] = []
        self.dimension: int | None = None
        self.add_pages(pages)

    def add_pages(self, pages: Iterable[MultimodalDocument]) -> None:
        snapshots = [_snapshot_page(page) for page in pages]
        if not snapshots:
            return
        encoder_pages = [_snapshot_page(page) for page in snapshots]
        raw_embeddings = list(self.encoder.embed_pages(encoder_pages))
        if len(raw_embeddings) != len(snapshots):
            raise ValueError("Visual encoder must return one embedding matrix per page")
        expected_dimension = self.dimension
        matrices: list[Any] = []
        for value in raw_embeddings:
            matrix = self._validated_matrix(
                value,
                kind="page",
                expected_dimension=expected_dimension,
            )
            expected_dimension = int(matrix.shape[1])
            matrices.append(matrix)
        self.dimension = expected_dimension
        self.pages.extend(snapshots)
        self.page_embeddings.extend(matrices)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return list(self.retrieve_pages(query, top_k=top_k))

    def retrieve_pages(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MultimodalDocument]:
        np = _numpy()
        top_k = _validate_top_k(top_k)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("ColPali retrieval requires a non-empty text query")
        if not self.pages:
            return []
        raw_queries = list(self.encoder.embed_queries([query]))
        if len(raw_queries) != 1:
            raise ValueError("Visual encoder must return one embedding matrix per query")
        query_matrix = self._validated_matrix(
            raw_queries[0],
            kind="query",
            expected_dimension=self.dimension,
        )

        ranked: list[tuple[float, int, Any, Any]] = []
        for index, (page, page_matrix) in enumerate(zip(self.pages, self.page_embeddings)):
            if not metadata_matches(page.metadata, filters):
                continue
            similarities = query_matrix @ page_matrix.T
            patch_indices = np.argmax(similarities, axis=1)
            token_scores = np.max(similarities, axis=1)
            score = float(np.sum(token_scores))
            if self.normalize_by_query_tokens:
                score /= max(1, query_matrix.shape[0])
            ranked.append((score, index, patch_indices, token_scores))
        ranked.sort(key=lambda item: (-item[0], self.pages[item[1]].doc_id or "", item[1]))

        results: list[MultimodalDocument] = []
        for score, index, patch_indices, token_scores in ranked[:top_k]:
            source = self.pages[index]
            metadata = {
                **deepcopy(source.metadata),
                "modality": source.modality.value,
                "uri": source.uri,
                "mime_type": source.mime_type,
                "retrieval_method": "colpali-maxsim",
                "maxsim_patch_indices": [int(value) for value in patch_indices.tolist()],
                "maxsim_token_scores": [float(value) for value in token_scores.tolist()],
                "visual_encoder": self.encoder.get_fingerprint(),
            }
            results.append(
                MultimodalDocument(
                    content=source.content,
                    metadata=metadata,
                    doc_id=source.doc_id,
                    score=score,
                    modality=source.modality,
                    uri=source.uri,
                    mime_type=source.mime_type,
                )
            )
        return results

    def _validated_matrix(
        self,
        value: Any,
        *,
        kind: str,
        expected_dimension: int | None,
    ) -> Any:
        np = _numpy()
        matrix = np.array(_to_numpy(value), dtype=float, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(f"ColPali {kind} embeddings must have shape (vectors, dimension)")
        if not np.isfinite(matrix).all():
            raise ValueError(f"ColPali {kind} embeddings must contain only finite values")
        if expected_dimension is not None and matrix.shape[1] != expected_dimension:
            raise ValueError(
                f"ColPali {kind} embedding dimension {matrix.shape[1]} does not match {expected_dimension}"
            )
        return _normalize_rows(matrix) if self.normalize_vectors else matrix


def _snapshot_page(page: MultimodalDocument) -> MultimodalDocument:
    if not isinstance(page, MultimodalDocument):
        raise TypeError("ColPaliRetriever pages must be MultimodalDocument instances")
    if page.modality != Modality.IMAGE:
        raise ValueError("ColPaliRetriever indexes page images; modality must be 'image'")
    return MultimodalDocument(
        content=page.content,
        metadata=deepcopy(page.metadata or {}),
        doc_id=page.doc_id,
        score=page.score,
        modality=page.modality,
        uri=page.uri,
        mime_type=page.mime_type,
    )


def _normalize_rows(matrix: Any) -> Any:
    np = _numpy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _split_embeddings(value: Any) -> list[Any]:
    if hasattr(value, "unbind"):
        return list(value.unbind(0))
    return list(value)


def _model_device(model: Any) -> str:
    device = getattr(model, "device", None)
    return str(device) if device is not None else "cpu"


def _load_page_image(page: MultimodalDocument) -> Any:
    if page.modality != Modality.IMAGE or not page.uri:
        raise ValueError("ColPaliEngineAdapter requires image pages with a local uri")
    path = Path(page.uri)
    if not path.is_file():
        raise FileNotFoundError(f"ColPali page image must be a local file: {page.uri}")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("ColPali image loading requires Pillow") from exc
    with Image.open(path) as image:
        return image.convert("RGB").copy()


__all__ = [
    "CallableVisualLateInteractionEncoder",
    "ColPaliEngineAdapter",
    "ColPaliRetriever",
    "VisualLateInteractionEncoder",
]
