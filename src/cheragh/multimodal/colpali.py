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
        model_name: str | None = None,
        *,
        model: Any | None = None,
        processor: Any | None = None,
        device: str | None = None,
        torch_dtype: Any | None = None,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        model_family: str = "colpali",
        batch_size: int = 4,
        query_format: str = "auto",
    ):
        _validate_batch_size(batch_size)
        families = {
            "colpali": ("ColPali", "ColPaliProcessor", "vidore/colpali-v1.3"),
            "colqwen2": ("ColQwen2", "ColQwen2Processor", "vidore/colqwen2-v1.0"),
        }
        if model_family not in families:
            raise ValueError("model_family must be 'colpali' or 'colqwen2'")
        if query_format not in {"auto", "processor", "colpali-v1.3"}:
            raise ValueError("query_format must be 'auto', 'processor' or 'colpali-v1.3'")
        model_name = model_name or families[model_family][2]
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ColPaliEngineAdapter requires colpali-engine, torch and Pillow. "
                "Install with: pip install cheragh[colpali]"
            ) from exc

        if model is None or processor is None:
            try:
                from colpali_engine import models
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
            model_class = getattr(models, families[model_family][0])
            processor_class = getattr(models, families[model_family][1])
            if model is None:
                model = model_class.from_pretrained(model_name, **load_kwargs)
            if processor is None:
                processing_options = {
                    key: value for key, value in load_kwargs.items()
                    if key in {"cache_dir", "revision", "token", "local_files_only", "trust_remote_code"}
                }
                processing_options.update(processor_kwargs or {})
                processor = processor_class.from_pretrained(model_name, **processing_options)

        self._torch = torch
        self.model_name = model_name
        self.model = model.eval()
        self.processor = processor
        self.device = device or _model_device(model)
        self.batch_size = batch_size
        self.model_family = model_family
        self.query_format = (
            "colpali-v1.3" if query_format == "auto" and model_family == "colpali"
            and model_name in {"vidore/colpali-v1.3", "vidore/colpali-v1.3-merged"}
            else "processor" if query_format == "auto" else query_format
        )

    def embed_pages(self, pages: Sequence[MultimodalDocument]) -> Sequence[Any]:
        results: list[Any] = []
        for start in range(0, len(pages), self.batch_size):
            images: list[Any] = []
            try:
                for page in pages[start : start + self.batch_size]:
                    images.append(_load_page_image(page))
                batch = self.processor.process_images(images)
                results.extend(self._forward(batch))
            finally:
                for image in images:
                    image.close()
        return results

    def embed_queries(self, queries: Sequence[str]) -> Sequence[Any]:
        if not queries:
            return []
        if any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("ColPali queries must be non-empty strings")
        results: list[Any] = []
        for start in range(0, len(queries), self.batch_size):
            group = list(queries[start : start + self.batch_size])
            if self.query_format == "colpali-v1.3":
                tokenizer = self.processor.tokenizer
                # Preserve the checkpoint's trained render despite processor
                # changes in colpali-engine 0.3.11/0.3.13. Real augmentation PAD
                # tokens remain attended; only tokenizer-added padding is removed.
                rendered = [tokenizer.bos_token + "Query: " + query + tokenizer.pad_token * 10 + "\n" for query in group]
                batch = tokenizer(rendered, padding="longest", return_tensors="pt", return_token_type_ids=True)
                if "token_type_ids" not in batch:
                    batch["token_type_ids"] = self._torch.zeros_like(batch["input_ids"])
            else:
                batch = self.processor.process_queries(group)
            results.extend(self._forward(batch))
        return results

    def get_fingerprint(self) -> str:
        return f"colpali-engine::{self.model_name}::{self.model_family}::{self.query_format}"

    def _forward(self, batch: Any) -> Any:
        if hasattr(batch, "to"):
            batch = batch.to(self.device)
        elif isinstance(batch, dict):
            batch = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
        with self._torch.inference_mode():
            embeddings = self.model(**batch)
        embeddings = getattr(embeddings, "embeddings", embeddings)
        return _split_embeddings(embeddings, attention_mask=batch.get("attention_mask"))


class ColPaliRetriever(BaseRetriever):
    """Exact page-image MaxSim retrieval with filters and patch provenance."""

    def __init__(
        self,
        pages: Iterable[MultimodalDocument],
        encoder: VisualLateInteractionEncoder | None = None,
        *,
        normalize_vectors: bool = True,
        normalize_by_query_tokens: bool = False,
        batch_size: int = 8,
        score_batch_size: int = 16,
    ):
        _validate_batch_size(batch_size)
        _validate_batch_size(score_batch_size)
        self.encoder = encoder if encoder is not None else ColPaliEngineAdapter()
        self.batch_size = batch_size
        self.score_batch_size = score_batch_size
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
        raw_embeddings: list[Any] = []
        for start in range(0, len(encoder_pages), self.batch_size):
            batch = encoder_pages[start : start + self.batch_size]
            encoded = list(self.encoder.embed_pages(batch))
            if len(encoded) != len(batch):
                raise ValueError("Visual encoder must return one embedding matrix per page")
            raw_embeddings.extend(encoded)
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
        candidates = [index for index, page in enumerate(self.pages) if metadata_matches(page.metadata, filters)]
        for start in range(0, len(candidates), self.score_batch_size):
            indices = candidates[start : start + self.score_batch_size]
            width = max(len(self.page_embeddings[index]) for index in indices)
            padded = np.zeros((len(indices), width, query_matrix.shape[1]), dtype=query_matrix.dtype)
            valid = np.zeros((len(indices), width), dtype=bool)
            for offset, index in enumerate(indices):
                rows = self.page_embeddings[index]
                padded[offset, :len(rows)] = rows
                valid[offset, :len(rows)] = True
            similarities = np.einsum("qd,bpd->bqp", query_matrix, padded)
            similarities = np.where(valid[:, None, :], similarities, -np.inf)
            for offset, index in enumerate(indices):
                patch_indices = np.argmax(similarities[offset], axis=1)
                token_scores = np.max(similarities[offset], axis=1)
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
                "maxsim_index_space": "page_embedding_rows",
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
    if str(getattr(value, "dtype", "")) == "torch.bfloat16":
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _split_embeddings(value: Any, *, attention_mask: Any | None = None) -> list[Any]:
    np = _numpy()
    array = np.asarray(_to_numpy(value))
    if array.ndim != 3:
        raise ValueError("ColPali model must return (batch, tokens, dimension) embeddings")
    mask = np.asarray(_to_numpy(attention_mask)) if attention_mask is not None else np.ones(array.shape[:2], dtype=bool)
    if mask.shape != array.shape[:2]:
        raise ValueError("ColPali attention mask does not match embedding shape")
    # The model zeros masked positions, but leaving zero rows in MaxSim can
    # incorrectly beat negative real similarities and make scores batch-dependent.
    return [row[mask[index].astype(bool)].copy() for index, row in enumerate(array)]


def _validate_batch_size(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("batch sizes must be positive integers")


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
