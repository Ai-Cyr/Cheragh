"""Dependency-light multimodal RAG.

This implementation provides a genuine cross-modal retrieval boundary while
remaining usable in offline tests through an injected encoder.  The bundled
CLIP adapter supports text and local images.  Audio and video documents are
retrieved through their transcript/description until a compatible custom
encoder is supplied.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any

from ..base import BaseRetriever, Document, ExtractiveLLMClient, LLMClient, _numpy, _validate_top_k
from ..citations import validate_citations
from ..filters import metadata_matches
from ..schema import RAGResponse, Source
from ..tracing import RAGTrace


class Modality(str, Enum):
    """Media modalities understood by the public multimodal schema."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABLE = "table"


@dataclass
class MultimodalDocument(Document):
    """Document with explicit media provenance.

    ``content`` contains text, a caption, a transcript, or a compact structural
    description. ``uri`` identifies the original asset without loading it at
    import time.
    """

    modality: Modality = Modality.TEXT
    uri: str | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        self.modality = Modality(self.modality)
        if not self.doc_id:
            raw = f"{self.modality.value}\0{self.uri or ''}\0{self.content}"
            self.doc_id = f"media-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class MultimodalQuery:
    """Text or media query passed to a multimodal encoder."""

    text: str = ""
    modality: Modality = Modality.TEXT
    uri: str | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", Modality(self.modality))
        if not self.text and not self.uri:
            raise ValueError("A multimodal query requires text or a media URI")


class MultimodalEmbeddingModel(ABC):
    """Protocol-like base class for joint text/media embedding spaces."""

    @abstractmethod
    def embed_documents(self, documents: Sequence[MultimodalDocument]) -> Any:
        """Return a two-dimensional array, one vector per document."""

    @abstractmethod
    def embed_query(self, query: MultimodalQuery) -> Any:
        """Return one query vector in the same embedding space."""

    def get_fingerprint(self) -> str:
        return self.__class__.__name__


class CallableMultimodalEmbedding(MultimodalEmbeddingModel):
    """Wrap two callables as a multimodal encoder.

    This adapter is useful for proprietary providers and deterministic tests.
    """

    def __init__(
        self,
        document_encoder: Callable[[Sequence[MultimodalDocument]], Any],
        query_encoder: Callable[[MultimodalQuery], Any],
        fingerprint: str = "callable-multimodal",
    ):
        self.document_encoder = document_encoder
        self.query_encoder = query_encoder
        self.fingerprint = fingerprint

    def embed_documents(self, documents: Sequence[MultimodalDocument]) -> Any:
        return _numpy().asarray(self.document_encoder(documents), dtype=float)

    def embed_query(self, query: MultimodalQuery) -> Any:
        return _numpy().asarray(self.query_encoder(query), dtype=float)

    def get_fingerprint(self) -> str:
        return self.fingerprint


class CLIPMultimodalEmbedding(MultimodalEmbeddingModel):
    """Optional Sentence-Transformers CLIP adapter for text and local images."""

    def __init__(self, model_name: str = "clip-ViT-B-32", **model_kwargs: Any):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CLIPMultimodalEmbedding requires sentence-transformers and Pillow. "
                "Install with: pip install cheragh[multimodal]"
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **model_kwargs)

    def embed_documents(self, documents: Sequence[MultimodalDocument]) -> Any:
        payloads = [self._payload(doc.modality, doc.content, doc.uri) for doc in documents]
        return _numpy().asarray(self.model.encode(payloads, normalize_embeddings=True, show_progress_bar=False))

    def embed_query(self, query: MultimodalQuery) -> Any:
        payload = self._payload(query.modality, query.text, query.uri)
        encoded = self.model.encode([payload], normalize_embeddings=True, show_progress_bar=False)
        return _numpy().asarray(encoded[0])

    def get_fingerprint(self) -> str:
        return f"SentenceTransformersCLIP::{self.model_name}"

    @staticmethod
    def _payload(modality: Modality, text: str, uri: str | None) -> Any:
        if modality == Modality.IMAGE and uri:
            try:
                from PIL import Image
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("Image embedding requires Pillow") from exc
            path = Path(uri)
            if not path.is_file():
                raise FileNotFoundError(f"Multimodal image must be a local file: {uri}")
            with Image.open(path) as image:
                return image.convert("RGB").copy()
        if text:
            return text
        raise ValueError(f"{modality.value} content requires text or a supported local asset")


class MultimodalRetriever(BaseRetriever):
    """Exact in-memory cross-modal retriever with optional modality weighting."""

    def __init__(
        self,
        documents: Iterable[MultimodalDocument],
        embedding_model: MultimodalEmbeddingModel,
        modality_weights: dict[Modality | str, float] | None = None,
    ):
        self.embedding_model = embedding_model
        self.documents: list[MultimodalDocument] = []
        self.embeddings: Any | None = None
        self.modality_weights: dict[Modality, float] = {
            Modality(key): float(value) for key, value in (modality_weights or {}).items()
        }
        self.add_documents(documents)

    def add_documents(self, documents: Iterable[MultimodalDocument]) -> None:
        np = _numpy()
        docs = [_snapshot_multimodal_document(doc) for doc in documents]
        if not docs:
            return
        vectors = np.asarray(self.embedding_model.embed_documents(docs), dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != len(docs):
            raise ValueError("Multimodal document embeddings must have shape (documents, dimension)")
        if not np.isfinite(vectors).all():
            raise ValueError("Multimodal embeddings must contain only finite values")
        vectors = _normalize_rows(vectors)
        if self.embeddings is not None and self.embeddings.shape[1] != vectors.shape[1]:
            raise ValueError("New multimodal embeddings use a different dimension")
        self.documents.extend(docs)
        self.embeddings = vectors if self.embeddings is None else np.vstack([self.embeddings, vectors])

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        _validate_top_k(top_k)
        return list(self.retrieve_multimodal(MultimodalQuery(text=query), top_k=top_k))

    def retrieve_multimodal(
        self,
        query: MultimodalQuery,
        top_k: int = 5,
        modalities: Iterable[Modality | str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[MultimodalDocument]:
        np = _numpy()
        _validate_top_k(top_k)
        if not self.documents or self.embeddings is None:
            return []
        allowed = {Modality(value) for value in modalities} if modalities is not None else None
        candidates = [
            index
            for index, doc in enumerate(self.documents)
            if (allowed is None or doc.modality in allowed) and metadata_matches(doc.metadata, filters)
        ]
        if not candidates:
            return []
        vector = np.asarray(self.embedding_model.embed_query(query), dtype=float).reshape(-1)
        if vector.size != self.embeddings.shape[1] or not np.isfinite(vector).all():
            raise ValueError("Multimodal query embedding has an invalid dimension or values")
        norm = float(np.linalg.norm(vector))
        vector = vector / norm if norm else vector
        scores = self.embeddings[candidates] @ vector
        for position, index in enumerate(candidates):
            scores[position] *= self.modality_weights.get(self.documents[index].modality, 1.0)
        order = np.argsort(scores)[::-1][:top_k]
        results: list[MultimodalDocument] = []
        for local_index in order:
            index = candidates[int(local_index)]
            source = self.documents[index]
            metadata = {
                **source.metadata,
                "modality": source.modality.value,
                "uri": source.uri,
                "mime_type": source.mime_type,
                "retrieval_method": "multimodal-dense",
            }
            results.append(
                MultimodalDocument(
                    content=source.content,
                    metadata=metadata,
                    doc_id=source.doc_id,
                    score=float(scores[int(local_index)]),
                    modality=source.modality,
                    uri=source.uri,
                    mime_type=source.mime_type,
                )
            )
        return results


class MultimodalRAGEngine:
    """Grounded generation over text and media retrieval results.

    A text-only LLM receives captions/transcripts plus asset provenance.  Callers
    can inject a vision-capable ``LLMClient`` whose implementation interprets
    the media references in the rendered prompt.
    """

    def __init__(
        self,
        retriever: MultimodalRetriever,
        llm_client: LLMClient | None = None,
        top_k: int = 5,
        require_citations: bool = True,
        trace_enabled: bool = True,
    ):
        self.retriever = retriever
        self.llm_client = llm_client or ExtractiveLLMClient()
        self.top_k = _validate_top_k(top_k)
        self.require_citations = require_citations
        self.trace_enabled = trace_enabled

    def ask(
        self,
        query: str | MultimodalQuery,
        top_k: int | None = None,
        modalities: Iterable[Modality | str] | None = None,
        filters: dict[str, Any] | None = None,
        **generate_kwargs: Any,
    ) -> RAGResponse:
        request = query if isinstance(query, MultimodalQuery) else MultimodalQuery(text=query)
        effective_top_k = self.top_k if top_k is None else _validate_top_k(top_k)
        trace_query = request.text or str(request.uri)
        trace = RAGTrace(query=trace_query) if self.trace_enabled else None
        retrieval_step = trace.start_step("multimodal_retrieval", top_k=effective_top_k) if trace else None
        docs = self.retriever.retrieve_multimodal(
            request,
            top_k=effective_top_k,
            modalities=modalities,
            filters=filters,
        )
        if retrieval_step:
            retrieval_step.finish(document_count=len(docs))
        if trace:
            trace.add_retrieval(trace_query, list(docs))
        context = self._format_context(docs)
        prompt = (
            "Réponds uniquement avec les sources multimodales fournies. "
            "Cite chaque affirmation avec [source: doc_id].\n\n"
            f"Sources :\n{context}\n\nQuestion : {request.text or request.uri}\nRéponse :"
        )
        if trace:
            trace.prompt = prompt
        generation_step = trace.start_step("multimodal_generation", document_count=len(docs)) if trace else None
        answer = self.llm_client.generate(prompt, **generate_kwargs)
        if generation_step:
            generation_step.finish(answer_chars=len(answer))
        validation = validate_citations(answer, docs, require_citations=self.require_citations)
        if trace:
            trace.record_generation(
                prompt=prompt,
                answer=answer,
                model=getattr(self.llm_client, "model", None),
            )
            trace.warnings.extend(validation.warnings)
            trace.finish(answer_chars=len(answer), prompt_chars=len(prompt))
        return RAGResponse(
            query=trace_query,
            answer=answer,
            sources=[Source.from_document(doc) for doc in docs],
            retrieved_documents=list(docs),
            prompt=prompt,
            metadata={
                "architecture": "multimodal_rag",
                "query_modality": request.modality.value,
                "top_k": effective_top_k,
            },
            citations=validation.citations,
            warnings=validation.warnings,
            grounded_score=validation.grounded_score,
            unsourced_claims=validation.unsourced_claims,
            citation_validation=validation,
            trace=trace,
        )

    @staticmethod
    def _format_context(documents: Sequence[MultimodalDocument]) -> str:
        blocks = []
        for doc in documents:
            asset = f" uri={doc.uri}" if doc.uri else ""
            blocks.append(
                f"[source: {doc.doc_id}] [modality: {doc.modality.value}]{asset}\n"
                f"{doc.content or '(asset sans description textuelle)'}"
            )
        return "\n\n---\n\n".join(blocks)


def _normalize_rows(matrix: Any) -> Any:
    np = _numpy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def _snapshot_multimodal_document(document: MultimodalDocument | dict[str, Any]) -> MultimodalDocument:
    """Copy media documents so later caller mutations cannot stale embeddings."""

    if not isinstance(document, MultimodalDocument):
        document = MultimodalDocument(**deepcopy(document))
    return MultimodalDocument(
        content=document.content,
        metadata=deepcopy(document.metadata or {}),
        doc_id=document.doc_id,
        score=document.score,
        modality=document.modality,
        uri=document.uri,
        mime_type=document.mime_type,
    )
