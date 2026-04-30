"""
Technique 14 : Semantic Chunking — version persistable.

Utilitaire de découpage (pas un retriever). Si un `cache_path` est
fourni à `chunk_documents`, les chunks produits sont sérialisés et
rechargés si le corpus n'a pas changé — on évite de re-embedder toutes
les phrases à chaque run.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

import numpy as np

from .base import Document, EmbeddingModel
from .cache import hash_documents, embedder_fingerprint, load_cache, save_cache
from .sentence_window import split_sentences


class SemanticChunker:
    _CACHEABLE_VERSION = 1

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        breakpoint_percentile: float = 95.0,
        buffer_size: int = 1,
        min_chunk_sentences: int = 2,
        max_chunk_sentences: int = 50,
    ):
        if not 0 < breakpoint_percentile < 100:
            raise ValueError("breakpoint_percentile doit être dans (0, 100).")
        self.embedding_model = embedding_model
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences

    # ------------------------------------------------------------------ #
    # API publique — avec cache optionnel
    # ------------------------------------------------------------------ #
    def chunk_documents(
        self,
        documents: List[Document],
        cache_path: Optional[str] = None,
    ) -> List[Document]:
        """Découpe avec cache optionnel. Même signature qu'avant + cache_path."""
        if cache_path:
            cached = load_cache(
                path=cache_path,
                expected_class=self.__class__.__name__,
                expected_content_hash=hash_documents(documents),
                expected_embedder_fp=embedder_fingerprint(self.embedding_model),
                expected_extra_fp=self._extra_fp(),
            )
            if cached is not None:
                return cached["chunks"]

        # Calcul normal
        all_chunks: List[Document] = []
        for doc in documents:
            chunks = self.chunk_text(doc.content)
            for i, chunk_text in enumerate(chunks):
                parent_id = doc.doc_id or str(uuid.uuid4())
                all_chunks.append(
                    Document(
                        content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "source_doc_id": parent_id,
                            "chunk_index": i,
                            "n_chunks_total": len(chunks),
                        },
                        doc_id=f"{parent_id}::semchunk::{i}",
                    )
                )

        if cache_path:
            save_cache(
                path=cache_path,
                retriever_class=self.__class__.__name__,
                content_hash=hash_documents(documents),
                embedder_fp=embedder_fingerprint(self.embedding_model),
                extra_fingerprint=self._extra_fp(),
                state={"chunks": all_chunks},
            )
        return all_chunks

    def chunk_text(self, text: str) -> List[str]:
        sentences = split_sentences(text)
        if len(sentences) <= self.min_chunk_sentences:
            return [text] if text.strip() else []

        buffered = self._build_buffered(sentences)
        embeddings = self.embedding_model.embed_documents(buffered)
        distances = self._consecutive_distances(embeddings)

        if len(distances) == 0:
            return [" ".join(sentences)]
        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        breakpoints = [i for i, d in enumerate(distances) if d > threshold]

        chunks = self._build_chunks(sentences, breakpoints)
        return self._enforce_size_bounds(chunks)

    # ------------------------------------------------------------------ #
    def _extra_fp(self) -> str:
        return (
            f"v={self._CACHEABLE_VERSION};"
            f"pct={self.breakpoint_percentile};buf={self.buffer_size};"
            f"min={self.min_chunk_sentences};max={self.max_chunk_sentences}"
        )

    def _build_buffered(self, sentences: List[str]) -> List[str]:
        b = self.buffer_size
        return [
            " ".join(sentences[max(0, i - b) : min(len(sentences), i + b + 1)])
            for i in range(len(sentences))
        ]

    @staticmethod
    def _consecutive_distances(embeddings: np.ndarray) -> List[float]:
        if len(embeddings) < 2:
            return []
        sims = (embeddings[:-1] * embeddings[1:]).sum(axis=1)
        return (1.0 - sims).tolist()

    @staticmethod
    def _build_chunks(sentences: List[str], breakpoints: List[int]) -> List[str]:
        if not breakpoints:
            return [" ".join(sentences)]
        chunks = []
        start = 0
        for bp in breakpoints:
            end = bp + 1
            chunks.append(" ".join(sentences[start:end]))
            start = end
        if start < len(sentences):
            chunks.append(" ".join(sentences[start:]))
        return chunks

    def _enforce_size_bounds(self, chunks: List[str]) -> List[str]:
        result: List[str] = []
        for c in chunks:
            sents = split_sentences(c)
            if len(sents) > self.max_chunk_sentences:
                for i in range(0, len(sents), self.max_chunk_sentences):
                    result.append(" ".join(sents[i : i + self.max_chunk_sentences]))
            else:
                result.append(c)

        merged: List[str] = []
        buffer = ""
        for c in result:
            sents_count = len(split_sentences(c))
            if sents_count < self.min_chunk_sentences:
                buffer = (buffer + " " + c).strip()
            else:
                if buffer:
                    c = (buffer + " " + c).strip()
                    buffer = ""
                merged.append(c)
        if buffer:
            if merged:
                merged[-1] = merged[-1] + " " + buffer
            else:
                merged.append(buffer)
        return merged
