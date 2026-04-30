"""
Technique 5 : Parent Document Retrieval — version persistable.
"""
from __future__ import annotations

import re
import uuid
from typing import Dict, List, Optional

import numpy as np

from .base import BaseRetriever, Document, EmbeddingModel, cosine_similarity
from .cache import hash_documents, embedder_fingerprint, load_cache, save_cache


class ParentDocumentRetriever(BaseRetriever):
    _CACHEABLE_VERSION = 1

    def __init__(
        self,
        parent_documents: List[Document],
        embedding_model: EmbeddingModel,
        child_chunk_size: int = 200,
        child_chunk_overlap: int = 40,
        cache_path: Optional[str] = None,
    ):
        if child_chunk_overlap >= child_chunk_size:
            raise ValueError("child_chunk_overlap doit être < child_chunk_size.")

        self.embedding_model = embedding_model
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self._cache_path = cache_path

        # S'assurer que chaque parent a un id STABLE (sinon le hash change à chaque run)
        self.parent_documents: Dict[str, Document] = {}
        for p in parent_documents:
            if p.doc_id is None:
                # Id déterministe basé sur le contenu pour la stabilité du cache
                p.doc_id = f"parent::{abs(hash(p.content)) % (10 ** 16)}"
            self.parent_documents[p.doc_id] = p

        self.child_documents: List[Document] = []
        self.child_embeddings: Optional[np.ndarray] = None

        if not self._try_load_cache():
            self.child_documents = self._create_children()
            self.child_embeddings = embedding_model.embed_documents(
                [c.content for c in self.child_documents]
            )
            self._save_cache()

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        query_vec = self.embedding_model.embed_query(query)
        scores = cosine_similarity(query_vec, self.child_embeddings)

        candidate_count = max(top_k * 3, 20)
        top_child_idx = np.argsort(scores)[::-1][:candidate_count]

        seen_parents: Dict[str, float] = {}
        parent_order: List[str] = []
        for i in top_child_idx:
            child = self.child_documents[i]
            parent_id = child.metadata["parent_id"]
            score = float(scores[i])
            if parent_id not in seen_parents:
                seen_parents[parent_id] = score
                parent_order.append(parent_id)
            else:
                seen_parents[parent_id] = max(seen_parents[parent_id], score)

        results: List[Document] = []
        for pid in parent_order[:top_k]:
            parent = self.parent_documents[pid]
            results.append(
                Document(
                    content=parent.content,
                    metadata={**parent.metadata, "best_child_score": seen_parents[pid]},
                    doc_id=parent.doc_id,
                    score=seen_parents[pid],
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #
    def _parent_list(self) -> List[Document]:
        # Liste ordonnée pour hash stable
        return [self.parent_documents[pid] for pid in sorted(self.parent_documents)]

    def _extra_fp(self) -> str:
        return (
            f"v={self._CACHEABLE_VERSION};"
            f"size={self.child_chunk_size};overlap={self.child_chunk_overlap}"
        )

    def _try_load_cache(self) -> bool:
        if not self._cache_path:
            return False
        state = load_cache(
            path=self._cache_path,
            expected_class=self.__class__.__name__,
            expected_content_hash=hash_documents(self._parent_list()),
            expected_embedder_fp=embedder_fingerprint(self.embedding_model),
            expected_extra_fp=self._extra_fp(),
        )
        if state is None:
            return False
        self.child_documents = state["child_documents"]
        self.child_embeddings = state["child_embeddings"]
        return True

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        save_cache(
            path=self._cache_path,
            retriever_class=self.__class__.__name__,
            content_hash=hash_documents(self._parent_list()),
            embedder_fp=embedder_fingerprint(self.embedding_model),
            extra_fingerprint=self._extra_fp(),
            state={
                "child_documents": self.child_documents,
                "child_embeddings": self.child_embeddings,
            },
        )

    # ------------------------------------------------------------------ #
    def _create_children(self) -> List[Document]:
        children: List[Document] = []
        for parent_id, parent in self.parent_documents.items():
            words = re.findall(r"\S+", parent.content)
            if not words:
                continue
            step = self.child_chunk_size - self.child_chunk_overlap
            for start in range(0, len(words), step):
                chunk_words = words[start : start + self.child_chunk_size]
                if not chunk_words:
                    break
                children.append(
                    Document(
                        content=" ".join(chunk_words),
                        metadata={"parent_id": parent_id, "child_start": start},
                        doc_id=f"{parent_id}::child::{start}",
                    )
                )
                if start + self.child_chunk_size >= len(words):
                    break
        return children
