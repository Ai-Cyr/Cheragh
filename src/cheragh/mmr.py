"""
Technique 10 : MMR — version persistable (mode autonome uniquement).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import BaseRetriever, Document, EmbeddingModel, cosine_similarity
from .cache import hash_documents, embedder_fingerprint, load_cache, save_cache


class MMRRetriever(BaseRetriever):
    _CACHEABLE_VERSION = 1

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        documents: Optional[List[Document]] = None,
        base_retriever: Optional[BaseRetriever] = None,
        lambda_mult: float = 0.5,
        fetch_k: int = 30,
        cache_path: Optional[str] = None,
    ):
        if (documents is None) == (base_retriever is None):
            raise ValueError("Spécifier exactement un de `documents` ou `base_retriever`.")
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("lambda_mult doit être dans [0, 1].")

        self.embedding_model = embedding_model
        self.documents = documents
        self.base_retriever = base_retriever
        self.lambda_mult = lambda_mult
        self.fetch_k = fetch_k
        self._cache_path = cache_path

        # Le cache n'a de sens qu'en mode autonome (avec corpus propre)
        self.doc_embeddings: Optional[np.ndarray] = None
        if documents is not None:
            if not self._try_load_cache():
                self.doc_embeddings = embedding_model.embed_documents([d.content for d in documents])
                self._save_cache()

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        query_vec = self.embedding_model.embed_query(query)

        if self.base_retriever is not None:
            candidates = self.base_retriever.retrieve(query, top_k=self.fetch_k)
            if not candidates:
                return []
            cand_embeddings = self.embedding_model.embed_documents([d.content for d in candidates])
        else:
            scores = cosine_similarity(query_vec, self.doc_embeddings)
            top_idx = np.argsort(scores)[::-1][: self.fetch_k]
            candidates = [self.documents[i] for i in top_idx]
            cand_embeddings = self.doc_embeddings[top_idx]

        relevance = cosine_similarity(query_vec, cand_embeddings)

        selected_idx: List[int] = []
        remaining = list(range(len(candidates)))
        while remaining and len(selected_idx) < top_k:
            if not selected_idx:
                best = max(remaining, key=lambda i: float(relevance[i]))
            else:
                selected_embs = cand_embeddings[selected_idx]
                best_score = -np.inf
                best = remaining[0]
                for i in remaining:
                    sim_to_selected = cand_embeddings[i] @ selected_embs.T
                    max_redundancy = float(np.max(sim_to_selected))
                    mmr = self.lambda_mult * float(relevance[i]) - (1 - self.lambda_mult) * max_redundancy
                    if mmr > best_score:
                        best_score = mmr
                        best = i
            selected_idx.append(best)
            remaining.remove(best)

        return [
            Document(
                content=candidates[i].content,
                metadata={
                    **candidates[i].metadata,
                    "mmr_rank": rank,
                    "relevance_to_query": float(relevance[i]),
                },
                doc_id=candidates[i].doc_id,
                score=float(relevance[i]),
            )
            for rank, i in enumerate(selected_idx)
        ]

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #
    def _extra_fp(self) -> str:
        # lambda_mult et fetch_k n'affectent pas les embeddings cachés → non inclus
        return f"v={self._CACHEABLE_VERSION};mode=standalone"

    def _try_load_cache(self) -> bool:
        if not self._cache_path or self.documents is None:
            return False
        state = load_cache(
            path=self._cache_path,
            expected_class=self.__class__.__name__,
            expected_content_hash=hash_documents(self.documents),
            expected_embedder_fp=embedder_fingerprint(self.embedding_model),
            expected_extra_fp=self._extra_fp(),
        )
        if state is None:
            return False
        self.doc_embeddings = state["doc_embeddings"]
        return True

    def _save_cache(self) -> None:
        if not self._cache_path or self.documents is None:
            return
        save_cache(
            path=self._cache_path,
            retriever_class=self.__class__.__name__,
            content_hash=hash_documents(self.documents),
            embedder_fp=embedder_fingerprint(self.embedding_model),
            extra_fingerprint=self._extra_fp(),
            state={"doc_embeddings": self.doc_embeddings},
        )
