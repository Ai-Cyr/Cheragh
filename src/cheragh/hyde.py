"""
Technique 3 : HyDE – Hypothetical Document Embeddings — version persistable.

Seuls les embeddings des documents sont cachés. Les hypothèses sont
générées à la query (temps réel) et ne sont donc pas cachées ici — à
mettre en cache au niveau applicatif par hash(query) si besoin.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import BaseRetriever, Document, EmbeddingModel, LLMClient, cosine_similarity
from .cache import hash_documents, embedder_fingerprint, load_cache, save_cache


HYDE_PROMPT_FR = """Écris un paragraphe court qui répond de manière factuelle et précise à la question suivante.
Ne dis pas que tu ne sais pas : produis une réponse plausible, détaillée, et rédigée comme si elle provenait d'un document de référence.

Question : {query}

Réponse :"""


class HyDERetriever(BaseRetriever):
    _CACHEABLE_VERSION = 1

    def __init__(
        self,
        documents: List[Document],
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        prompt_template: str = HYDE_PROMPT_FR,
        n_hypotheses: int = 1,
        cache_path: Optional[str] = None,
    ):
        if n_hypotheses < 1:
            raise ValueError("n_hypotheses doit être >= 1.")

        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.n_hypotheses = n_hypotheses
        self._cache_path = cache_path

        self.doc_embeddings: Optional[np.ndarray] = None
        if not self._try_load_cache():
            self.doc_embeddings = embedding_model.embed_documents([d.content for d in documents])
            self._save_cache()

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        prompt = self.prompt_template.format(query=query)
        hypotheses = [self.llm_client.generate(prompt) for _ in range(self.n_hypotheses)]

        hyp_vecs = self.embedding_model.embed_documents(hypotheses)
        query_vec = hyp_vecs.mean(axis=0)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores = cosine_similarity(query_vec, self.doc_embeddings)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            Document(
                content=self.documents[i].content,
                metadata={**self.documents[i].metadata, "hypothetical_doc_preview": hypotheses[0][:200]},
                doc_id=self.documents[i].doc_id,
                score=float(scores[i]),
            )
            for i in top_idx
        ]

    # ------------------------------------------------------------------ #
    def _extra_fp(self) -> str:
        return f"v={self._CACHEABLE_VERSION}"

    def _try_load_cache(self) -> bool:
        if not self._cache_path:
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
        if not self._cache_path:
            return
        save_cache(
            path=self._cache_path,
            retriever_class=self.__class__.__name__,
            content_hash=hash_documents(self.documents),
            embedder_fp=embedder_fingerprint(self.embedding_model),
            extra_fingerprint=self._extra_fp(),
            state={"doc_embeddings": self.doc_embeddings},
        )
