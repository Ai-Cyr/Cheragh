"""
Technique 3 : HyDE – Hypothetical Document Embeddings — version persistable.

Seuls les embeddings des documents sont cachés. Les hypothèses sont
générées à la query (temps réel) et ne sont donc pas cachées ici — à
mettre en cache au niveau applicatif par hash(query) si besoin.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

import numpy as np

from .base import BaseRetriever, Document, EmbeddingModel, LLMClient, _snapshot_document, _snapshot_documents, _validate_top_k, cosine_similarity
from .cache import hash_documents, embedder_fingerprint, load_cache, save_cache


HYDE_PROMPT_FR = """Écris un paragraphe court qui répond de manière factuelle et précise à la question suivante.
Ne dis pas que tu ne sais pas : produis une réponse plausible, détaillée, et rédigée comme si elle provenait d'un document de référence.

Question : {query}

Réponse :"""


class HyDERetriever(BaseRetriever):
    """Average hypothetical-document vectors and rank by inner product (Eq. 7).

    ``include_query`` enables Eq. 8: the query is encoded on the **document**
    side, alongside the generated passages. ``similarity='cosine'`` selects a
    normalized variant when the embedding provider does not normalize itself.
    Generator sampling can be configured via ``generation_kwargs``; the
    default temperature follows the original HyDE experiment (0.7).
    """
    _CACHEABLE_VERSION = 1

    def __init__(
        self,
        documents: List[Document],
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        prompt_template: str = HYDE_PROMPT_FR,
        n_hypotheses: int = 1,
        cache_path: Optional[str] = None,
        allow_unsafe_pickle: bool = False,
        *,
        include_query: bool = False,
        similarity: Literal["dot", "cosine"] = "dot",
        generation_kwargs: dict[str, Any] | None = None,
    ):
        self.n_hypotheses = _validate_top_k(n_hypotheses, name="n_hypotheses")
        if not isinstance(include_query, bool):
            raise TypeError("include_query must be a boolean")
        if similarity not in {"dot", "cosine"}:
            raise ValueError("similarity must be 'dot' or 'cosine'")
        self.documents = _snapshot_documents(documents)
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.include_query = include_query
        self.similarity = similarity
        self.generation_kwargs = dict(generation_kwargs) if generation_kwargs is not None else {"temperature": 0.7}
        self._cache_path = cache_path
        self._allow_unsafe_pickle = allow_unsafe_pickle

        self.doc_embeddings: Optional[np.ndarray] = None
        if not self._try_load_cache():
            self.doc_embeddings = (
                _validated_matrix(embedding_model.embed_documents([d.content for d in self.documents]), len(self.documents))
                if self.documents else np.empty((0, 0))
            )
            self._save_cache()
        if self.documents:
            self.doc_embeddings = _validated_matrix(self.doc_embeddings, len(self.documents))

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        top_k = _validate_top_k(top_k)
        if not self.documents:
            return []
        assert self.doc_embeddings is not None
        prompt = self.prompt_template.format(query=query)
        hypotheses = [self.llm_client.generate(prompt, **self.generation_kwargs) for _ in range(self.n_hypotheses)]
        if any(not isinstance(hypothesis, str) or not hypothesis.strip() for hypothesis in hypotheses):
            raise ValueError("HyDE requires non-empty hypothetical documents")
        texts = hypotheses + ([query] if self.include_query else [])
        hyp_vecs = _validated_matrix(self.embedding_model.embed_documents(texts), len(texts))
        scale = np.max(np.abs(hyp_vecs))
        query_vec = (hyp_vecs / scale).mean(axis=0) * scale if scale else hyp_vecs.mean(axis=0)
        if query_vec.shape[0] != self.doc_embeddings.shape[1]:
            raise ValueError("HyDE hypothesis and source embedding dimensions must match")
        if self.similarity == "cosine":
            scores = cosine_similarity(query_vec, self.doc_embeddings)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                scores = self.doc_embeddings @ query_vec
            if not np.isfinite(scores).all():
                raise ValueError("HyDE inner-product scores exceed the finite numeric range")
        top_idx = np.argsort(-scores, kind="stable")[:top_k]
        results = []
        for i in top_idx:
            result = _snapshot_document(self.documents[i])
            result.score = float(scores[i])
            result.metadata.update({
                "hypothetical_doc_preview": hypotheses[0][:200],
                "hyde_hypothesis_count": len(hypotheses),
                "hyde_include_query": self.include_query,
                "hyde_similarity": self.similarity,
            })
            results.append(result)
        return results

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
            allow_unsafe_pickle=self._allow_unsafe_pickle,
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


def _validated_matrix(values, rows: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("HyDE requires finite embeddings with shape (texts, nonzero dimension)")
    return matrix.copy()
