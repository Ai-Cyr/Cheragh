"""Technique 1: hybrid sparse+dense retrieval with optional persistence."""
from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

from .base import BaseRetriever, Document, EmbeddingModel, _numpy, cosine_similarity, min_max_normalize
from .cache import embedder_fingerprint, hash_documents, load_cache, save_cache
from .filters import metadata_matches
from .tokenization import RetrievalTokenizer

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


class HybridSearchRetriever(BaseRetriever):
    """Hybrid BM25 + dense retriever, with disk cache support.

    ``rank-bm25`` is used when available. Otherwise the package falls back to a
    small built-in BM25 implementation so the retriever remains usable in a
    standard Python environment with only NumPy installed.

    v0.9 adds a reusable unicode tokenizer and richer metadata filters. Filters
    can be supplied at construction time and refined per query with
    ``retrieve(..., filters={...})``.
    """

    _CACHEABLE_VERSION = 4

    def __init__(
        self,
        documents: List[Document],
        embedding_model: EmbeddingModel,
        alpha: float = 0.5,
        cache_path: Optional[str] = None,
        filters: Optional[dict] = None,
        tokenizer: RetrievalTokenizer | None = None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.documents = documents
        self.embedding_model = embedding_model
        self.alpha = alpha
        self.filters = filters
        self.tokenizer = tokenizer or RetrievalTokenizer()
        self._cache_path = cache_path
        self.bm25 = None
        self._tokenized_corpus = None
        self.doc_embeddings: Optional[np.ndarray] = None

        if not self._try_load_cache():
            self._build_index()
            self._save_cache()

    def retrieve(self, query: str, top_k: int = 5, filters: Optional[dict[str, Any]] = None) -> List[Document]:
        np = _numpy()
        if not self.documents:
            return []
        effective_filters = _merge_filters(self.filters, filters)
        candidate_indices = self._matching_indices(effective_filters)
        if not candidate_indices:
            return []
        bm25_scores = np.asarray(self.bm25.get_scores(self._tokenize(query)), dtype=float)
        query_vec = self.embedding_model.embed_query(query)
        dense_scores = cosine_similarity(query_vec, self.doc_embeddings)

        bm25_norm = min_max_normalize(bm25_scores)
        dense_norm = min_max_normalize(dense_scores)
        hybrid_scores = self.alpha * dense_norm + (1.0 - self.alpha) * bm25_norm

        candidate_scores = hybrid_scores[candidate_indices]
        order = np.argsort(candidate_scores)[::-1][:top_k]
        results: List[Document] = []
        for local_i in order:
            i = candidate_indices[int(local_i)]
            doc = self.documents[i]
            results.append(
                Document(
                    content=doc.content,
                    metadata={
                        **doc.metadata,
                        "bm25_score": float(bm25_norm[i]),
                        "dense_score": float(dense_norm[i]),
                    },
                    doc_id=doc.doc_id,
                    score=float(hybrid_scores[i]),
                )
            )
        return results

    def _extra_fp(self) -> str:
        return f"v={self._CACHEABLE_VERSION};alpha={self.alpha};tokenizer={self.tokenizer!r}"

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
        tokenized = state.get("tokenized_corpus")
        if tokenized is not None:
            self._tokenized_corpus = tokenized
            self.bm25 = _build_bm25(tokenized)
        else:
            # Backward compatibility for legacy caches that pickled the BM25 object.
            self.bm25 = state["bm25"]
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
            state={"tokenized_corpus": self._tokenized_corpus, "doc_embeddings": self.doc_embeddings},
        )

    def _build_index(self) -> None:
        tokenized_corpus = [self._tokenize(d.content) for d in self.documents]
        self._tokenized_corpus = tokenized_corpus
        self.bm25 = _build_bm25(tokenized_corpus)
        self.doc_embeddings = self.embedding_model.embed_documents([d.content for d in self.documents])

    def _matching_indices(self, filters: Optional[dict]) -> list[int]:
        if not filters:
            return list(range(len(self.documents)))
        return [idx for idx, doc in enumerate(self.documents) if metadata_matches(doc.metadata, filters)]

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)


class _SimpleBM25:
    """Minimal BM25Okapi-compatible scorer used as a dependency-free fallback."""

    def __init__(self, tokenized_corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        np = _numpy()
        self.k1 = k1
        self.b = b
        self.corpus = [list(doc) for doc in tokenized_corpus]
        self.doc_freqs = [Counter(doc) for doc in self.corpus]
        self.doc_len = np.asarray([len(doc) for doc in self.corpus], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.n_docs = len(self.corpus)
        df = Counter()
        for doc in self.corpus:
            df.update(set(doc))
        self.idf = {term: math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        np = _numpy()
        if self.n_docs == 0:
            return np.array([])
        scores = np.zeros(self.n_docs, dtype=float)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                freq = freqs.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0))
                scores[i] += idf * (freq * (self.k1 + 1.0)) / denom
        return scores


def _build_bm25(tokenized_corpus: Sequence[Sequence[str]]):
    try:  # pragma: no cover - depends on optional dependency
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except ImportError:
        return _SimpleBM25(tokenized_corpus)


def _merge_filters(base: Optional[dict[str, Any]], extra: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not base:
        return extra
    if not extra:
        return base
    merged = dict(base)
    merged.update(extra)
    return merged
