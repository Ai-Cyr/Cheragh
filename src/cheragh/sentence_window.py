"""
Technique 12 : Sentence Window Retrieval — version persistable.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseRetriever, Document, EmbeddingModel, cosine_similarity
from .cache import hash_documents, embedder_fingerprint, load_cache, save_cache


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+(?=[A-ZÀ-Ÿ0-9])")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) > 1]


class SentenceWindowRetriever(BaseRetriever):
    _CACHEABLE_VERSION = 1

    def __init__(
        self,
        documents: List[Document],
        embedding_model: EmbeddingModel,
        window_size: int = 3,
        cache_path: Optional[str] = None,
    ):
        if window_size < 0:
            raise ValueError("window_size doit être >= 0.")

        self.embedding_model = embedding_model
        self.window_size = window_size
        self._cache_path = cache_path
        self._documents = documents  # référence pour hash uniquement

        # State à construire ou à recharger
        self.sentences: List[str] = []
        self.sentence_locations: List[Tuple[int, int]] = []
        self.doc_sentences: Dict[int, List[str]] = {}
        self.doc_refs: Dict[int, Document] = {}
        self.sentence_embeddings: Optional[np.ndarray] = None

        if not self._try_load_cache():
            self._build_index()
            self._save_cache()

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        query_vec = self.embedding_model.embed_query(query)
        scores = cosine_similarity(query_vec, self.sentence_embeddings)
        fetch = max(top_k * 4, 20)
        top_sentence_idx = np.argsort(scores)[::-1][:fetch]

        seen_windows: Dict[Tuple[int, int, int], Document] = {}
        for sent_idx in top_sentence_idx:
            di, si = self.sentence_locations[sent_idx]
            all_sents = self.doc_sentences[di]
            start = max(0, si - self.window_size)
            end = min(len(all_sents), si + self.window_size + 1)
            key = (di, start, end)
            if key in seen_windows:
                if float(scores[sent_idx]) > (seen_windows[key].score or -np.inf):
                    seen_windows[key].score = float(scores[sent_idx])
                continue

            window_text = " ".join(all_sents[start:end])
            source_doc = self.doc_refs[di]
            seen_windows[key] = Document(
                content=window_text,
                metadata={
                    **source_doc.metadata,
                    "matched_sentence": all_sents[si],
                    "window_start": start,
                    "window_end": end,
                    "total_sentences_in_doc": len(all_sents),
                },
                doc_id=f"{source_doc.doc_id or di}::win::{start}-{end}",
                score=float(scores[sent_idx]),
            )
            if len(seen_windows) >= top_k:
                break

        return sorted(seen_windows.values(), key=lambda d: d.score or 0, reverse=True)[:top_k]

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #
    def _extra_fp(self) -> str:
        return f"v={self._CACHEABLE_VERSION};window={self.window_size}"

    def _try_load_cache(self) -> bool:
        if not self._cache_path:
            return False
        state = load_cache(
            path=self._cache_path,
            expected_class=self.__class__.__name__,
            expected_content_hash=hash_documents(self._documents),
            expected_embedder_fp=embedder_fingerprint(self.embedding_model),
            expected_extra_fp=self._extra_fp(),
        )
        if state is None:
            return False
        self.sentences = state["sentences"]
        self.sentence_locations = state["sentence_locations"]
        self.doc_sentences = state["doc_sentences"]
        self.doc_refs = state["doc_refs"]
        self.sentence_embeddings = state["sentence_embeddings"]
        return True

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        save_cache(
            path=self._cache_path,
            retriever_class=self.__class__.__name__,
            content_hash=hash_documents(self._documents),
            embedder_fp=embedder_fingerprint(self.embedding_model),
            extra_fingerprint=self._extra_fp(),
            state={
                "sentences": self.sentences,
                "sentence_locations": self.sentence_locations,
                "doc_sentences": self.doc_sentences,
                "doc_refs": self.doc_refs,
                "sentence_embeddings": self.sentence_embeddings,
            },
        )

    # ------------------------------------------------------------------ #
    def _build_index(self) -> None:
        for di, doc in enumerate(self._documents):
            sents = split_sentences(doc.content)
            if not sents:
                continue
            self.doc_sentences[di] = sents
            self.doc_refs[di] = doc
            for si, s in enumerate(sents):
                self.sentences.append(s)
                self.sentence_locations.append((di, si))
        self.sentence_embeddings = self.embedding_model.embed_documents(self.sentences)
