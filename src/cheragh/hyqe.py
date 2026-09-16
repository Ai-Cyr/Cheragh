"""HyQE candidate reranking and reusable corpus indexing.

Implements Eq. 2 (max) or Eq. 5 (mean) of arXiv:2410.15262:
cos(query, context) + weight * aggregate cos(query, hypothetical question).
Questions depend only on the context and are cached across user queries.
"""
from __future__ import annotations

import json
import math
import re
from typing import Literal, Sequence

import numpy as np

from .base import (
    BaseRetriever, Document, EmbeddingModel, LLMClient, _snapshot_document,
    _snapshot_documents, _validate_top_k, cosine_similarity,
)
from .cache import CacheBackend, MemoryCache, embedder_fingerprint, hash_documents, load_cache, make_cache_key, save_cache
from .cache.decorators import _component_fingerprint
from .reranking import BaseReranker


QUESTION_GENERATION_PROMPT_FR = """Tu reçois un extrait de document. Génère {n_questions} questions distinctes et pertinentes auxquelles CE extrait permet de répondre de façon directe et factuelle.

Règles :
- Les questions doivent être naturelles (comme un utilisateur réel les poserait).
- Elles doivent couvrir les DIFFÉRENTS faits contenus dans l'extrait.
- Varie les formulations (question directe, indirecte, avec "comment", "quel", "pourquoi", etc.).

Réponds UNIQUEMENT avec les questions, une par ligne, sans numérotation ni préambule.

Extrait :
{document}

Questions :"""


class HyQEReranker(BaseReranker):
    """Rerank a first-stage candidate pool using paper HyQE scores.

    Hypothetical questions use ``embed_query``; source contexts use
    ``embed_documents``. This preserves the two roles of asymmetric encoders.
    Long contexts are partitioned into overlapping windows; every window is
    processed, with up to ``n_questions_per_doc`` questions **per window**.
    ``question_weight`` is a tunable lambda, not a universal optimum.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        *,
        n_questions_per_doc: int = 5,
        question_weight: float = 0.5,
        aggregation: Literal["max", "mean"] = "max",
        include_original_content: bool = True,
        max_context_chars: int = 6000,
        prompt_template: str = QUESTION_GENERATION_PROMPT_FR,
        cache: CacheBackend | None = None,
        cache_fingerprint: str | None = None,
    ):
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.n_questions_per_doc = _validate_top_k(n_questions_per_doc, name="n_questions_per_doc")
        if isinstance(question_weight, bool) or not math.isfinite(question_weight) or question_weight < 0:
            raise ValueError("question_weight must be finite and non-negative")
        if aggregation not in {"max", "mean"}:
            raise ValueError("aggregation must be 'max' or 'mean'")
        if not isinstance(include_original_content, bool):
            raise TypeError("include_original_content must be a boolean")
        self.question_weight = float(question_weight)
        self.aggregation = aggregation
        self.include_original_content = include_original_content
        self.max_context_chars = _validate_top_k(max_context_chars, name="max_context_chars")
        self.prompt_template = prompt_template
        self.cache = cache if cache is not None else MemoryCache(max_entries=10_000)
        self.cache_fingerprint = cache_fingerprint

    def _fingerprint(self) -> str:
        return make_cache_key(
            "hyqe-v2", self.n_questions_per_doc, self.max_context_chars, self.prompt_template,
            self.cache_fingerprint or _component_fingerprint(self.llm_client, purpose="llm"),
            _component_fingerprint(self.embedding_model, purpose="embedder"),
        )

    def _generate_questions(self, content: str) -> list[str]:
        questions: list[str] = []
        seen: set[str] = set()
        overlap = min(128, self.max_context_chars // 10)
        for offset in range(0, len(content), self.max_context_chars - overlap):
            part = content[offset:offset + self.max_context_chars]
            raw = self.llm_client.generate(self.prompt_template.format(
                n_questions=self.n_questions_per_doc, document=part,
            ))
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                parsed = None
            lines = parsed if isinstance(parsed, list) else raw.splitlines()
            count = 0
            for line in lines:
                if not isinstance(line, str):
                    continue
                question = re.sub(r"^\s*(?:\d+[.)]\s*|[-•]\s*)", "", line).strip()
                if len(question) < 6 or question.casefold() in seen:
                    continue
                seen.add(question.casefold())
                questions.append(question)
                count += 1
                if count >= self.n_questions_per_doc:
                    break
            if offset + self.max_context_chars >= len(content):
                break
        return questions

    def _features(self, content: str) -> dict:
        key = make_cache_key(self._fingerprint(), content)
        cached = self.cache.get(key, namespace="hyqe-features")
        if cached is not None:
            return cached
        questions = self._generate_questions(content)
        source = _matrix(self.embedding_model.embed_documents([content]), 1)[0]
        vectors = (
            _matrix([self.embedding_model.embed_query(question) for question in questions], len(questions))
            if questions else np.empty((0, source.shape[0]), dtype=float)
        )
        if vectors.shape[1] != source.shape[0]:
            raise ValueError("HyQE query and document embedding dimensions must match")
        state = {"questions": questions, "source": source, "vectors": vectors}
        self.cache.set(key, state, namespace="hyqe-features")
        return state

    def _score(self, query: str, documents: Sequence[Document], features: Sequence[dict], top_k: int) -> list[Document]:
        if not documents:
            return []
        query_vector = _matrix([self.embedding_model.embed_query(query)], 1)[0]
        output: list[Document] = []
        for document, state in zip(documents, features):
            source = _matrix([state["source"]], 1)
            if source.shape[1] != query_vector.shape[0]:
                raise ValueError("HyQE query and document embedding dimensions must match")
            original_score = float(cosine_similarity(query_vector, source)[0])
            questions = state["questions"]
            best_match = ""
            question_score = 0.0
            if questions:
                vectors = _matrix(state["vectors"], len(questions))
                if vectors.shape[1] != query_vector.shape[0]:
                    raise ValueError("HyQE query and question embedding dimensions must match")
                similarities = cosine_similarity(query_vector, vectors)
                best_match = questions[int(np.argmax(similarities))]
                question_score = float(np.max(similarities) if self.aggregation == "max" else np.mean(similarities))
            result = _snapshot_document(document)
            # False is the documented question-only ablation, preserving the
            # older constructor's explicit opt-out from source similarity.
            result.score = (
                original_score + self.question_weight * question_score
                if self.include_original_content else question_score
            )
            result.metadata.update({
                "hyqe_best_match": best_match[:300],
                "hyqe_context_score": original_score,
                "hyqe_question_score": question_score,
                "hyqe_question_count": len(questions),
                "hyqe_aggregation": self.aggregation,
                "hyqe_question_weight": self.question_weight if self.include_original_content else 1.0,
            })
            output.append(result)
        return sorted(output, key=lambda document: document.score, reverse=True)[:top_k]

    def rerank(self, query: str, documents: Sequence[Document], top_k: int = 5) -> list[Document]:
        top_k = _validate_top_k(top_k)
        snapshots = _snapshot_documents(documents)
        return self._score(query, snapshots, [self._features(doc.content) for doc in snapshots], top_k)


class HyQERetriever(BaseRetriever):
    """Precompute HyQE features and score every source document exactly.

    For a large corpus use ``RerankingRetriever(..., reranker=HyQEReranker(...))``
    to apply the paper's inexpensive first-stage/candidate-reranking workflow.
    ``cache_path`` preserves the legacy trusted-pickle interface. New code can
    supply a safe JSON ``CacheBackend`` through ``cache`` instead.
    """

    _CACHEABLE_VERSION = 2

    def __init__(
        self,
        documents: list[Document],
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        n_questions_per_doc: int = 5,
        include_original_content: bool = True,
        cache_path: str | None = None,
        allow_unsafe_pickle: bool = False,
        *,
        question_weight: float = 0.5,
        aggregation: Literal["max", "mean"] = "max",
        max_context_chars: int = 6000,
        prompt_template: str = QUESTION_GENERATION_PROMPT_FR,
        cache: CacheBackend | None = None,
        cache_fingerprint: str | None = None,
    ):
        self.documents = _snapshot_documents(documents)
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.reranker = HyQEReranker(
            embedding_model, llm_client, n_questions_per_doc=n_questions_per_doc,
            include_original_content=include_original_content, question_weight=question_weight,
            aggregation=aggregation, max_context_chars=max_context_chars, prompt_template=prompt_template,
            cache=cache, cache_fingerprint=cache_fingerprint,
        )
        self.n_questions_per_doc = self.reranker.n_questions_per_doc
        self.include_original_content = include_original_content
        fingerprint = self.reranker._fingerprint()
        state = load_cache(
            path=cache_path, expected_class=self.__class__.__name__,
            expected_content_hash=hash_documents(self.documents),
            expected_embedder_fp=embedder_fingerprint(embedding_model), expected_extra_fp=fingerprint,
            allow_unsafe_pickle=allow_unsafe_pickle,
        ) if cache_path else None
        self._features = state["features"] if state is not None else [
            self.reranker._features(doc.content) for doc in self.documents
        ]
        if cache_path and state is None:
            save_cache(
                path=cache_path, retriever_class=self.__class__.__name__,
                content_hash=hash_documents(self.documents), embedder_fp=embedder_fingerprint(embedding_model),
                extra_fingerprint=fingerprint, state={"features": self._features},
            )

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        top_k = _validate_top_k(top_k)
        return self.reranker._score(query, self.documents, self._features, top_k)


def _matrix(values, rows: int) -> np.ndarray:
    vectors = np.asarray(values, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != rows or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise ValueError("HyQE requires finite embeddings with shape (texts, nonzero dimension)")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
