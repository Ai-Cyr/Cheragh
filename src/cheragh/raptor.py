"""
Technique 16 : RAPTOR — version persistable.

RAPTOR est la technique la plus coûteuse à construire : clustering +
appel LLM par cluster, à chaque niveau. Le cache sauvegarde l'arbre
complet (feuilles + résumés de tous les niveaux) et les embeddings.
"""
from __future__ import annotations

from copy import copy
from typing import Any, List, Mapping, Optional

import numpy as np

from .base import (BaseRetriever, Document, EmbeddingModel, LLMClient, _snapshot_document,
                   _snapshot_documents, _validate_non_negative_int, _validate_top_k, cosine_similarity)
from .cache import hash_documents, embedder_fingerprint, load_cache, save_cache
from .security.access_control import AccessPolicy, Principal


CLUSTER_SUMMARY_PROMPT_FR = """Tu reçois plusieurs extraits documentaires appartenant au même thème.
Produis un résumé synthétique de l'ensemble (150-250 mots) qui capture les idées-clés,
les faits chiffrés importants, et les relations entre les éléments.

Écris le résumé comme un paragraphe dense et informatif — il servira de point d'entrée
pour retrouver ce thème lors de recherches futures.

Extraits :
{documents}

Résumé :"""


class RAPTORRetriever(BaseRetriever):
    _CACHEABLE_VERSION = 2

    def __init__(
        self,
        documents: List[Document],
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        n_levels: int = 3,
        branching_factor: int = 5,
        min_cluster_size: int = 2,
        random_state: int = 42,
        cache_path: Optional[str] = None,
        allow_unsafe_pickle: bool = False,
    ):
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.n_levels = _validate_non_negative_int(n_levels, name="n_levels")
        self.branching_factor = _validate_top_k(branching_factor, name="branching_factor")
        self.min_cluster_size = _validate_top_k(min_cluster_size, name="min_cluster_size")
        self.random_state = random_state
        self._cache_path = cache_path
        self._allow_unsafe_pickle = allow_unsafe_pickle
        self._input_documents = _snapshot_documents(documents)
        explicit_ids = [doc.doc_id for doc in self._input_documents if doc.doc_id]
        if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in explicit_ids):
            raise ValueError("RAPTOR document IDs must be non-empty strings")
        if len(set(explicit_ids)) != len(explicit_ids):
            raise ValueError("RAPTOR document IDs must be unique")
        used_ids = set(explicit_ids)
        for index, document in enumerate(self._input_documents):
            if not document.doc_id:
                candidate = f"raptor-source-{index}"
                while candidate in used_ids:
                    candidate += "-"
                document.doc_id = candidate
                used_ids.add(candidate)

        self.all_nodes: List[Document] = []
        self._all_embeddings: Optional[np.ndarray] = None

        if not self._try_load_cache():
            self._build_tree()
            self._save_cache()

    def for_principal(self, principal: Principal | Mapping[str, Any] | None,
                      policy: AccessPolicy | None = None) -> "RAPTORRetriever":
        """Exclude summaries unless every original source is authorized."""
        policy = policy if policy is not None else AccessPolicy()
        allowed = {doc.doc_id for doc in self._input_documents
                   if policy.authorize(_snapshot_document(doc), principal).allowed}
        selected = [index for index, doc in enumerate(self.all_nodes)
                    if doc.metadata.get("source_doc_ids")
                    and set(doc.metadata["source_doc_ids"]) <= allowed]
        scoped = copy(self)
        scoped.all_nodes = [_snapshot_document(self.all_nodes[index]) for index in selected]
        scoped._input_documents = [_snapshot_document(doc) for doc in self._input_documents if doc.doc_id in allowed]
        scoped._all_embeddings = (None if self._all_embeddings is None else self._all_embeddings[selected].copy())
        return scoped

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        top_k = _validate_top_k(top_k)
        if not self.all_nodes:
            return []
        assert self._all_embeddings is not None
        query_vec = self.embedding_model.embed_query(query)
        scores = cosine_similarity(query_vec, self._all_embeddings)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = [_snapshot_document(self.all_nodes[index]) for index in top_idx]
        for document, index in zip(results, top_idx):
            document.score = float(scores[index])
        return results

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #
    def _extra_fp(self) -> str:
        return (
            f"v={self._CACHEABLE_VERSION};"
            f"levels={self.n_levels};branch={self.branching_factor};"
            f"min_cluster={self.min_cluster_size};seed={self.random_state}"
        )

    def _try_load_cache(self) -> bool:
        if not self._cache_path:
            return False
        state = load_cache(
            path=self._cache_path,
            expected_class=self.__class__.__name__,
            expected_content_hash=hash_documents(self._input_documents),
            expected_embedder_fp=embedder_fingerprint(self.embedding_model),
            expected_extra_fp=self._extra_fp(),
            allow_unsafe_pickle=self._allow_unsafe_pickle,
        )
        if state is None:
            return False
        self.all_nodes = _snapshot_documents(state["all_nodes"])
        self._all_embeddings = self._validated_embeddings(state["all_embeddings"], len(self.all_nodes))
        return True

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        save_cache(
            path=self._cache_path,
            retriever_class=self.__class__.__name__,
            content_hash=hash_documents(self._input_documents),
            embedder_fp=embedder_fingerprint(self.embedding_model),
            extra_fingerprint=self._extra_fp(),
            state={
                "all_nodes": self.all_nodes,
                "all_embeddings": self._all_embeddings,
            },
        )

    # ------------------------------------------------------------------ #
    def _build_tree(self) -> None:
        self.all_nodes = _snapshot_documents(self._input_documents)
        for d in self.all_nodes:
            d.metadata.update({"raptor_level": 0, "source_doc_ids": [d.doc_id]})

        current_level_docs = list(self.all_nodes)
        for level in range(1, self.n_levels + 1):
            if len(current_level_docs) < self.min_cluster_size * 2:
                break
            summary_docs = self._build_level(current_level_docs, level)
            if not summary_docs:
                break
            self.all_nodes.extend(summary_docs)
            current_level_docs = summary_docs

        self._all_embeddings = (self._validated_embeddings(self.embedding_model.embed_documents(
            [d.content for d in self.all_nodes]), len(self.all_nodes)) if self.all_nodes else np.empty((0, 0)))

    @staticmethod
    def _validated_embeddings(values: Any, count: int) -> np.ndarray:
        matrix = np.array(values, dtype=float, copy=True)
        if (matrix.ndim != 2 or matrix.shape[0] != count or (count and matrix.shape[1] == 0)
                or not np.isfinite(matrix).all()):
            raise ValueError("RAPTOR requires finite embeddings with one row per document")
        return matrix

    def _build_level(self, docs: List[Document], level: int) -> List[Document]:
        from sklearn.cluster import KMeans

        embeddings = self._validated_embeddings(self.embedding_model.embed_documents([d.content for d in docs]), len(docs))
        n_clusters = min(self.branching_factor, max(2, len(docs) // self.min_cluster_size))

        kmeans = KMeans(
            n_clusters=n_clusters, random_state=self.random_state, n_init=10
        ).fit(embeddings)
        labels = kmeans.labels_

        summary_docs: List[Document] = []
        for cluster_id in range(n_clusters):
            cluster_docs = [d for d, lbl in zip(docs, labels) if lbl == cluster_id]
            if len(cluster_docs) < self.min_cluster_size:
                continue
            summary_text = self._summarize_cluster(cluster_docs)
            if not summary_text:
                continue
            child_ids = [d.doc_id for d in cluster_docs if d.doc_id]
            source_ids = sorted({source_id for doc in cluster_docs for source_id in doc.metadata["source_doc_ids"]})
            summary_id = f"raptor::L{level}::C{cluster_id}"
            used_ids = {doc.doc_id for doc in self.all_nodes}
            while summary_id in used_ids:
                summary_id += "-"
            summary_docs.append(
                Document(
                    content=summary_text,
                    metadata={
                        "raptor_level": level,
                        "raptor_cluster_id": cluster_id,
                        "raptor_child_ids": child_ids,
                        "raptor_cluster_size": len(cluster_docs),
                        "source_doc_ids": source_ids,
                    },
                    doc_id=summary_id,
                )
            )
        return summary_docs

    def _summarize_cluster(self, cluster_docs: List[Document]) -> str:
        joined = "\n\n---\n\n".join(
            f"[extrait {i+1}]\n{d.content}" for i, d in enumerate(cluster_docs)
        )
        if len(joined) > 8000:
            joined = joined[:8000] + "\n... (tronqué)"
        prompt = CLUSTER_SUMMARY_PROMPT_FR.format(documents=joined)
        return self.llm_client.generate(prompt).strip()
