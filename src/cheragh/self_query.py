"""
Technique 6 : Self-Query Retrieval — version persistable.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseRetriever, Document, EmbeddingModel, LLMClient, cosine_similarity
from .cache import hash_documents, embedder_fingerprint, load_cache, save_cache


SELF_QUERY_PROMPT_FR = """Tu es un assistant qui transforme une question en langage naturel en une recherche structurée.

Métadonnées disponibles dans le corpus :
{metadata_schema}

Règles :
- `cleaned_query` : la partie sémantique de la question, sans les contraintes structurées.
- `filters` : dict de contraintes exactes (égalité) sur les métadonnées. Mettre un dict vide {{}} si aucune contrainte.
- Pour les comparaisons numériques/dates, utiliser les opérateurs $gte, $lte, $gt, $lt, $ne, $in.
  Exemples : {{"year": {{"$gte": 2023}}}}, {{"category": {{"$in": ["RH", "Finance"]}}}}

Réponds UNIQUEMENT par un JSON valide, sans préambule ni balise markdown.

Question : {query}

JSON :"""


class SelfQueryRetriever(BaseRetriever):
    _CACHEABLE_VERSION = 1

    def __init__(
        self,
        documents: List[Document],
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        metadata_schema: Dict[str, str],
        cache_path: Optional[str] = None,
    ):
        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.metadata_schema = metadata_schema
        self._cache_path = cache_path

        self.doc_embeddings: Optional[np.ndarray] = None
        if not self._try_load_cache():
            self.doc_embeddings = embedding_model.embed_documents([d.content for d in documents])
            self._save_cache()

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        cleaned_query, filters = self._parse_query(query)
        mask = np.array(
            [self._match_filters(d.metadata, filters) for d in self.documents], dtype=bool
        )
        if not mask.any():
            return []

        query_vec = self.embedding_model.embed_query(cleaned_query or query)
        scores = cosine_similarity(query_vec, self.doc_embeddings)
        scores = np.where(mask, scores, -np.inf)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results: List[Document] = []
        for i in top_idx:
            if scores[i] == -np.inf:
                break
            doc = self.documents[i]
            results.append(
                Document(
                    content=doc.content,
                    metadata={**doc.metadata, "applied_filters": filters, "cleaned_query": cleaned_query},
                    doc_id=doc.doc_id,
                    score=float(scores[i]),
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Cache
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

    # ------------------------------------------------------------------ #
    def _parse_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        schema_str = "\n".join(f"- {k} : {v}" for k, v in self.metadata_schema.items())
        prompt = SELF_QUERY_PROMPT_FR.format(metadata_schema=schema_str, query=query)
        raw = self.llm_client.generate(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return query, {}
        try:
            parsed = json.loads(match.group(0))
            return parsed.get("cleaned_query", query), parsed.get("filters", {}) or {}
        except json.JSONDecodeError:
            return query, {}

    @staticmethod
    def _match_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for field, condition in filters.items():
            value = metadata.get(field)
            if isinstance(condition, dict):
                for op, op_val in condition.items():
                    if op == "$gte" and not (value is not None and value >= op_val):
                        return False
                    elif op == "$lte" and not (value is not None and value <= op_val):
                        return False
                    elif op == "$gt" and not (value is not None and value > op_val):
                        return False
                    elif op == "$lt" and not (value is not None and value < op_val):
                        return False
                    elif op == "$ne" and value == op_val:
                        return False
                    elif op == "$in" and value not in op_val:
                        return False
            else:
                if value != condition:
                    return False
        return True
