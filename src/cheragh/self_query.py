"""
Technique 6 : Self-Query Retrieval — version persistable.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseRetriever, Document, EmbeddingModel, LLMClient, _snapshot_documents, _validate_top_k, cosine_similarity
from .cache import hash_documents, embedder_fingerprint, load_cache, save_cache
from .filters import metadata_matches


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
        allow_unsafe_pickle: bool = False,
    ):
        self.documents = _snapshot_documents(documents)
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.metadata_schema = dict(metadata_schema)
        self._cache_path = cache_path
        self._allow_unsafe_pickle = allow_unsafe_pickle

        self.doc_embeddings: Optional[np.ndarray] = None
        if not self._try_load_cache():
            self.doc_embeddings = embedding_model.embed_documents([d.content for d in documents])
            self._save_cache()

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        top_k = _validate_top_k(top_k)
        cleaned_query, filters = self._parse_query(query)
        mask = np.array(
            [self._match_filters(d.metadata, filters) for d in self.documents], dtype=bool
        )
        if not mask.any():
            return []

        if self.doc_embeddings is None:
            raise ValueError("Self-query document embeddings are unavailable")
        query_vec = self.embedding_model.embed_query(cleaned_query or query)
        scores = cosine_similarity(query_vec, self.doc_embeddings)
        scores = np.where(mask, scores, -np.inf)
        top_idx = np.argsort(-scores, kind="stable")[:top_k]

        results: List[Document] = []
        for i in top_idx:
            if scores[i] == -np.inf:
                break
            doc = self.documents[i]
            results.append(
                Document(
                    content=doc.content,
                    metadata={**deepcopy(doc.metadata), "applied_filters": deepcopy(filters), "cleaned_query": cleaned_query},
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

    # ------------------------------------------------------------------ #
    def _parse_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        schema_str = "\n".join(f"- {k} : {v}" for k, v in self.metadata_schema.items())
        prompt = SELF_QUERY_PROMPT_FR.format(metadata_schema=schema_str, query=query)
        raw = self.llm_client.generate(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("Self-query generator must return a structured JSON object")
        try:
            parsed = json.loads(match.group(0), object_pairs_hook=_unique_json_fields, parse_constant=_invalid_json_constant)
        except ValueError as exc:
            raise ValueError("Self-query generator returned invalid JSON") from exc
        if not isinstance(parsed, dict) or set(parsed) != {"cleaned_query", "filters"}:
            raise ValueError("Self-query requires exactly cleaned_query and filters fields")
        cleaned = parsed["cleaned_query"]
        filters = parsed["filters"]
        if not isinstance(cleaned, str) or not isinstance(filters, dict):
            raise ValueError("Self-query requires a string cleaned_query and object filters")
        for field, condition in filters.items():
            if field not in self.metadata_schema:
                raise ValueError(f"Self-query generated an undeclared metadata field: {field}")
            if isinstance(condition, dict):
                if not condition or set(condition) - {"$eq", "$ne", "$in", "$gte", "$lte", "$gt", "$lt"}:
                    raise ValueError("Self-query generated an unsupported comparison operator")
                if "$in" in condition and not isinstance(condition["$in"], list):
                    raise ValueError("Self-query $in comparison requires a JSON array")
        return cleaned, filters

    @staticmethod
    def _match_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        return metadata_matches(metadata, filters)


def _unique_json_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate structured-query JSON key")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise ValueError(f"invalid structured-query JSON constant: {value}")
