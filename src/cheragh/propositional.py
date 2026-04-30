"""
Technique 18 : Propositional Indexing — version persistable.

Comme HyQE, Propositional est coûteuse à l'indexation (LLM extrait les
propositions de chaque chunk). Le cache sauvegarde les propositions
générées ET leurs embeddings.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

from .base import BaseRetriever, Document, EmbeddingModel, LLMClient, cosine_similarity
from .cache import hash_documents, embedder_fingerprint, load_cache, save_cache


PROPOSITION_EXTRACTION_PROMPT_FR = """Décompose l'extrait ci-dessous en propositions atomiques.

Une proposition atomique est un énoncé court (une phrase simple) qui :
- exprime UN SEUL fait ou UNE SEULE règle,
- est AUTONOME (compréhensible sans lire le reste de l'extrait : remplace "il/elle/cela/ce dernier" par la valeur explicite),
- est factuelle (pas de question, pas de commentaire).

Réponds UNIQUEMENT avec les propositions, une par ligne, sans numérotation ni préambule.

Extrait :
{document}

Propositions :"""


class PropositionalRetriever(BaseRetriever):
    _CACHEABLE_VERSION = 1

    def __init__(
        self,
        documents: List[Document],
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        return_propositions: bool = False,
        max_propositions_per_doc: int = 20,
        cache_path: Optional[str] = None,
    ):
        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.return_propositions = return_propositions
        self.max_propositions_per_doc = max_propositions_per_doc
        self._cache_path = cache_path

        self._propositions: List[str] = []
        self._prop_to_doc: List[int] = []
        self._prop_embeddings: Optional[np.ndarray] = None

        if not self._try_load_cache():
            self._build_index()
            self._save_cache()

        if not self._propositions:
            raise ValueError("Aucune proposition extraite. Vérifier le LLM / les documents.")

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        query_vec = self.embedding_model.embed_query(query)
        scores = cosine_similarity(query_vec, self._prop_embeddings)

        if self.return_propositions:
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [
                Document(
                    content=self._propositions[i],
                    metadata={
                        **self.documents[self._prop_to_doc[i]].metadata,
                        "source_doc_id": self.documents[self._prop_to_doc[i]].doc_id,
                    },
                    doc_id=f"prop::{i}",
                    score=float(scores[i]),
                )
                for i in top_idx
            ]

        fetch = max(top_k * 5, 30)
        top_idx = np.argsort(scores)[::-1][:fetch]

        best_per_doc: Dict[int, float] = {}
        best_match: Dict[int, str] = {}
        order: List[int] = []
        for i in top_idx:
            di = self._prop_to_doc[i]
            s = float(scores[i])
            if di not in best_per_doc or s > best_per_doc[di]:
                best_per_doc[di] = s
                best_match[di] = self._propositions[i]
                if di not in order:
                    order.append(di)

        ordered = sorted(order, key=lambda di: best_per_doc[di], reverse=True)
        return [
            Document(
                content=self.documents[di].content,
                metadata={**self.documents[di].metadata, "matched_proposition": best_match[di]},
                doc_id=self.documents[di].doc_id,
                score=best_per_doc[di],
            )
            for di in ordered[:top_k]
        ]

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #
    def _extra_fp(self) -> str:
        # return_propositions est query-time, pas besoin dans le fingerprint
        return f"v={self._CACHEABLE_VERSION};max_props={self.max_propositions_per_doc}"

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
        self._propositions = state["propositions"]
        self._prop_to_doc = state["prop_to_doc"]
        self._prop_embeddings = state["prop_embeddings"]
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
            state={
                "propositions": self._propositions,   # inclut les props LLM-générées
                "prop_to_doc": self._prop_to_doc,
                "prop_embeddings": self._prop_embeddings,
            },
        )

    # ------------------------------------------------------------------ #
    def _build_index(self) -> None:
        for di, doc in enumerate(self.documents):
            props = self._extract_propositions(doc.content)
            for p in props[: self.max_propositions_per_doc]:
                self._propositions.append(p)
                self._prop_to_doc.append(di)
        if self._propositions:
            self._prop_embeddings = self.embedding_model.embed_documents(self._propositions)

    def _extract_propositions(self, content: str) -> List[str]:
        prompt = PROPOSITION_EXTRACTION_PROMPT_FR.format(document=content[:3000])
        raw = self.llm_client.generate(prompt)
        lines = [re.sub(r"^[\d\.\)\-\s•]+", "", line).strip() for line in raw.split("\n")]
        return [
            line for line in lines
            if len(line) > 10 and not line.endswith("?") and len(line.split()) >= 3
        ]
