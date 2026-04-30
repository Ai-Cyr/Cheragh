"""
Technique 8 : Query Decomposition (Multi-Hop)
==============================================

Problème : une question peut nécessiter plusieurs faits issus de
documents différents. Ex. "Quel est le CA 2024 des filiales acquises
depuis 2022 ?" → il faut d'abord identifier les filiales, puis trouver
leurs CA.

Solution : le LLM décompose la question en sous-questions atomiques
(multi-hop). On retrieve pour chacune, on déduplique, et on renvoie
l'union — idéal pour alimenter un générateur capable de synthétiser.

Différence avec RAG-Fusion :
    - RAG-Fusion = reformulations DE LA MÊME question (paraphrases).
    - Query Decomposition = SOUS-questions qui couvrent des aspects
      différents et devront être combinées pour répondre.
"""
from __future__ import annotations

import re
from typing import Dict, List

from .base import BaseRetriever, Document, LLMClient


DECOMPOSITION_PROMPT_FR = """Décompose la question complexe suivante en {max_subquestions} sous-questions atomiques, indépendantes et simples à répondre.

Chaque sous-question doit pouvoir se répondre avec UN seul fait ou UN seul document.
Si la question d'origine est déjà simple, renvoie-la telle quelle (une seule ligne).

Réponds UNIQUEMENT avec les sous-questions, une par ligne, sans numérotation ni préambule.

Question complexe : {query}

Sous-questions :"""


class QueryDecompositionRetriever(BaseRetriever):
    """
    Retriever multi-hop : décompose la question puis agrège les résultats.

    Parameters
    ----------
    base_retriever : BaseRetriever
    llm_client : LLMClient
    max_subquestions : int, default=4
    per_subquestion_top_k : int, default=3
        Top-k récupéré pour chaque sous-question avant dédoublonnage.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_client: LLMClient,
        max_subquestions: int = 4,
        per_subquestion_top_k: int = 3,
    ):
        self.base_retriever = base_retriever
        self.llm_client = llm_client
        self.max_subquestions = max_subquestions
        self.per_subquestion_top_k = per_subquestion_top_k

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        # 1) Décomposition
        subquestions = self._decompose(query)

        # 2) Retrieval par sous-question + agrégation du meilleur score
        seen: Dict[str, Document] = {}
        subquestion_map: Dict[str, List[str]] = {}  # doc_key -> [sub-questions ayant ramené ce doc]

        for sub in subquestions:
            hits = self.base_retriever.retrieve(sub, top_k=self.per_subquestion_top_k)
            for doc in hits:
                key = doc.doc_id if doc.doc_id is not None else f"content::{hash(doc.content)}"
                # Garder le meilleur score
                if key not in seen or (doc.score or 0) > (seen[key].score or 0):
                    seen[key] = doc
                subquestion_map.setdefault(key, []).append(sub)

        # 3) Tri par score décroissant
        merged = sorted(seen.values(), key=lambda d: d.score or 0, reverse=True)

        # 4) On enrichit les métadonnées avec les sous-questions ayant matché
        results: List[Document] = []
        for doc in merged[:top_k]:
            key = doc.doc_id if doc.doc_id is not None else f"content::{hash(doc.content)}"
            results.append(
                Document(
                    content=doc.content,
                    metadata={
                        **doc.metadata,
                        "matched_subquestions": subquestion_map.get(key, []),
                        "decomposed_from": query,
                    },
                    doc_id=doc.doc_id,
                    score=doc.score,
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _decompose(self, query: str) -> List[str]:
        prompt = DECOMPOSITION_PROMPT_FR.format(
            max_subquestions=self.max_subquestions, query=query
        )
        raw = self.llm_client.generate(prompt)
        lines = [re.sub(r"^[\d\.\)\-\s•]+", "", line).strip() for line in raw.split("\n")]
        subs = [line for line in lines if len(line) > 3][: self.max_subquestions]
        # Toujours inclure la question d'origine pour ne pas perdre de contexte
        if query not in subs:
            subs = [query] + subs
        return subs
