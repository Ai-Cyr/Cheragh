"""
Technique 11 : Corrective RAG (CRAG)
=====================================

Yan et al. (2024) — "Corrective Retrieval Augmented Generation".

Principe : ne pas faire aveuglément confiance au retrieval. Un
évaluateur LLM note chaque document retrouvé (Correct / Ambigu /
Incorrect) et déclenche une stratégie corrective si nécessaire :

    - Tous les docs Correct   → on procède normalement
    - Tous les docs Incorrect → on reformule la query et on ré-essaie
                                (fallback : dans la version complète on
                                appelle aussi un web search)
    - Mixte                   → on ne garde que les docs corrects,
                                complétés par un ré-retrieval sur query
                                réécrite si besoin.

Cette technique réduit drastiquement les hallucinations causées par
du retrieval bruité.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional

from .base import BaseRetriever, Document, LLMClient


class DocQuality(str, Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


EVALUATOR_PROMPT_FR = """Évalue si l'extrait ci-dessous contient des informations utiles et pertinentes pour répondre à la question.

Réponds UNIQUEMENT par un seul mot parmi :
- correct     : l'extrait contient des informations directement pertinentes et fiables
- ambiguous   : l'extrait est partiellement lié mais incomplet ou indirect
- incorrect   : l'extrait n'est pas pertinent ou hors sujet

Question : {query}

Extrait :
{document}

Évaluation :"""


QUERY_REWRITE_PROMPT_FR = """La recherche documentaire n'a pas donné de résultats satisfaisants pour la question ci-dessous.
Reformule la question avec des termes différents (synonymes, décomposition, vocabulaire plus technique ou plus général) pour améliorer la recherche.

Réponds UNIQUEMENT par la nouvelle question, sans préambule.

Question originale : {query}

Question reformulée :"""


class CorrectiveRAGRetriever(BaseRetriever):
    """
    Retriever CRAG avec évaluateur et fallback par réécriture de requête.

    Parameters
    ----------
    base_retriever : BaseRetriever
    llm_client : LLMClient
    max_retries : int, default=1
        Nombre maximum de réécritures de query en fallback.
    min_correct : int, default=1
        Nombre minimum de documents "correct" requis pour ne pas déclencher
        la stratégie corrective.
    include_ambiguous : bool, default=True
        Si True, les documents "ambiguous" sont inclus dans les résultats
        (mais moins bien classés que les "correct").
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_client: LLMClient,
        max_retries: int = 1,
        min_correct: int = 1,
        include_ambiguous: bool = True,
    ):
        self.base_retriever = base_retriever
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.min_correct = min_correct
        self.include_ambiguous = include_ambiguous

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        current_query = query
        all_correct: List[Document] = []
        all_ambiguous: List[Document] = []
        attempts = 0

        while attempts <= self.max_retries:
            # 1) Retrieval
            retrieved = self.base_retriever.retrieve(current_query, top_k=top_k * 2)

            # 2) Évaluation de chaque doc
            for doc in retrieved:
                label = self._evaluate(current_query, doc.content)
                doc.metadata["crag_label"] = label.value
                doc.metadata["evaluated_against_query"] = current_query
                if label == DocQuality.CORRECT:
                    all_correct.append(doc)
                elif label == DocQuality.AMBIGUOUS:
                    all_ambiguous.append(doc)
                # incorrect → on jette

            # 3) Critère d'arrêt
            if len(all_correct) >= self.min_correct:
                break

            # 4) Fallback : réécriture de query
            attempts += 1
            if attempts <= self.max_retries:
                current_query = self._rewrite_query(current_query)
                # metadata pour traçabilité
                # (la nouvelle query sera utilisée au prochain tour)

        # 5) Construction du résultat final : correct d'abord, puis ambigus
        results = all_correct[:top_k]
        if self.include_ambiguous and len(results) < top_k:
            remaining = top_k - len(results)
            # Dédoublonner les ambigus déjà dans les correct
            correct_ids = {d.doc_id for d in results if d.doc_id}
            unique_ambiguous = [d for d in all_ambiguous if d.doc_id not in correct_ids]
            results.extend(unique_ambiguous[:remaining])

        return results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _evaluate(self, query: str, document: str) -> DocQuality:
        prompt = EVALUATOR_PROMPT_FR.format(query=query, document=document[:2000])
        raw = self.llm_client.generate(prompt).strip().lower()
        # Extraction robuste du label
        if "incorrect" in raw:
            return DocQuality.INCORRECT
        if "ambiguous" in raw or "ambigu" in raw:
            return DocQuality.AMBIGUOUS
        if "correct" in raw:
            return DocQuality.CORRECT
        # Par défaut, on considère ambigu plutôt que de jeter
        return DocQuality.AMBIGUOUS

    def _rewrite_query(self, query: str) -> str:
        prompt = QUERY_REWRITE_PROMPT_FR.format(query=query)
        raw = self.llm_client.generate(prompt).strip()
        return raw.split("\n")[0].strip() or query
