"""
Technique 4 : RAG-Fusion (Multi-Query + Reciprocal Rank Fusion)
================================================================

Raja et al. — extension de Multi-Query Retrieval (LangChain).

Principe :
    1. Le LLM reformule la question en N variantes (paraphrases,
       sous-questions, angles différents).
    2. On exécute un retrieval par variante.
    3. On fusionne les listes de résultats avec Reciprocal Rank Fusion
       (RRF), qui ne dépend pas des scores absolus (robuste aux
       différentes échelles).

Formule RRF : score(d) = Σ 1 / (k + rank_i(d))  — k=60 par défaut.
"""
from __future__ import annotations

import re
from typing import List

from .base import BaseRetriever, Document, LLMClient, _validate_top_k
from .reranking import ReciprocalRankFusionReranker


MULTI_QUERY_PROMPT_FR = """Tu es un assistant spécialisé en recherche d'information.
Génère {n_queries} reformulations distinctes et complémentaires de la question suivante
pour améliorer la recherche dans une base documentaire. Chaque reformulation doit couvrir
un angle différent (synonymes, sous-questions, termes techniques, formulation plus large,
formulation plus précise).

Retourne UNIQUEMENT les reformulations, une par ligne, sans numérotation ni préambule.

Question originale : {query}
"""


class RAGFusionRetriever(BaseRetriever):
    """
    RAG-Fusion : génération multi-requêtes + fusion RRF.

    Parameters
    ----------
    base_retriever : BaseRetriever
        Retriever utilisé pour chaque sous-requête.
    llm_client : LLMClient
        LLM pour générer les reformulations.
    n_queries : int, default=4
        Nombre de reformulations à générer (la question d'origine est incluse en plus).
    rrf_k : int, default=60
        Constante RRF (60 est la valeur standard de la littérature).
    per_query_top_k : int, default=10
        Top-k récupéré pour chaque sous-requête avant fusion.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_client: LLMClient,
        n_queries: int = 4,
        rrf_k: int = 60,
        per_query_top_k: int = 10,
    ):
        self.base_retriever = base_retriever
        self.llm_client = llm_client
        self.n_queries = _validate_top_k(n_queries, name="n_queries")
        self.rrf_k = _validate_top_k(rrf_k, name="rrf_k")
        self.per_query_top_k = _validate_top_k(per_query_top_k, name="per_query_top_k")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        top_k = _validate_top_k(top_k)
        # 1) Génération des reformulations
        queries = self._generate_queries(query)

        # 2) Retrieval par requête
        all_result_lists: List[List[Document]] = []
        for q in queries:
            results = self.base_retriever.retrieve(q, top_k=self.per_query_top_k)
            all_result_lists.append(results)

        # 3) Fusion RRF
        fused = self._reciprocal_rank_fusion(all_result_lists)

        # 4) Top-k final
        return fused[:top_k]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _generate_queries(self, query: str) -> List[str]:
        prompt = MULTI_QUERY_PROMPT_FR.format(n_queries=self.n_queries, query=query)
        raw = self.llm_client.generate(prompt)
        variants = [query]
        seen = {query.strip().casefold()}
        for line in raw.splitlines():
            variant = re.sub(r"^\s*(?:\d+[.)]\s*|[-•]\s*)", "", line).strip()
            if len(variant) <= 3 or variant.casefold() in seen:
                continue
            seen.add(variant.casefold())
            variants.append(variant)
            if len(variants) > self.n_queries:
                break
        return variants

    def _reciprocal_rank_fusion(self, result_lists: List[List[Document]]) -> List[Document]:
        """RRF : chaque document reçoit 1/(k + rang) dans chaque liste où il apparaît."""
        count = sum(len(results) for results in result_lists)
        if not count:
            return []
        fused = ReciprocalRankFusionReranker(k=self.rrf_k).fuse(result_lists, top_k=count)
        for document in fused:
            document.metadata["original_score"] = document.metadata["first_stage_score"]
        return fused
