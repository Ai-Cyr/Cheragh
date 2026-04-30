"""
Technique 15 : Query Router & Ensemble Retrieval
=================================================

Deux patterns d'orchestration complémentaires :

    A. QueryRouter       — un LLM ou un classifieur sélectionne UN retriever
                           approprié parmi plusieurs (ex. un index par
                           domaine : RH, Finance, Juridique, SAV).
                           Moins de calcul, plus de précision sur corpus
                           segmentés.

    B. EnsembleRetriever — exécute TOUS les retrievers en parallèle, fusionne
                           les résultats avec Reciprocal Rank Fusion et un
                           poids par retriever. Utile quand on combine
                           plusieurs approches (sémantique, lexicale,
                           BDD structurée, graph, etc.).

Ces deux patterns sont la base d'une architecture RAG multi-index / multi-source
en production.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import BaseRetriever, Document, LLMClient


ROUTER_PROMPT_FR = """Tu es un routeur qui choisit le meilleur index documentaire pour répondre à une question.

Index disponibles :
{routes_description}

Règles :
- Choisis UN seul index, celui le plus susceptible de contenir la réponse.
- Réponds UNIQUEMENT par le NOM EXACT de l'index, sans préambule ni explication.
- Si aucun n'est clairement adapté, choisis celui qui semble le plus général.

Question : {query}

Nom de l'index choisi :"""


# ===================================================================== #
# Router
# ===================================================================== #
class QueryRouter(BaseRetriever):
    """
    Route la requête vers UN retriever parmi plusieurs via un LLM.

    Parameters
    ----------
    routes : Dict[str, BaseRetriever]
        Mapping {nom_de_route: retriever}.
    route_descriptions : Dict[str, str]
        Mapping {nom_de_route: description en langage naturel de ce que
        contient cet index}. Utilisé par le LLM pour choisir.
    llm_client : LLMClient
    default_route : str, optional
        Nom de la route par défaut si le LLM renvoie un nom inconnu.
    """

    def __init__(
        self,
        routes: Dict[str, BaseRetriever],
        route_descriptions: Dict[str, str],
        llm_client: LLMClient,
        default_route: Optional[str] = None,
    ):
        if not routes:
            raise ValueError("Au moins une route est requise.")
        missing = set(routes) - set(route_descriptions)
        if missing:
            raise ValueError(f"Descriptions manquantes pour : {missing}")
        if default_route and default_route not in routes:
            raise ValueError(f"default_route='{default_route}' n'existe pas dans routes.")

        self.routes = routes
        self.route_descriptions = route_descriptions
        self.llm_client = llm_client
        self.default_route = default_route or next(iter(routes))

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        # 1) Décision : quelle route ?
        chosen = self._route(query)

        # 2) Exécution du retriever choisi
        docs = self.routes[chosen].retrieve(query, top_k=top_k)

        # 3) Traçabilité : on marque la route utilisée dans les metadata
        for d in docs:
            d.metadata["route_used"] = chosen
        return docs

    def _route(self, query: str) -> str:
        desc_str = "\n".join(f"- {name} : {desc}" for name, desc in self.route_descriptions.items())
        prompt = ROUTER_PROMPT_FR.format(routes_description=desc_str, query=query)
        raw = self.llm_client.generate(prompt).strip().split("\n")[0].strip()

        # Matching tolérant (casse, espaces)
        normalized = raw.lower().strip(" .\"'")
        for name in self.routes:
            if name.lower() == normalized or name.lower() in normalized:
                return name
        return self.default_route


# ===================================================================== #
# Ensemble
# ===================================================================== #
class EnsembleRetriever(BaseRetriever):
    """
    Exécute plusieurs retrievers et fusionne les résultats via RRF pondéré.

    Parameters
    ----------
    retrievers : List[BaseRetriever]
        Retrievers à combiner.
    weights : List[float], optional
        Poids par retriever (par défaut : poids égaux). Les poids n'ont pas
        besoin d'être normalisés — ils sont utilisés tels quels dans RRF.
    rrf_k : int, default=60
        Constante RRF standard.
    per_retriever_top_k : int, default=10
        Top-k récupéré de chaque retriever avant fusion.
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: Optional[List[float]] = None,
        rrf_k: int = 60,
        per_retriever_top_k: int = 10,
    ):
        if not retrievers:
            raise ValueError("Au moins un retriever est requis.")
        if weights is None:
            weights = [1.0] * len(retrievers)
        if len(weights) != len(retrievers):
            raise ValueError("len(weights) doit égaler len(retrievers).")
        if any(w < 0 for w in weights):
            raise ValueError("Les poids doivent être >= 0.")

        self.retrievers = retrievers
        self.weights = weights
        self.rrf_k = rrf_k
        self.per_retriever_top_k = per_retriever_top_k

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Document] = {}
        source_map: Dict[str, List[int]] = {}  # doc_key -> [indices des retrievers l'ayant ramené]

        for r_idx, (retriever, weight) in enumerate(zip(self.retrievers, self.weights)):
            if weight == 0:
                continue
            hits = retriever.retrieve(query, top_k=self.per_retriever_top_k)
            for rank, doc in enumerate(hits):
                key = doc.doc_id if doc.doc_id is not None else f"content::{hash(doc.content)}"
                rrf_scores[key] = rrf_scores.get(key, 0.0) + weight / (self.rrf_k + rank + 1)
                if key not in doc_lookup:
                    doc_lookup[key] = doc
                source_map.setdefault(key, []).append(r_idx)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results: List[Document] = []
        for key, rrf_score in ranked[:top_k]:
            base = doc_lookup[key]
            results.append(
                Document(
                    content=base.content,
                    metadata={
                        **base.metadata,
                        "ensemble_sources": source_map[key],
                        "ensemble_rrf_score": float(rrf_score),
                    },
                    doc_id=base.doc_id,
                    score=float(rrf_score),
                )
            )
        return results
