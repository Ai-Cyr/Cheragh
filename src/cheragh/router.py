"""
Technique 15 : Query Router & Ensemble Retrieval
=================================================

Deux patterns d'orchestration complémentaires :

    A. QueryRouter       — un LLM ou un classifieur sélectionne UN retriever
                           approprié parmi plusieurs (ex. un index par
                           domaine : RH, Finance, Juridique, SAV).
                           Moins de calcul, plus de précision sur corpus
                           segmentés.

    B. EnsembleRetriever — exécute TOUS les retrievers, fusionne
                           les résultats avec Reciprocal Rank Fusion et un
                           poids par retriever. Utile quand on combine
                           plusieurs approches (sémantique, lexicale,
                           BDD structurée, graph, etc.).

Ces deux patterns sont la base d'une architecture RAG multi-index / multi-source
en production.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Dict, List, Optional

from .base import BaseRetriever, Document, LLMClient, _snapshot_document, _validate_top_k
from .reranking import _rrf_document_key


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
        if any(not isinstance(name, str) or not name.strip() for name in routes):
            raise ValueError("Route names must be non-empty strings")
        if len({name.casefold() for name in routes}) != len(routes):
            raise ValueError("Route names must be unique ignoring case")
        if any(not callable(getattr(retriever, "retrieve", None)) for retriever in routes.values()):
            raise TypeError("Every route must implement retrieve()")
        missing = set(routes) - set(route_descriptions)
        if missing:
            raise ValueError(f"Descriptions manquantes pour : {missing}")
        if any(not isinstance(route_descriptions[name], str) for name in routes):
            raise TypeError("Route descriptions must be strings")
        if not callable(getattr(llm_client, "generate", None)):
            raise TypeError("llm_client must implement generate()")
        if default_route is not None and default_route not in routes:
            raise ValueError(f"default_route='{default_route}' n'existe pas dans routes.")

        self.routes = dict(routes)
        self.route_descriptions = {name: route_descriptions[name] for name in routes}
        self.llm_client = llm_client
        self.default_route = default_route or next(iter(routes))

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        from .engine import _validate_query, _validated_document_results

        query = _validate_query(query)
        top_k = _validate_top_k(top_k)
        # 1) Décision : quelle route ?
        chosen = self._route(query)

        # 2) Exécution du retriever choisi
        docs = _validated_document_results(
            self.routes[chosen].retrieve(query, top_k=top_k), limit=top_k, name=f"route {chosen} results",
        )

        # 3) Traçabilité : on marque la route utilisée dans les metadata
        for d in docs:
            d.metadata["route_used"] = chosen
        return docs

    def _route(self, query: str) -> str:
        desc_str = "\n".join(f"- {name} : {desc}" for name, desc in self.route_descriptions.items())
        prompt = ROUTER_PROMPT_FR.format(routes_description=desc_str, query=query)
        generated = self.llm_client.generate(prompt)
        if not isinstance(generated, str):
            raise TypeError("Router LLM must return a string")
        raw = generated.strip().split("\n")[0].strip()

        # Matching tolérant (casse, espaces)
        normalized = raw.casefold().strip(" .\"'")
        for name in self.routes:
            if name.casefold() == normalized:
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
        if any(not callable(getattr(retriever, "retrieve", None)) for retriever in retrievers):
            raise TypeError("Every ensemble member must implement retrieve()")
        if weights is None:
            weights = [1.0] * len(retrievers)
        if len(weights) != len(retrievers):
            raise ValueError("len(weights) doit égaler len(retrievers).")
        if any(isinstance(weight, bool) or not isinstance(weight, Real) for weight in weights):
            raise TypeError("Ensemble weights must be real numbers, excluding booleans")
        if any(not math.isfinite(float(weight)) or weight < 0 for weight in weights):
            raise ValueError("Ensemble weights must be finite and non-negative")

        self.retrievers = list(retrievers)
        self.weights = [float(weight) for weight in weights]
        self.rrf_k = _validate_top_k(rrf_k, name="rrf_k")
        self.per_retriever_top_k = _validate_top_k(per_retriever_top_k, name="per_retriever_top_k")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        from .engine import _validate_query, _validated_document_results

        query = _validate_query(query)
        top_k = _validate_top_k(top_k)
        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Document] = {}
        source_map: Dict[str, List[int]] = {}  # doc_key -> [indices des retrievers l'ayant ramené]

        for r_idx, (retriever, weight) in enumerate(zip(self.retrievers, self.weights)):
            if weight == 0:
                continue
            hits = _validated_document_results(
                retriever.retrieve(query, top_k=self.per_retriever_top_k), limit=self.per_retriever_top_k,
                name=f"ensemble retriever {r_idx} results",
            )
            seen_in_source: set[str] = set()
            for rank, doc in enumerate(hits):
                key = _rrf_document_key(doc)
                if key in seen_in_source:
                    continue
                seen_in_source.add(key)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + weight / (self.rrf_k + rank + 1)
                if not math.isfinite(rrf_scores[key]):
                    raise ValueError("Ensemble RRF score exceeds the finite numeric range")
                if key not in doc_lookup:
                    doc_lookup[key] = doc
                source_map.setdefault(key, []).append(r_idx)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results: List[Document] = []
        for key, rrf_score in ranked[:top_k]:
            base = _snapshot_document(doc_lookup[key])
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
