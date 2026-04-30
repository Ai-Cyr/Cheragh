"""
Technique 20 : Adaptive Retrieval (Retrieval Gate)
===================================================

Inspiré de Self-RAG (Asai et al. 2023) et Adaptive-RAG (Jeong et al. 2024).

Problème : le RAG appelle systématiquement le retriever, même pour des
questions qui n'en ont pas besoin ("Bonjour", "Résume ce texte", "2+2 ?").
Cela ajoute latence, coût, et peut même dégrader la qualité
(le LLM se laisse distraire par des docs hors-sujet).

Solution : un **classifieur LLM** décide EN AMONT si la requête nécessite
du retrieval. Trois comportements possibles :
    - RETRIEVE    : appeler le retriever normalement.
    - NO_RETRIEVE : renvoyer une liste vide (le pipeline laisse le LLM
                    répondre avec ses connaissances paramétriques).
    - REPHRASE    : la question est trop vague/mal formulée, on la
                    réécrit d'abord, puis on retrieve.

Ce module implémente la porte comme un `BaseRetriever` composable, donc
transparent pour `AdvancedRAGPipeline`.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from .base import BaseRetriever, Document, LLMClient


class GateDecision(str, Enum):
    RETRIEVE = "retrieve"
    NO_RETRIEVE = "no_retrieve"
    REPHRASE = "rephrase"


GATE_PROMPT_FR = """Tu es un classifieur qui décide, pour une question donnée, si une recherche documentaire est nécessaire.

Questions qui NE nécessitent PAS de recherche :
- Salutations, small-talk, remerciements.
- Questions générales de culture largement connue (ex: capitale d'un pays, définition courante d'un mot du dictionnaire).
- Reformulations / traductions / résumés d'un contenu déjà fourni dans la conversation.
- Calculs simples, conversions d'unité.

Questions qui NÉCESSITENT une recherche :
- Toute question sur des faits d'entreprise, de domaine métier, de documents spécifiques.
- Toute question factuelle pointue (date, chiffre, procédure, règle) hors culture générale très commune.
- Toute question dont tu n'es pas certain à 100% de la réponse.

Questions à REFORMULER puis chercher :
- Questions très vagues ou ambiguës qui n'auraient pas de bons résultats en l'état.

Réponds UNIQUEMENT par un seul mot : RETRIEVE, NO_RETRIEVE, ou REPHRASE.

Question : {query}

Décision :"""


REPHRASE_PROMPT_FR = """La question suivante est vague ou ambiguë. Reformule-la en une question précise, spécifique et recherche-able dans une base documentaire.

Réponds UNIQUEMENT par la nouvelle question, sans préambule.

Question originale : {query}

Question reformulée :"""


class AdaptiveRetriever(BaseRetriever):
    """
    Retriever avec "porte" : décide s'il faut retriever, et comment.

    Parameters
    ----------
    base_retriever : BaseRetriever
        Retriever à appeler quand la décision est RETRIEVE.
    llm_client : LLMClient
        LLM utilisé comme classifieur.
    allow_rephrase : bool, default=True
        Si False, REPHRASE est traité comme RETRIEVE direct.

    Notes
    -----
    - Quand la décision est NO_RETRIEVE, `retrieve()` renvoie une liste
      vide. Le pipeline `AdvancedRAGPipeline` passera alors au LLM une
      liste vide de sources, ce qui revient à laisser le LLM répondre
      seul. Adapter votre prompt système pour gérer ce cas.
    - La décision est tracée dans `metadata["gate_decision"]` sur chaque
      doc retourné (et accessible via l'attribut `last_decision`).
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_client: LLMClient,
        allow_rephrase: bool = True,
    ):
        self.base_retriever = base_retriever
        self.llm_client = llm_client
        self.allow_rephrase = allow_rephrase
        # État de la dernière décision (utile pour logs/observabilité)
        self.last_decision: Optional[GateDecision] = None
        self.last_used_query: Optional[str] = None

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        decision = self._decide(query)
        self.last_decision = decision
        used_query = query

        if decision == GateDecision.NO_RETRIEVE:
            self.last_used_query = None
            return []

        if decision == GateDecision.REPHRASE and self.allow_rephrase:
            used_query = self._rephrase(query)

        self.last_used_query = used_query
        docs = self.base_retriever.retrieve(used_query, top_k=top_k)

        # Traçabilité
        for d in docs:
            d.metadata["gate_decision"] = decision.value
            d.metadata["gate_used_query"] = used_query
        return docs

    # ------------------------------------------------------------------ #
    def _decide(self, query: str) -> GateDecision:
        prompt = GATE_PROMPT_FR.format(query=query)
        raw = self.llm_client.generate(prompt).strip().upper()
        if "NO_RETRIEVE" in raw or "NO-RETRIEVE" in raw or raw.startswith("NO"):
            return GateDecision.NO_RETRIEVE
        if "REPHRASE" in raw:
            return GateDecision.REPHRASE
        # Défaut prudent : retrieve (mieux vaut un appel en trop qu'un oubli)
        return GateDecision.RETRIEVE

    def _rephrase(self, query: str) -> str:
        prompt = REPHRASE_PROMPT_FR.format(query=query)
        rephrased = self.llm_client.generate(prompt).strip()
        return rephrased.split("\n")[0].strip() or query
