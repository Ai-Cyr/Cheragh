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

import asyncio
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
import re
from typing import Any, List, Optional, Protocol, cast

from .base import BaseRetriever, Document, LLMClient, _snapshot_document, _validate_top_k
from .schema import RAGResponse


class GateDecision(str, Enum):
    RETRIEVE = "retrieve"
    NO_RETRIEVE = "no_retrieve"
    REPHRASE = "rephrase"


class AdaptiveRAGRoute(str, Enum):
    """Retrieval strategy selected from estimated query complexity."""

    NO_RETRIEVAL = "no_retrieval"
    SINGLE_STEP = "single_step"
    ITERATIVE = "iterative"


@dataclass(frozen=True)
class AdaptiveRoutingDecision:
    """Auditable classifier output used by :class:`AdaptiveRAGEngine`."""

    route: AdaptiveRAGRoute
    confidence: float | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.route, AdaptiveRAGRoute):
            raise TypeError("route must be an AdaptiveRAGRoute")
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise TypeError("confidence must be a float or None")
            confidence = float(self.confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be finite and between 0 and 1")
            object.__setattr__(self, "confidence", confidence)
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string")


class QueryComplexityClassifier(Protocol):
    """Classify a query into no-, single- or iterative-retrieval routes."""

    def classify(self, query: str) -> AdaptiveRoutingDecision:
        """Return one validated routing decision."""


class RAGAnswerEngine(Protocol):
    """Minimal answer-engine boundary consumed by Adaptive RAG."""

    def ask(self, query: str, top_k: int | None = None, **kwargs: Any) -> Any:
        """Return a RAGResponse or a wrapper exposing ``.response``."""


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


COMPLEXITY_PROMPT_FR = """Classe la question selon la quantité de recherche documentaire requise.

- NO_RETRIEVAL : salutation, calcul simple ou transformation du texte déjà fourni.
- SINGLE_STEP : une recherche documentaire ciblée suffit.
- ITERATIVE : comparaison, causalité ou question multi-parties demandant plusieurs recherches liées.

Réponds uniquement par NO_RETRIEVAL, SINGLE_STEP ou ITERATIVE.

Question : {query}

Route :"""


DIRECT_ANSWER_PROMPT_FR = """Réponds directement à la requête suivante sans recherche documentaire.
N'invente pas de source et indique clairement toute incertitude.

Requête : {query}

Réponse :"""


class HeuristicComplexityClassifier:
    """Conservative offline classifier for the Adaptive-RAG routing boundary.

    It is a deterministic fallback, not a learned replacement for the paper's
    complexity classifier. Applications can inject a trained classifier or the
    LLM adapter below without changing the engine.
    """

    _NO_RETRIEVAL_PATTERNS = (
        r"^(bonjour|bonsoir|salut|merci|hello|hi)\b",
        r"^\s*\d+(?:\.\d+)?\s*[+*/-]\s*\d+(?:\.\d+)?\s*\??$",
        r"^(traduis|translate|reformule|rewrite)\b.+(?:ce texte|this text)",
    )
    _ITERATIVE_MARKERS = (
        "compare",
        "comparer",
        "différence",
        "difference",
        "pourquoi",
        "why",
        "cause",
        "impact",
        "conséquence",
        "consequence",
        " puis ",
        " ainsi que ",
        " versus ",
        " vs ",
    )

    def classify(self, query: str) -> AdaptiveRoutingDecision:
        normalized = _validate_query(query)
        lowered = normalized.casefold()
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in self._NO_RETRIEVAL_PATTERNS):
            return AdaptiveRoutingDecision(
                AdaptiveRAGRoute.NO_RETRIEVAL,
                confidence=0.7,
                rationale="deterministic_no_retrieval_pattern",
            )
        marker_count = sum(marker in lowered for marker in self._ITERATIVE_MARKERS)
        question_count = normalized.count("?")
        if marker_count or question_count > 1:
            return AdaptiveRoutingDecision(
                AdaptiveRAGRoute.ITERATIVE,
                confidence=0.65,
                rationale="deterministic_multi_part_pattern",
            )
        return AdaptiveRoutingDecision(
            AdaptiveRAGRoute.SINGLE_STEP,
            confidence=0.6,
            rationale="deterministic_single_step_default",
        )


class LLMComplexityClassifier:
    """LLM-backed classifier with a conservative, auditable fallback."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        fallback: QueryComplexityClassifier | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.fallback = fallback or HeuristicComplexityClassifier()

    def classify(self, query: str) -> AdaptiveRoutingDecision:
        normalized = _validate_query(query)
        raw = self.llm_client.generate(COMPLEXITY_PROMPT_FR.format(query=normalized)).strip().upper()
        # Test the longest/more specific label first so incidental prose cannot
        # turn NO_RETRIEVAL into a retrieval route.
        if "NO_RETRIEVAL" in raw or "NO-RETRIEVAL" in raw:
            route = AdaptiveRAGRoute.NO_RETRIEVAL
        elif "ITERATIVE" in raw or "MULTI_STEP" in raw or "MULTI-STEP" in raw:
            route = AdaptiveRAGRoute.ITERATIVE
        elif "SINGLE_STEP" in raw or "SINGLE-STEP" in raw:
            route = AdaptiveRAGRoute.SINGLE_STEP
        else:
            decision = self.fallback.classify(normalized)
            return AdaptiveRoutingDecision(
                decision.route,
                confidence=decision.confidence,
                rationale=f"llm_output_unrecognized:{decision.rationale}",
            )
        return AdaptiveRoutingDecision(route, confidence=None, rationale="llm_complexity_classifier")


class AdaptiveRAGEngine:
    """Route queries between no retrieval, single-step RAG and iterative RAG.

    This architecture captures Adaptive-RAG's public inference boundary while
    keeping its classifier and iterative engine replaceable. A typical setup
    supplies :class:`RAGEngine` for ``single_step_engine`` and
    :class:`MultiHopRAGEngine` for ``iterative_engine``.
    """

    def __init__(
        self,
        single_step_engine: RAGAnswerEngine,
        *,
        iterative_engine: RAGAnswerEngine | None = None,
        llm_client: LLMClient | None = None,
        classifier: QueryComplexityClassifier | None = None,
        top_k: int = 5,
        direct_answer_prompt: str = DIRECT_ANSWER_PROMPT_FR,
        fallback_to_single_step: bool = True,
    ) -> None:
        if not callable(getattr(single_step_engine, "ask", None)):
            raise TypeError("single_step_engine must define ask()")
        if iterative_engine is not None and not callable(getattr(iterative_engine, "ask", None)):
            raise TypeError("iterative_engine must define ask()")
        inferred_llm = getattr(single_step_engine, "llm_client", None)
        resolved_llm = llm_client or inferred_llm
        if not callable(getattr(resolved_llm, "generate", None)):
            raise ValueError("llm_client is required for the no-retrieval route")
        self.llm_client = cast(LLMClient, resolved_llm)
        if classifier is not None and not callable(getattr(classifier, "classify", None)):
            raise TypeError("classifier must define classify()")
        if not isinstance(direct_answer_prompt, str) or "{query}" not in direct_answer_prompt:
            raise ValueError("direct_answer_prompt must be a string containing {query}")
        if not isinstance(fallback_to_single_step, bool):
            raise TypeError("fallback_to_single_step must be a boolean")
        self.single_step_engine = single_step_engine
        self.iterative_engine = iterative_engine
        self.classifier = classifier or HeuristicComplexityClassifier()
        self.top_k = _validate_top_k(top_k)
        self.direct_answer_prompt = direct_answer_prompt
        self.fallback_to_single_step = fallback_to_single_step
        self.last_decision: AdaptiveRoutingDecision | None = None

    def classify(self, query: str) -> AdaptiveRoutingDecision:
        """Expose the selected route without executing generation or retrieval."""

        decision = self.classifier.classify(_validate_query(query))
        if not isinstance(decision, AdaptiveRoutingDecision):
            raise TypeError("classifier.classify() must return AdaptiveRoutingDecision")
        self.last_decision = decision
        return decision

    def ask(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> RAGResponse:
        normalized = _validate_query(query)
        effective_top_k = self.top_k if top_k is None else _validate_top_k(top_k)
        decision = self.classify(normalized)

        if decision.route == AdaptiveRAGRoute.NO_RETRIEVAL:
            prompt = self.direct_answer_prompt.format(query=normalized)
            answer = self.llm_client.generate(prompt, **generate_kwargs)
            return RAGResponse(
                query=normalized,
                answer=answer,
                sources=[],
                retrieved_documents=[],
                prompt=prompt,
                metadata={
                    "architecture": "adaptive_rag",
                    "top_k": effective_top_k,
                    "adaptive_rag": _decision_metadata(decision, decision.route),
                },
                warnings=["adaptive_rag_no_retrieval"],
                grounded_score=0.0,
            )

        engine = self.single_step_engine
        executed_route = decision.route
        fallback_reason: str | None = None
        if decision.route == AdaptiveRAGRoute.ITERATIVE:
            if self.iterative_engine is not None:
                engine = self.iterative_engine
            elif self.fallback_to_single_step:
                executed_route = AdaptiveRAGRoute.SINGLE_STEP
                fallback_reason = "iterative_engine_unavailable"
            else:
                raise RuntimeError("iterative route selected but iterative_engine is not configured")

        raw_response = engine.ask(normalized, top_k=effective_top_k, **generate_kwargs)
        response = _unwrap_rag_response(raw_response)
        response.query = normalized
        response.metadata = deepcopy(response.metadata or {})
        route_metadata = _decision_metadata(decision, executed_route)
        if fallback_reason is not None:
            route_metadata["fallback_reason"] = fallback_reason
            response.warnings = [*response.warnings, "adaptive_rag_iterative_fallback"]
        response.metadata["adaptive_rag"] = route_metadata
        inner_architecture = response.metadata.get("architecture")
        if inner_architecture and inner_architecture != "adaptive_rag":
            response.metadata["inner_architecture"] = inner_architecture
        response.metadata["architecture"] = "adaptive_rag"
        return response

    async def aask(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> RAGResponse:
        """Async wrapper mirroring :meth:`ask`."""

        return await asyncio.to_thread(self.ask, query, top_k, **generate_kwargs)


def _decision_metadata(
    decision: AdaptiveRoutingDecision,
    executed_route: AdaptiveRAGRoute,
) -> dict[str, Any]:
    return {
        "requested_route": decision.route.value,
        "executed_route": executed_route.value,
        "confidence": decision.confidence,
        "rationale": decision.rationale,
    }


def _unwrap_rag_response(value: Any) -> RAGResponse:
    response = value if isinstance(value, RAGResponse) else getattr(value, "response", None)
    if not isinstance(response, RAGResponse):
        raise TypeError("routed engine must return RAGResponse or expose .response")
    return deepcopy(response)


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    return normalized


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
        if not callable(getattr(base_retriever, "retrieve", None)):
            raise TypeError("base_retriever must define retrieve()")
        if not callable(getattr(llm_client, "generate", None)):
            raise TypeError("llm_client must define generate()")
        self.base_retriever = base_retriever
        self.llm_client = llm_client
        if not isinstance(allow_rephrase, bool):
            raise TypeError("allow_rephrase must be a boolean")
        self.allow_rephrase = allow_rephrase
        # État de la dernière décision (utile pour logs/observabilité)
        self.last_decision: Optional[GateDecision] = None
        self.last_used_query: Optional[str] = None

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        top_k = _validate_top_k(top_k)
        query = _validate_query(query)
        decision = self._decide(query)
        self.last_decision = decision
        used_query = query

        if decision == GateDecision.NO_RETRIEVE:
            self.last_used_query = None
            return []

        if decision == GateDecision.REPHRASE and self.allow_rephrase:
            used_query = self._rephrase(query)

        self.last_used_query = used_query
        raw_documents = self.base_retriever.retrieve(used_query, top_k=top_k)
        docs = _validated_retrieval_snapshots(raw_documents, top_k=top_k)

        # Traçabilité
        for d in docs:
            d.metadata["gate_decision"] = decision.value
            d.metadata["gate_used_query"] = used_query
        return docs

    # ------------------------------------------------------------------ #
    def _decide(self, query: str) -> GateDecision:
        prompt = GATE_PROMPT_FR.format(query=query)
        raw = self.llm_client.generate(prompt).strip().upper()
        # Labels explicites d'abord : un préfixe « Non, ... » ou « NOTE: » ne
        # doit pas masquer un RETRIEVE explicite plus loin dans la réponse.
        if "NO_RETRIEVE" in raw or "NO-RETRIEVE" in raw or "NO RETRIEVE" in raw:
            return GateDecision.NO_RETRIEVE
        if "REPHRASE" in raw:
            return GateDecision.REPHRASE
        if "RETRIEVE" in raw:
            return GateDecision.RETRIEVE
        if raw.startswith("NO"):
            return GateDecision.NO_RETRIEVE
        # Défaut prudent : retrieve (mieux vaut un appel en trop qu'un oubli)
        return GateDecision.RETRIEVE

    def _rephrase(self, query: str) -> str:
        prompt = REPHRASE_PROMPT_FR.format(query=query)
        rephrased = self.llm_client.generate(prompt).strip()
        return _validate_query(rephrased.split("\n")[0] or query)


def _validated_retrieval_snapshots(
    documents: Sequence[Document],
    *,
    top_k: int,
) -> list[Document]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise TypeError("base_retriever.retrieve() must return a sequence of Document objects")
    snapshots: list[Document] = []
    for index, document in enumerate(documents[:top_k]):
        if not isinstance(document, Document):
            raise TypeError(f"retrieved documents[{index}] must be a Document")
        if not isinstance(document.content, str):
            raise TypeError(f"retrieved documents[{index}].content must be a string")
        if not isinstance(document.metadata, dict):
            raise TypeError(f"retrieved documents[{index}].metadata must be a dict")
        if document.doc_id is not None and not isinstance(document.doc_id, str):
            raise TypeError(f"retrieved documents[{index}].doc_id must be a string or None")
        if document.score is not None:
            if isinstance(document.score, bool) or not isinstance(document.score, Real):
                raise TypeError(f"retrieved documents[{index}].score must be a real number or None")
            if not math.isfinite(float(document.score)):
                raise ValueError(f"retrieved documents[{index}].score must be finite")
        snapshots.append(_snapshot_document(document))
    return snapshots


__all__ = [
    "AdaptiveRAGEngine",
    "AdaptiveRAGRoute",
    "AdaptiveRetriever",
    "AdaptiveRoutingDecision",
    "GateDecision",
    "HeuristicComplexityClassifier",
    "LLMComplexityClassifier",
    "QueryComplexityClassifier",
]
