"""
Technique 17 : FLARE
=====================

Jiang et al. (2023) — "Active Retrieval Augmented Generation".
Variante implémentée : FLARE-Direct, la plus simple et robuste.

Problème : un RAG classique fait UN seul retrieval au début puis génère
toute la réponse. Pour les réponses longues ou multi-facettes, les faits
nécessaires plus loin dans la réponse ne sont souvent pas dans le
contexte initial.

Solution FLARE : boucle itérative qui **alterne génération et retrieval** :
    1. Génère un brouillon de la prochaine phrase (lookahead).
    2. Utilise ce brouillon comme REQUÊTE pour récupérer des documents
       pertinents à cette phrase spécifique.
    3. Régénère la phrase finale en s'appuyant sur les nouveaux documents.
    4. Ajoute à la réponse en cours et recommence jusqu'à complétion.

Contrairement aux autres modules, FLARE est un **Pipeline** (pas un
Retriever) car il entrelace retrieval et génération.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Dict, List, Protocol

from .base import BaseRetriever, Document, LLMClient, _snapshot_document, _validate_top_k
from .citations import validate_citations
from .schema import RAGResponse, Source


DRAFT_NEXT_PROMPT_FR = """Tu rédiges une réponse à une question, phrase par phrase.

Question : {query}

Réponse déjà rédigée :
{partial_answer}

Rédige UNIQUEMENT la prochaine phrase de la réponse (pas plus).
Si la réponse est complète, réponds exactement : [DONE]

Prochaine phrase :"""


FINAL_NEXT_PROMPT_FR = """Tu rédiges une réponse à une question, phrase par phrase, en t'appuyant sur des extraits fournis.

Question : {query}

Réponse déjà rédigée :
{partial_answer}

Extraits pertinents pour la prochaine phrase :
{context}

Rédige UNIQUEMENT la prochaine phrase (pas plus) en t'appuyant sur les extraits ci-dessus.
Si la réponse est complète, réponds exactement : [DONE]
Cite les sources à la fin de la phrase sous la forme [source: doc_id] si applicable.

Prochaine phrase :"""


@dataclass(frozen=True)
class TokenConfidence:
    """Confidence attached to one predicted token or text span.

    Providers exposing generation log-probabilities can adapt their output to
    this small, provider-neutral type. ``confidence`` is a probability in the
    closed interval ``[0, 1]``.
    """

    text: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("TokenConfidence.text must be a non-empty string")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise TypeError("TokenConfidence.confidence must be a float")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("TokenConfidence.confidence must be finite and between 0 and 1")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class DraftUncertainty:
    """Decision produced after inspecting a look-ahead draft."""

    requires_retrieval: bool
    confidence: float | None = None
    low_confidence_spans: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.requires_retrieval, bool):
            raise TypeError("requires_retrieval must be a boolean")
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise TypeError("confidence must be a float or None")
            confidence = float(self.confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be finite and between 0 and 1")
            object.__setattr__(self, "confidence", confidence)
        if isinstance(self.low_confidence_spans, (str, bytes)) or not isinstance(
            self.low_confidence_spans,
            Sequence,
        ):
            raise TypeError("low_confidence_spans must be a sequence of strings")
        if any(not isinstance(span, str) or not span.strip() for span in self.low_confidence_spans):
            raise ValueError("low_confidence_spans must contain non-empty strings")
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string")
        object.__setattr__(
            self,
            "low_confidence_spans",
            tuple(span.strip() for span in self.low_confidence_spans),
        )


class DraftUncertaintyEstimator(Protocol):
    """Provider-neutral uncertainty boundary used by :class:`FLAREPipeline`."""

    def assess(self, query: str, partial_answer: str, draft: str) -> DraftUncertainty:
        """Decide whether the predicted draft needs fresh evidence."""


@dataclass(frozen=True)
class LengthBasedDraftUncertainty:
    """Backward-compatible FLARE-Direct heuristic.

    This fallback keeps Cheragh runnable with LLM clients that expose only
    text. It is intentionally labelled as a heuristic rather than token-level
    uncertainty.
    """

    min_draft_length: int = 20

    def __post_init__(self) -> None:
        if isinstance(self.min_draft_length, bool) or not isinstance(self.min_draft_length, int):
            raise TypeError("min_draft_length must be an integer")
        if self.min_draft_length < 0:
            raise ValueError("min_draft_length must be >= 0")

    def assess(self, query: str, partial_answer: str, draft: str) -> DraftUncertainty:
        del query, partial_answer
        should_retrieve = len(draft) >= self.min_draft_length
        return DraftUncertainty(
            requires_retrieval=should_retrieve,
            low_confidence_spans=(draft,) if should_retrieve else (),
            rationale="draft_length_fallback",
        )


class TokenConfidenceUncertaintyEstimator:
    """Trigger retrieval when a provider reports low-confidence tokens.

    ``confidence_provider`` may wrap token log-probabilities from any LLM. It
    receives the complete look-ahead draft and returns token/span confidence
    values. Retrieval focuses on the low-confidence spans instead of querying
    with the whole sentence, matching FLARE's active-retrieval boundary more
    closely while remaining independent from a specific model SDK.
    """

    def __init__(
        self,
        confidence_provider: Callable[[str], Sequence[TokenConfidence]],
        *,
        threshold: float = 0.5,
        min_low_confidence_tokens: int = 1,
    ) -> None:
        if not callable(confidence_provider):
            raise TypeError("confidence_provider must be callable")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a float")
        if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be finite and between 0 and 1")
        self.confidence_provider = confidence_provider
        self.threshold = float(threshold)
        self.min_low_confidence_tokens = _validate_top_k(
            min_low_confidence_tokens,
            name="min_low_confidence_tokens",
        )

    def assess(self, query: str, partial_answer: str, draft: str) -> DraftUncertainty:
        del query, partial_answer
        raw_confidences = self.confidence_provider(draft)
        if isinstance(raw_confidences, (str, bytes)) or not isinstance(raw_confidences, Sequence):
            raise TypeError("confidence_provider must return a sequence of TokenConfidence values")
        confidences = list(raw_confidences)
        if any(not isinstance(item, TokenConfidence) for item in confidences):
            raise TypeError("confidence_provider must return TokenConfidence values")
        low_confidence = tuple(
            item.text.strip() for item in confidences if item.confidence < self.threshold
        )
        overall = min((float(item.confidence) for item in confidences), default=None)
        return DraftUncertainty(
            requires_retrieval=len(low_confidence) >= self.min_low_confidence_tokens,
            confidence=overall,
            low_confidence_spans=low_confidence,
            rationale="token_confidence_threshold",
        )


class FLAREPipeline:
    """
    Pipeline FLARE-Direct : boucle génération → retrieval → régénération.

    Parameters
    ----------
    retriever : BaseRetriever
    llm_client : LLMClient
    max_iterations : int, default=8
        Nombre maximum de phrases générées avant arrêt de sécurité.
    retrieval_top_k : int, default=3
        Nombre de docs récupérés à chaque itération.
    min_draft_length : int, default=20
        Longueur minimale (caractères) d'un brouillon pour déclencher un
        retrieval. Évite de chercher sur des fragments vides.
    uncertainty_estimator : DraftUncertaintyEstimator, optional
        When supplied, this component fully controls the retrieval decision.
        It can adapt provider token log-probabilities through
        :class:`TokenConfidenceUncertaintyEstimator`. Without it, the legacy
        length heuristic remains active.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_client: LLMClient,
        max_iterations: int = 8,
        retrieval_top_k: int = 3,
        min_draft_length: int = 20,
        uncertainty_estimator: DraftUncertaintyEstimator | None = None,
    ):
        if not callable(getattr(retriever, "retrieve", None)):
            raise TypeError("retriever must define retrieve()")
        if not callable(getattr(llm_client, "generate", None)):
            raise TypeError("llm_client must define generate()")
        self.retriever = retriever
        self.llm_client = llm_client
        self.max_iterations = _validate_top_k(max_iterations, name="max_iterations")
        self.retrieval_top_k = _validate_top_k(retrieval_top_k, name="retrieval_top_k")
        if isinstance(min_draft_length, bool) or not isinstance(min_draft_length, int):
            raise TypeError("min_draft_length must be an integer")
        if min_draft_length < 0:
            raise ValueError("min_draft_length must be >= 0")
        self.min_draft_length = min_draft_length
        if uncertainty_estimator is not None and not callable(
            getattr(uncertainty_estimator, "assess", None)
        ):
            raise TypeError("uncertainty_estimator must define assess()")
        self.uncertainty_estimator = uncertainty_estimator or LengthBasedDraftUncertainty(
            min_draft_length=min_draft_length
        )

    def run(self, query: str) -> Dict:
        """Execute FLARE and return the backward-compatible dictionary payload."""

        payload, _ = self._execute(query, retrieval_top_k=self.retrieval_top_k)
        return payload

    def ask(self, query: str, top_k: int | None = None) -> RAGResponse:
        """Execute FLARE and expose the shared structured response contract.

        FLARE generates an answer through several prompts, so ``prompt`` is
        intentionally empty instead of pretending one prompt produced the full
        answer. Per-iteration draft and retrieval diagnostics live in metadata.
        """

        effective_top_k = self.retrieval_top_k if top_k is None else _validate_top_k(top_k)
        payload, documents = self._execute(query, retrieval_top_k=effective_top_k)
        validation = validate_citations(payload["answer"], documents, require_citations=False)
        if not documents:
            validation.grounded_score = 0.0
            if payload["answer"]:
                validation.warnings.append("flare_no_retrieval")
        return RAGResponse(
            query=payload["query"],
            answer=payload["answer"],
            sources=[Source.from_document(document) for document in documents],
            retrieved_documents=[_snapshot_document(document) for document in documents],
            prompt="",
            metadata={
                "architecture": "flare",
                "top_k": effective_top_k,
                "multi_prompt_generation": True,
                "iterations": payload["iterations"],
            },
            citations=validation.citations,
            warnings=validation.warnings,
            grounded_score=validation.grounded_score,
            unsourced_claims=validation.unsourced_claims,
            citation_validation=validation,
        )

    def _execute(
        self,
        query: str,
        *,
        retrieval_top_k: int,
    ) -> tuple[Dict, list[Document]]:
        retrieval_top_k = _validate_top_k(retrieval_top_k, name="retrieval_top_k")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        partial_answer = ""
        all_sources: Dict[str, Document] = {}
        iteration_log: List[Dict] = []

        for it in range(self.max_iterations):
            # 1) Génération d'un BROUILLON de la prochaine phrase
            draft_prompt = DRAFT_NEXT_PROMPT_FR.format(
                query=query,
                partial_answer=partial_answer or "(rien encore)",
            )
            draft = self.llm_client.generate(draft_prompt).strip()

            if "[DONE]" in draft or not draft:
                break

            # 2) Retrieval guided by uncertainty in the look-ahead draft.
            assessment = self.uncertainty_estimator.assess(query, partial_answer, draft)
            if not isinstance(assessment, DraftUncertainty):
                raise TypeError("uncertainty_estimator.assess() must return DraftUncertainty")
            retrieval_query: str | None = None
            if assessment.requires_retrieval:
                focus = " ".join(assessment.low_confidence_spans).strip() or draft
                retrieval_query = f"{query}\nIncertitude à vérifier: {focus}"
                hits = _validated_hits(
                    self.retriever.retrieve(retrieval_query, top_k=retrieval_top_k),
                    top_k=retrieval_top_k,
                )
            else:
                hits = []

            # 3) Régénération de la phrase finale avec les docs en contexte
            if hits:
                context_str = self._format_context(hits)
                final_prompt = FINAL_NEXT_PROMPT_FR.format(
                    query=query,
                    partial_answer=partial_answer or "(rien encore)",
                    context=context_str,
                )
                final_sentence = self.llm_client.generate(final_prompt).strip()
            else:
                # Pas de retrieval → on garde le brouillon tel quel
                final_sentence = draft

            if "[DONE]" in final_sentence or not final_sentence:
                break

            # 4) Ajout à la réponse et accumulation des sources
            partial_answer = (partial_answer + " " + final_sentence).strip()
            for d in hits:
                source_key = d.doc_id or f"content::{d.content}"
                if source_key not in all_sources:
                    all_sources[source_key] = _snapshot_document(d)

            iteration_log.append({
                "iteration": it + 1,
                "draft": draft,
                "retrieval_triggered": assessment.requires_retrieval,
                "retrieval_query": retrieval_query,
                "draft_confidence": assessment.confidence,
                "low_confidence_spans": list(assessment.low_confidence_spans),
                "uncertainty_rationale": assessment.rationale,
                "n_retrieved": len(hits),
                "final_sentence": final_sentence,
            })

            # Heuristique d'arrêt : si la phrase est une conclusion, on stoppe
            if any(tok in final_sentence.lower() for tok in ["en conclusion", "en résumé", "[done]"]):
                break

        documents = list(all_sources.values())
        return {
            "query": query,
            "answer": partial_answer,
            "sources": [
                {"doc_id": d.doc_id, "score": d.score, "preview": d.content[:200]}
                for d in documents
            ],
            "iterations": iteration_log,
        }, documents

    @staticmethod
    def _format_context(docs: Sequence[Document]) -> str:
        parts = []
        for i, d in enumerate(docs, start=1):
            src = d.doc_id or f"doc_{i}"
            parts.append(f"[source: {src}]\n{d.content}")
        return "\n\n---\n\n".join(parts)


def _validated_hits(documents: Sequence[Document], *, top_k: int) -> list[Document]:
    """Validate an injected retriever and enforce its advertised hard limit."""

    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise TypeError("retriever.retrieve() must return a sequence of Document objects")
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
    "DraftUncertainty",
    "DraftUncertaintyEstimator",
    "FLAREPipeline",
    "LengthBasedDraftUncertainty",
    "TokenConfidence",
    "TokenConfidenceUncertaintyEstimator",
]
