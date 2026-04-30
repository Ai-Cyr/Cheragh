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

from typing import Dict, List, Optional

from .base import BaseRetriever, Document, LLMClient


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
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_client: LLMClient,
        max_iterations: int = 8,
        retrieval_top_k: int = 3,
        min_draft_length: int = 20,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.retrieval_top_k = retrieval_top_k
        self.min_draft_length = min_draft_length

    def run(self, query: str) -> Dict:
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

            # 2) Retrieval guidé par le brouillon (FLARE-Direct)
            if len(draft) >= self.min_draft_length:
                retrieval_query = f"{query}\n{draft}"
                hits = self.retriever.retrieve(retrieval_query, top_k=self.retrieval_top_k)
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
                if d.doc_id and d.doc_id not in all_sources:
                    all_sources[d.doc_id] = d

            iteration_log.append({
                "iteration": it + 1,
                "draft": draft,
                "n_retrieved": len(hits),
                "final_sentence": final_sentence,
            })

            # Heuristique d'arrêt : si la phrase est une conclusion, on stoppe
            if any(tok in final_sentence.lower() for tok in ["en conclusion", "en résumé", "[done]"]):
                break

        return {
            "query": query,
            "answer": partial_answer,
            "sources": [
                {"doc_id": d.doc_id, "score": d.score, "preview": d.content[:200]}
                for d in all_sources.values()
            ],
            "iterations": iteration_log,
        }

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        parts = []
        for i, d in enumerate(docs, start=1):
            src = d.doc_id or f"doc_{i}"
            parts.append(f"[source: {src}]\n{d.content}")
        return "\n\n---\n\n".join(parts)
