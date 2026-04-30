"""
Technique 9 : Step-Back Prompting
==================================

Zheng et al. (Google DeepMind, 2023) — "Take a Step Back".

Problème : une question très spécifique peut manquer de contexte
général pour que le LLM réponde correctement. Exemple :
    "Quelle est la peine encourue pour un retard de déclaration TVA
     de 45 jours en France en 2024 ?"

Solution : le LLM génère une question plus ABSTRAITE / PLUS GÉNÉRALE
("step-back") avant le retrieval, par ex. :
    "Quelles sont les sanctions liées au retard de déclaration TVA ?"

On retrieve sur les DEUX (la spécifique ET l'abstraite) → le LLM
générateur dispose à la fois de la règle générale et du cas précis.

Complémentaire à HyDE : HyDE *matérialise* une réponse, Step-Back
*abstrait* la question.
"""
from __future__ import annotations

from typing import Dict, List

from .base import BaseRetriever, Document, LLMClient


STEP_BACK_PROMPT_FR = """Reformule la question spécifique suivante en une question plus générale et plus abstraite
(question "step-back") qui permettrait de retrouver les principes ou règles sous-jacents.

Exemples :
- Spécifique : "Puis-je déduire un repas d'affaires à 85 € le 12 mars 2024 à Lyon ?"
  Step-back : "Quelles sont les règles de déductibilité des repas d'affaires ?"
- Spécifique : "Mon manager peut-il refuser mes congés pour la semaine du 15 juillet ?"
  Step-back : "Quelles sont les règles encadrant le refus de congés par l'employeur ?"

Réponds UNIQUEMENT par la question step-back, sans préambule.

Question spécifique : {query}

Question step-back :"""


class StepBackRetriever(BaseRetriever):
    """
    Retriever combinant la question d'origine et une version step-back.

    Parameters
    ----------
    base_retriever : BaseRetriever
    llm_client : LLMClient
    n_original : int, default=3
        Nombre de documents à ramener pour la question d'origine.
    n_stepback : int, default=3
        Nombre de documents à ramener pour la question step-back.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_client: LLMClient,
        n_original: int = 3,
        n_stepback: int = 3,
    ):
        self.base_retriever = base_retriever
        self.llm_client = llm_client
        self.n_original = n_original
        self.n_stepback = n_stepback

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        # 1) Génération de la question step-back
        stepback_query = self._generate_stepback(query)

        # 2) Retrieval sur les deux
        original_hits = self.base_retriever.retrieve(query, top_k=self.n_original)
        stepback_hits = self.base_retriever.retrieve(stepback_query, top_k=self.n_stepback)

        # 3) Dédoublonnage en conservant l'origine (spécifique / général)
        seen: Dict[str, Document] = {}
        for doc in original_hits:
            key = doc.doc_id if doc.doc_id is not None else f"content::{hash(doc.content)}"
            seen[key] = Document(
                content=doc.content,
                metadata={**doc.metadata, "retrieval_source": "original", "stepback_query": stepback_query},
                doc_id=doc.doc_id,
                score=doc.score,
            )
        for doc in stepback_hits:
            key = doc.doc_id if doc.doc_id is not None else f"content::{hash(doc.content)}"
            if key in seen:
                # Déjà trouvé par la question originale → marquer comme "both"
                seen[key].metadata["retrieval_source"] = "both"
            else:
                seen[key] = Document(
                    content=doc.content,
                    metadata={**doc.metadata, "retrieval_source": "stepback", "stepback_query": stepback_query},
                    doc_id=doc.doc_id,
                    score=doc.score,
                )

        # 4) On privilégie les "both" puis "original" puis "stepback" et trie par score
        priority = {"both": 0, "original": 1, "stepback": 2}
        merged = sorted(
            seen.values(),
            key=lambda d: (priority.get(d.metadata.get("retrieval_source", "stepback"), 3), -(d.score or 0)),
        )
        return merged[:top_k]

    # ------------------------------------------------------------------ #
    def _generate_stepback(self, query: str) -> str:
        prompt = STEP_BACK_PROMPT_FR.format(query=query)
        stepback = self.llm_client.generate(prompt).strip()
        # Garder une seule ligne si le LLM est bavard
        return stepback.split("\n")[0].strip()
