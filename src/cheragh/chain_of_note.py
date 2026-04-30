"""
Technique 19 : Chain-of-Note (CoN)
===================================

Yu et al. (2023) — "Chain-of-Note: Enhancing Robustness in Retrieval-
Augmented Language Models".

Problème : quand les documents récupérés contiennent un mélange
d'information pertinente, non-pertinente, et parfois contradictoire,
le LLM générateur s'embrouille et hallucine.

Solution : avant de générer la réponse, le LLM rédige pour chaque
document une NOTE structurée en 3 parties :
    1. Pertinence (directement pertinent / partiellement / non pertinent)
    2. Information-clé extraite (ou "aucune")
    3. Limites / ce qui manque

Puis, au moment de la génération, le LLM s'appuie sur les NOTES (pas
sur les documents bruts). Cela réduit la confusion, force la prise de
recul, et rend explicites les manques.

Différences avec Contextual Compression (7) :
    - Compression extrait des PHRASES brutes du document.
    - Chain-of-Note RÉDIGE des notes structurées (avec jugement de
      pertinence et analyse des manques).

On peut aussi enchaîner : CoN sur des docs déjà compressés.
"""
from __future__ import annotations

import re
from typing import List

from .base import BaseRetriever, Document, LLMClient


NOTE_TAKING_PROMPT_FR = """Tu prends des notes sur un extrait de document pour répondre à une question.

Rédige la note EXACTEMENT dans ce format (respecter les balises) :

PERTINENCE: <directement pertinent | partiellement pertinent | non pertinent>
INFORMATION_CLE: <les faits/chiffres/règles extraits, ou "aucune">
LIMITES: <ce qui manque dans cet extrait pour répondre pleinement, ou "aucune">

Sois concis (3-5 lignes maximum par section). Ne reformule pas inutilement : cite les faits tels quels si possible.

Question : {query}

Extrait :
{document}

Note :"""


class ChainOfNoteRetriever(BaseRetriever):
    """
    Annote chaque document retrouvé avec une note structurée.

    Parameters
    ----------
    base_retriever : BaseRetriever
    llm_client : LLMClient
    drop_not_relevant : bool, default=True
        Si True, les docs notés "non pertinent" sont supprimés du résultat.
    fetch_multiplier : int, default=2
        Le base_retriever est appelé avec top_k * fetch_multiplier pour
        compenser les docs qui seront filtrés.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_client: LLMClient,
        drop_not_relevant: bool = True,
        fetch_multiplier: int = 2,
    ):
        self.base_retriever = base_retriever
        self.llm_client = llm_client
        self.drop_not_relevant = drop_not_relevant
        self.fetch_multiplier = max(1, fetch_multiplier)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        candidates = self.base_retriever.retrieve(
            query, top_k=top_k * self.fetch_multiplier
        )

        noted_docs: List[Document] = []
        for doc in candidates:
            prompt = NOTE_TAKING_PROMPT_FR.format(query=query, document=doc.content[:3000])
            raw_note = self.llm_client.generate(prompt).strip()
            parsed = self._parse_note(raw_note)

            if self.drop_not_relevant and parsed["pertinence"] == "non pertinent":
                continue

            # Nouveau contenu = la note structurée (le générateur final
            # lira cela à la place du document brut)
            noted_content = (
                f"[Note structurée sur le document {doc.doc_id or '?'}]\n"
                f"Pertinence : {parsed['pertinence']}\n"
                f"Information clé : {parsed['information_cle']}\n"
                f"Limites : {parsed['limites']}"
            )

            noted_docs.append(
                Document(
                    content=noted_content,
                    metadata={
                        **doc.metadata,
                        "con_pertinence": parsed["pertinence"],
                        "con_raw_note": raw_note,
                        "original_content": doc.content,
                    },
                    doc_id=doc.doc_id,
                    score=doc.score,
                )
            )

            if len(noted_docs) >= top_k:
                break

        # Tri : directement pertinent > partiellement > non pertinent
        priority = {"directement pertinent": 0, "partiellement pertinent": 1, "non pertinent": 2}
        noted_docs.sort(
            key=lambda d: (priority.get(d.metadata.get("con_pertinence", ""), 3), -(d.score or 0))
        )
        return noted_docs[:top_k]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_note(raw: str) -> dict:
        """Parse le format PERTINENCE / INFORMATION_CLE / LIMITES."""
        out = {"pertinence": "partiellement pertinent", "information_cle": "", "limites": ""}

        # Extraction tolérante aux majuscules/accents
        for key, field in [
            ("pertinence", "PERTINENCE"),
            ("information_cle", "INFORMATION_CLE"),
            ("limites", "LIMITES"),
        ]:
            # Match "FIELD:" jusqu'à la prochaine balise (ou fin)
            pattern = rf"{field}\s*:\s*(.+?)(?=\n\s*(?:PERTINENCE|INFORMATION_CLE|LIMITES)\s*:|$)"
            m = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
            if m:
                out[key] = m.group(1).strip()

        # Normaliser le label de pertinence
        p = out["pertinence"].lower()
        if "non pertinent" in p or "non-pertinent" in p:
            out["pertinence"] = "non pertinent"
        elif "partiellement" in p or "partiel" in p:
            out["pertinence"] = "partiellement pertinent"
        elif "directement" in p or "pertinent" in p:
            out["pertinence"] = "directement pertinent"
        return out
