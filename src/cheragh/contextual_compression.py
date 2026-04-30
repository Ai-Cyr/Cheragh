"""
Technique 7 : Contextual Compression
=====================================

Problème : les chunks retournés contiennent souvent beaucoup de texte
non pertinent qui dilue le signal, augmente le coût tokens du LLM
générateur, et dégrade la qualité de la réponse ("lost in the middle").

Solution : après retrieval, on passe chaque chunk à un LLM extracteur
qui ne garde que les phrases/passages pertinents pour la question.
Si aucun passage n'est pertinent, on jette le chunk.

C'est un "filtre LLM" placé entre le retriever et le générateur.
"""
from __future__ import annotations

from typing import List

from .base import BaseRetriever, Document, LLMClient


COMPRESSION_PROMPT_FR = """Tu reçois une question et un extrait de document.
Ta tâche : renvoyer UNIQUEMENT les phrases de l'extrait qui sont directement pertinentes pour répondre à la question.

Règles strictes :
- Conserve les phrases pertinentes TELLES QUELLES, sans reformuler.
- Supprime tout le reste.
- Si AUCUNE phrase n'est pertinente, réponds exactement : NO_OUTPUT
- Ne préface pas ta réponse, ne commente pas.

Question : {query}

Extrait :
{document}

Phrases pertinentes :"""


class ContextualCompressionRetriever(BaseRetriever):
    """
    Compresse les documents retournés par un retriever en ne gardant que
    les passages pertinents via un LLM extracteur.

    Parameters
    ----------
    base_retriever : BaseRetriever
    llm_client : LLMClient
        LLM utilisé pour la compression (un modèle rapide/petit suffit,
        ex. gpt-4o-mini, claude-haiku).
    drop_empty : bool, default=True
        Si True, supprime les documents dont la compression retourne NO_OUTPUT.
    min_compressed_length : int, default=20
        Longueur minimale (caractères) d'une compression valide. En-dessous,
        on considère que rien n'a été extrait.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_client: LLMClient,
        drop_empty: bool = True,
        min_compressed_length: int = 20,
    ):
        self.base_retriever = base_retriever
        self.llm_client = llm_client
        self.drop_empty = drop_empty
        self.min_compressed_length = min_compressed_length

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        # 1) Retrieval de départ — on prend un peu plus pour compenser les
        #    documents qui seront compressés en NO_OUTPUT et filtrés.
        retrieved = self.base_retriever.retrieve(query, top_k=top_k * 2)

        compressed_docs: List[Document] = []
        for doc in retrieved:
            prompt = COMPRESSION_PROMPT_FR.format(query=query, document=doc.content)
            compressed_content = self.llm_client.generate(prompt).strip()

            is_empty = (
                "NO_OUTPUT" in compressed_content
                or len(compressed_content) < self.min_compressed_length
            )
            if is_empty and self.drop_empty:
                continue

            compressed_docs.append(
                Document(
                    content=compressed_content if not is_empty else doc.content,
                    metadata={
                        **doc.metadata,
                        "original_length": len(doc.content),
                        "compressed_length": len(compressed_content),
                        "was_compressed": not is_empty,
                    },
                    doc_id=doc.doc_id,
                    score=doc.score,
                )
            )

            if len(compressed_docs) >= top_k:
                break

        return compressed_docs[:top_k]
