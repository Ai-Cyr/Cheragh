"""
Pipeline orchestrateur
=======================

Combine plusieurs techniques dans un pipeline de bout en bout :
retrieval → (optionnel) reranking → génération de la réponse.

Exemple d'architecture production recommandée :
    HybridSearch → RAG-Fusion → Cross-Encoder Rerank → LLM
"""
from __future__ import annotations

from typing import List

from .base import BaseRetriever, Document, LLMClient
from .citations import citation_location


DEFAULT_ANSWER_PROMPT_FR = """Tu es un assistant qui répond en t'appuyant UNIQUEMENT sur les extraits fournis.
Si l'information n'est pas dans les extraits, dis-le explicitement.
Cite les sources en fin de phrase sous la forme [source: doc_id].
Quand un extrait affiche une localisation (page, lignes ou caractères), utilise-la pour vérifier précisément le passage.

Extraits :
{context}

Question : {query}

Réponse :"""


class AdvancedRAGPipeline:
    """
    Pipeline complet : retriever + LLM.

    Parameters
    ----------
    retriever : BaseRetriever
        N'importe quel retriever de la bibliothèque (ou une composition).
    llm_client : LLMClient
        LLM pour la génération finale.
    answer_prompt : str
        Prompt template avec placeholders {context} et {query}.
    top_k : int, default=5
        Nombre de documents à passer au LLM.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_client: LLMClient,
        answer_prompt: str = DEFAULT_ANSWER_PROMPT_FR,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.answer_prompt = answer_prompt
        self.top_k = top_k

    def run(self, query: str) -> dict:
        """Exécute le pipeline et retourne la réponse + les sources."""
        docs = self.retriever.retrieve(query, top_k=self.top_k)
        context = self._format_context(docs)
        prompt = self.answer_prompt.format(context=context, query=query)
        answer = self.llm_client.generate(prompt)
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "doc_id": d.doc_id,
                    "score": d.score,
                    "preview": d.content[:200],
                    "location": citation_location(d),
                }
                for d in docs
            ],
        }

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        parts = []
        for i, d in enumerate(docs, start=1):
            src = d.doc_id or f"doc_{i}"
            location = citation_location(d)
            location_line = f"\nlocation: {location}" if location else ""
            parts.append(f"[source: {src}]{location_line}\n{d.content}")
        return "\n\n---\n\n".join(parts)
