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

from typing import Any, List

from .base import BaseRetriever, Document, LLMClient, _validate_top_k
from .reranking import ReciprocalRankFusionReranker, _rrf_document_key


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
        self.n_original = _validate_top_k(n_original, name="n_original")
        self.n_stepback = _validate_top_k(n_stepback, name="n_stepback")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        top_k = _validate_top_k(top_k)
        # 1) Génération de la question step-back
        stepback_query = self._generate_stepback(query)

        # 2) Retrieval sur les deux
        original_hits = self.base_retriever.retrieve(query, top_k=self.n_original)
        stepback_hits = self.base_retriever.retrieve(stepback_query, top_k=self.n_stepback)

        # Rank fusion preserves both evidence streams without comparing scores
        # of different queries or placing all specific hits before abstractions.
        original_keys = {_rrf_document_key(doc) for doc in original_hits}
        stepback_keys = {_rrf_document_key(doc) for doc in stepback_hits}
        merged = ReciprocalRankFusionReranker().fuse([original_hits, stepback_hits], top_k=top_k)
        for document in merged:
            key = _rrf_document_key(document)
            origin = "both" if key in original_keys & stepback_keys else "original" if key in original_keys else "stepback"
            document.metadata.update(retrieval_source=origin, stepback_query=stepback_query)
        return merged

    # ------------------------------------------------------------------ #
    def _generate_stepback(self, query: str) -> str:
        prompt = STEP_BACK_PROMPT_FR.format(query=query)
        stepback = self.llm_client.generate(prompt).strip()
        # Garder une seule ligne si le LLM est bavard
        return stepback.split("\n")[0].strip() or query


class StepBackRAGEngine:
    """Abstraction → evidence-based principles → original-question reasoning.

    Implements the two-stage method of Zheng et al. (arXiv:2310.06117),
    retaining the original sources as evidence for both generation stages.
    The intermediate abstraction answer is never promoted to a source.
    """

    def __init__(self, retriever: BaseRetriever, llm_client: LLMClient, *,
                 n_original: int = 3, n_stepback: int = 3, **engine_kwargs: Any):
        self.retriever = StepBackRetriever(retriever, llm_client, n_original, n_stepback)
        self.llm_client = llm_client
        if "answer_prompt" in engine_kwargs or "retriever" in engine_kwargs or "llm_client" in engine_kwargs:
            raise ValueError("StepBackRAGEngine owns the staged answer prompt and retrieval")
        self.engine_kwargs = engine_kwargs

    def ask(self, query: str, *, top_k: int | None = None):
        from .base import _snapshot_documents
        from .engine import RAGEngine
        from .pipeline import AdvancedRAGPipeline

        limit = self.retriever.n_original + self.retriever.n_stepback if top_k is None else _validate_top_k(top_k)
        documents = self.retriever.retrieve(query, top_k=limit)
        for document in documents:
            if not document.doc_id:
                document.doc_id = "stepback-" + _rrf_document_key(document).removeprefix("content::")
        stepback = documents[0].metadata["stepback_query"] if documents else query
        general = [doc for doc in documents if doc.metadata["retrieval_source"] in {"both", "stepback"}]
        principle_prompt = (
            "Réponds à la question générale à partir des sources ci-dessous. Dégage les concepts et faits "
            "utiles, préserve leurs conditions et exceptions, cite [source: doc_id]. "
            "Si les sources sont insuffisantes, indique-le.\n\n"
            f"Question générale : {stepback}\n\nSources :\n{AdvancedRAGPipeline._format_context(general)}"
        )
        principles = self.llm_client.generate(principle_prompt) if general else "Preuves générales insuffisantes."
        # Escape generated text before inserting it in a later .format template.
        abstraction = f"Question générale : {stepback}\nSynthèse provisoire : {principles}".replace("{", "{{").replace("}", "}}")
        prompt = (
            "Réponds à la question initiale en appliquant les principes généraux aux faits spécifiques. "
            "La synthèse provisoire ci-dessous est un raisonnement auxiliaire, pas une nouvelle source. "
            "Vérifie-la contre les extraits, tiens compte des exceptions et n'invente pas de faits. "
            "Cite uniquement les sources fournies sous la forme [source: doc_id]. "
            "Si les preuves sont insuffisantes, indique-le.\n\n" + abstraction +
            "\n\nExtraits :\n{context}\n\nQuestion : {query}\n\nRéponse :"
        )

        class Evidence(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
                return _snapshot_documents(documents[:top_k])

        response = RAGEngine(Evidence(), self.llm_client, answer_prompt=prompt,
                             **self.engine_kwargs).ask(query, top_k=limit)
        response.metadata.update(stepback_query=stepback, stepback_principles=principles, method="step-back")
        return response
