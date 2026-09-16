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
from copy import deepcopy
from typing import List

from .base import BaseRetriever, Document, LLMClient, _validate_top_k


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
        if not isinstance(drop_not_relevant, bool):
            raise TypeError("drop_not_relevant must be a boolean")
        self.drop_not_relevant = drop_not_relevant
        self.fetch_multiplier = _validate_top_k(fetch_multiplier, name="fetch_multiplier")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        top_k = _validate_top_k(top_k)
        candidates = self.base_retriever.retrieve(
            query, top_k=top_k * self.fetch_multiplier
        )

        noted_docs: List[Document] = []
        for doc in candidates:
            prompt = NOTE_TAKING_PROMPT_FR.format(query=query, document=doc.content)
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
                        **deepcopy(doc.metadata),
                        "con_pertinence": parsed["pertinence"],
                        "con_raw_note": raw_note,
                        "original_content": doc.content,
                    },
                    doc_id=doc.doc_id,
                    score=doc.score,
                )
            )

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


class ChainOfNoteRAGEngine:
    """Sequential reading notes followed by answer generation (CoN §2.3).

    Three note kinds distinguish direct evidence, contextual inference, and
    unknown answers. Original evidence is retained alongside notes; a generated
    note is never presented as a source. ``allow_parametric_knowledge`` enables
    the paper's contextual/inherent-knowledge path, and labels that knowledge
    separately so it cannot receive a fabricated source citation.

    This runs the prompted variant. Fine-tuning a reader on notes+answers is a
    separate training step; this constructor does not silently train a model.
    """

    def __init__(self, retriever: BaseRetriever, llm_client: LLMClient, *, top_k: int = 5,
                 allow_parametric_knowledge: bool = True, **engine_kwargs):
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = _validate_top_k(top_k)
        if not isinstance(allow_parametric_knowledge, bool):
            raise TypeError("allow_parametric_knowledge must be a boolean")
        self.allow_parametric_knowledge = allow_parametric_knowledge
        if any(key in engine_kwargs for key in ("answer_prompt", "retriever", "llm_client", "top_k")):
            raise ValueError("ChainOfNoteRAGEngine owns the note/answer stages")
        self.engine_kwargs = engine_kwargs

    def ask(self, query: str, *, top_k: int | None = None):
        import json
        from .base import _snapshot_documents
        from .citations import validate_citations
        from .engine import RAGEngine
        from .reranking import _rrf_document_key
        from .schema import RAGResponse, Source

        limit = self.top_k if top_k is None else _validate_top_k(top_k)
        documents = _snapshot_documents(self.retriever.retrieve(query, top_k=limit)[:limit])
        for document in documents:
            if not document.doc_id:
                document.doc_id = "con-" + _rrf_document_key(document).removeprefix("content::")
        notes = []
        for document in documents:
            prompt = (
                "Read the document and earlier reading notes to assess the question. "
                "Return ONLY a JSON object with keys kind, summary, evidence_quotes, parametric_knowledge, limitations. "
                "kind must be direct (document answers the question), contextual (useful context, needs inference), "
                "or unknown (insufficient evidence and knowledge). evidence_quotes is a list of EXACT substrings "
                "of this document; never quote earlier notes as evidence. summary is a concise assessment. "
                "Explicitly identify contradictions with earlier notes. parametric_knowledge is a separate string "
                "of any inherent model knowledge needed; never attribute it to a document. "
                + ("You may use inherent knowledge, explicitly labelled. " if self.allow_parametric_knowledge else
                   "Do not use inherent model knowledge; parametric_knowledge must be empty. ")
                + f"\nQuestion: {query}\nEarlier notes: {json.dumps(notes, ensure_ascii=False)}"
                + f"\nDocument [source: {document.doc_id}]:\n{document.content}\nJSON:"
            )
            raw = self.llm_client.generate(prompt).strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
            try:
                note = json.loads(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("Chain-of-Note reader must return a JSON note") from exc
            if not isinstance(note, dict) or note.get("kind") not in {"direct", "contextual", "unknown"}:
                raise ValueError("Chain-of-Note kind must be direct, contextual or unknown")
            if any(not isinstance(note.get(key), str) for key in ("summary", "parametric_knowledge", "limitations")):
                raise ValueError("Chain-of-Note summary, parametric_knowledge and limitations must be strings")
            quotes = note.get("evidence_quotes")
            if not isinstance(quotes, list) or any(not isinstance(quote, str) or not quote.strip() or
                                                  quote not in document.content for quote in quotes):
                raise ValueError("Chain-of-Note evidence quotes must occur verbatim in the source")
            if note["kind"] == "direct" and not quotes:
                raise ValueError("A direct Chain-of-Note answer requires quoted evidence")
            if not self.allow_parametric_knowledge and note["parametric_knowledge"].strip():
                raise ValueError("Parametric knowledge is disabled for this Chain-of-Note engine")
            notes.append({"doc_id": document.doc_id, "kind": note["kind"], "summary": note["summary"],
                          "evidence_quotes": quotes, "parametric_knowledge": note["parametric_knowledge"],
                          "limitations": note["limitations"]})

        if not notes or all(note["kind"] == "unknown" for note in notes):
            answer = "Je ne sais pas : les documents et les notes ne permettent pas de répondre."
            return RAGResponse(query=query, answer=answer, sources=[Source.from_document(doc) for doc in documents],
                               retrieved_documents=documents, prompt="", metadata={"method": "chain-of-note", "notes": notes,
                               "abstained": True}, citation_validation=validate_citations(answer, documents))
        rendered_notes = json.dumps(notes, ensure_ascii=False).replace("{", "{{").replace("}", "}}")
        prompt = (
            "Answer the question by synthesizing the sequential reading notes. Check notes against original "
            "sources; notes are assessments, not additional evidence. Resolve conflicts or state uncertainty. "
            "Use direct evidence when available; for contextual notes, distinguish inference from observed facts. "
            "Cite only original sources as [source: doc_id], only when they support the claim. "
            "Never attach source citations to inherent/parametric knowledge. Say unknown when information is insufficient. "
            + ("Explicitly label any use of parametric knowledge. " if self.allow_parametric_knowledge else
               "Use only supplied documentary evidence. ")
            + "\nReading notes:\n" + rendered_notes + "\n\nOriginal sources:\n{context}\n\nQuestion: {query}\nAnswer:"
        )

        class Evidence(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
                return _snapshot_documents(documents[:top_k])

        response = RAGEngine(Evidence(), self.llm_client, top_k=limit, answer_prompt=prompt,
                             **self.engine_kwargs).ask(query)
        response.metadata.update(method="chain-of-note", notes=notes, abstained=False,
                                 allow_parametric_knowledge=self.allow_parametric_knowledge)
        return response
