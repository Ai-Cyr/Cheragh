"""
Technique 13 : HyQE – Hypothetical Question Embeddings — version persistable.

IMPORTANT : HyQE est coûteuse à l'indexation (N appels LLM par document).
Le cache sauvegarde AUSSI les questions générées, pas seulement les
embeddings, pour éviter de réappeler le LLM.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

from .base import BaseRetriever, Document, EmbeddingModel, LLMClient, cosine_similarity
from .cache import hash_documents, embedder_fingerprint, load_cache, save_cache


QUESTION_GENERATION_PROMPT_FR = """Tu reçois un extrait de document. Génère {n_questions} questions distinctes et pertinentes auxquelles CE extrait permet de répondre de façon directe et factuelle.

Règles :
- Les questions doivent être naturelles (comme un utilisateur réel les poserait).
- Elles doivent couvrir les DIFFÉRENTS faits contenus dans l'extrait.
- Varie les formulations (question directe, indirecte, avec "comment", "quel", "pourquoi", etc.).

Réponds UNIQUEMENT avec les questions, une par ligne, sans numérotation ni préambule.

Extrait :
{document}

Questions :"""


class HyQERetriever(BaseRetriever):
    _CACHEABLE_VERSION = 1

    def __init__(
        self,
        documents: List[Document],
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        n_questions_per_doc: int = 5,
        include_original_content: bool = True,
        cache_path: Optional[str] = None,
    ):
        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.n_questions_per_doc = n_questions_per_doc
        self.include_original_content = include_original_content
        self._cache_path = cache_path

        self._index_texts: List[str] = []
        self._index_to_doc: List[int] = []
        self._index_embeddings: Optional[np.ndarray] = None

        if not self._try_load_cache():
            self._build_index()
            self._save_cache()

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        query_vec = self.embedding_model.embed_query(query)
        scores = cosine_similarity(query_vec, self._index_embeddings)
        fetch = max(top_k * 4, 20)
        top_idx = np.argsort(scores)[::-1][:fetch]

        best_per_doc: Dict[int, float] = {}
        best_match_text: Dict[int, str] = {}
        order: List[int] = []
        for i in top_idx:
            di = self._index_to_doc[i]
            s = float(scores[i])
            if di not in best_per_doc or s > best_per_doc[di]:
                best_per_doc[di] = s
                best_match_text[di] = self._index_texts[i]
                if di not in order:
                    order.append(di)

        ordered_docs = sorted(order, key=lambda di: best_per_doc[di], reverse=True)
        return [
            Document(
                content=self.documents[di].content,
                metadata={**self.documents[di].metadata, "hyqe_best_match": best_match_text[di][:300]},
                doc_id=self.documents[di].doc_id,
                score=best_per_doc[di],
            )
            for di in ordered_docs[:top_k]
        ]

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #
    def _extra_fp(self) -> str:
        return (
            f"v={self._CACHEABLE_VERSION};"
            f"n_q={self.n_questions_per_doc};"
            f"inc_orig={int(self.include_original_content)}"
        )

    def _try_load_cache(self) -> bool:
        if not self._cache_path:
            return False
        state = load_cache(
            path=self._cache_path,
            expected_class=self.__class__.__name__,
            expected_content_hash=hash_documents(self.documents),
            expected_embedder_fp=embedder_fingerprint(self.embedding_model),
            expected_extra_fp=self._extra_fp(),
        )
        if state is None:
            return False
        self._index_texts = state["index_texts"]
        self._index_to_doc = state["index_to_doc"]
        self._index_embeddings = state["index_embeddings"]
        return True

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        save_cache(
            path=self._cache_path,
            retriever_class=self.__class__.__name__,
            content_hash=hash_documents(self.documents),
            embedder_fp=embedder_fingerprint(self.embedding_model),
            extra_fingerprint=self._extra_fp(),
            state={
                "index_texts": self._index_texts,       # inclut les questions LLM-générées
                "index_to_doc": self._index_to_doc,
                "index_embeddings": self._index_embeddings,
            },
        )

    # ------------------------------------------------------------------ #
    def _build_index(self) -> None:
        for di, doc in enumerate(self.documents):
            questions = self._generate_questions(doc.content)
            for q in questions:
                self._index_texts.append(q)
                self._index_to_doc.append(di)
            if self.include_original_content:
                self._index_texts.append(doc.content)
                self._index_to_doc.append(di)
        self._index_embeddings = self.embedding_model.embed_documents(self._index_texts)

    def _generate_questions(self, content: str) -> List[str]:
        prompt = QUESTION_GENERATION_PROMPT_FR.format(
            n_questions=self.n_questions_per_doc, document=content[:3000]
        )
        raw = self.llm_client.generate(prompt)
        lines = [re.sub(r"^[\d\.\)\-\s•]+", "", line).strip() for line in raw.split("\n")]
        return [q for q in lines if len(q) > 5 and "?" in q][: self.n_questions_per_doc]
