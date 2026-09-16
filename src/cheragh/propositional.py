"""
Technique 18 : Propositional Indexing — version persistable.

Comme HyQE, Propositional est coûteuse à l'indexation (LLM extrait les
propositions de chaque chunk). Le cache sauvegarde les propositions
générées ET leurs embeddings.
"""
from __future__ import annotations

import re
import json
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

from .base import BaseRetriever, Document, EmbeddingModel, LLMClient, _snapshot_documents, _validate_top_k, cosine_similarity
from .cache import hash_documents, embedder_fingerprint, load_cache, save_cache
from .cache.decorators import _component_fingerprint


PROPOSITION_EXTRACTION_PROMPT_FR = """Décompose l'extrait ci-dessous en propositions atomiques.

Une proposition atomique est un énoncé court (une phrase simple) qui :
- exprime UN SEUL fait ou UNE SEULE règle,
- est AUTONOME (compréhensible sans lire le reste de l'extrait : remplace "il/elle/cela/ce dernier" par la valeur explicite),
- est factuelle (pas de question, pas de commentaire).

Réponds UNIQUEMENT avec les propositions, une par ligne, sans numérotation ni préambule.

Extrait :
{document}

Propositions :"""


class PropositionalRetriever(BaseRetriever):
    _CACHEABLE_VERSION = 2

    def __init__(
        self,
        documents: List[Document],
        embedding_model: EmbeddingModel,
        llm_client: LLMClient | None = None,
        return_propositions: bool = False,
        max_propositions_per_doc: int = 20,
        cache_path: Optional[str] = None,
        allow_unsafe_pickle: bool = False,
        *,
        propositionizer: TransformersPropositionizer | None = None,
    ):
        if llm_client is None and propositionizer is None:
            raise ValueError("Provide llm_client or a trained propositionizer")
        if llm_client is not None and propositionizer is not None:
            raise ValueError("Choose llm_client or propositionizer, not both")
        self.documents = _snapshot_documents(documents)
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.propositionizer = propositionizer
        if not isinstance(return_propositions, bool):
            raise TypeError("return_propositions must be a boolean")
        self.return_propositions = return_propositions
        self.max_propositions_per_doc = _validate_top_k(max_propositions_per_doc, name="max_propositions_per_doc")
        self._cache_path = cache_path
        self._allow_unsafe_pickle = allow_unsafe_pickle

        self._propositions: List[str] = []
        self._prop_to_doc: List[int] = []
        self._prop_embeddings: Optional[np.ndarray] = None

        if not self._try_load_cache():
            self._build_index()
            self._save_cache()

        if not self._propositions:
            raise ValueError("Aucune proposition extraite. Vérifier le LLM / les documents.")

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        top_k = _validate_top_k(top_k)
        query_vec = self.embedding_model.embed_query(query)
        scores = cosine_similarity(query_vec, self._prop_embeddings)

        if self.return_propositions:
            top_idx = np.argsort(-scores, kind="stable")[:top_k]
            return [
                Document(
                    content=self._propositions[i],
                    metadata={
                        **deepcopy(self.documents[self._prop_to_doc[i]].metadata),
                        "source_doc_id": self.documents[self._prop_to_doc[i]].doc_id,
                    },
                    doc_id=f"prop::{i}",
                    score=float(scores[i]),
                )
                for i in top_idx
            ]

        top_idx = np.argsort(-scores, kind="stable")

        best_per_doc: Dict[int, float] = {}
        best_match: Dict[int, str] = {}
        order: List[int] = []
        for i in top_idx:
            di = self._prop_to_doc[i]
            s = float(scores[i])
            if di not in best_per_doc or s > best_per_doc[di]:
                best_per_doc[di] = s
                best_match[di] = self._propositions[i]
                if di not in order:
                    order.append(di)

        ordered = sorted(order, key=lambda di: best_per_doc[di], reverse=True)
        return [
            Document(
                content=self.documents[di].content,
                metadata={**deepcopy(self.documents[di].metadata), "matched_proposition": best_match[di]},
                doc_id=self.documents[di].doc_id,
                score=best_per_doc[di],
            )
            for di in ordered[:top_k]
        ]

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #
    def _extra_fp(self) -> str:
        # return_propositions est query-time, pas besoin dans le fingerprint
        return (f"v={self._CACHEABLE_VERSION};max_props={self.max_propositions_per_doc};"
                f"generator={_component_fingerprint(self.propositionizer or self.llm_client, purpose='llm')}")

    def _try_load_cache(self) -> bool:
        if not self._cache_path:
            return False
        state = load_cache(
            path=self._cache_path,
            expected_class=self.__class__.__name__,
            expected_content_hash=hash_documents(self.documents),
            expected_embedder_fp=embedder_fingerprint(self.embedding_model),
            expected_extra_fp=self._extra_fp(),
            allow_unsafe_pickle=self._allow_unsafe_pickle,
        )
        if state is None:
            return False
        self._propositions = state["propositions"]
        self._prop_to_doc = state["prop_to_doc"]
        self._prop_embeddings = state["prop_embeddings"]
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
                "propositions": self._propositions,   # inclut les props LLM-générées
                "prop_to_doc": self._prop_to_doc,
                "prop_embeddings": self._prop_embeddings,
            },
        )

    # ------------------------------------------------------------------ #
    def _build_index(self) -> None:
        for di, doc in enumerate(self.documents):
            props = (self.propositionizer.extract(doc) if self.propositionizer is not None
                     else self._extract_propositions(doc.content))
            for p in props[: self.max_propositions_per_doc]:
                self._propositions.append(p)
                self._prop_to_doc.append(di)
        if self._propositions:
            self._prop_embeddings = self.embedding_model.embed_documents(self._propositions)

    def _extract_propositions(self, content: str) -> List[str]:
        prompt = PROPOSITION_EXTRACTION_PROMPT_FR.format(document=content)
        raw = self.llm_client.generate(prompt)
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = None
        lines = parsed if isinstance(parsed, list) else raw.splitlines()
        lines = [re.sub(r"^\s*(?:\d+[.)]\s*|[-•]\s*)", "", line).strip()
                 for line in lines if isinstance(line, str)]
        return [
            line for line in dict.fromkeys(lines)
            if len(line) > 10 and not line.endswith("?") and len(line.split()) >= 3
        ]


class TransformersPropositionizer:
    """The authors' trained Flan-T5 propositionizer with its original format.

    Source: github.com/chentong0/factoid-wiki. Pass pre-chunked documents if
    an input exceeds ``max_input_tokens``; the adapter refuses silent truncation.
    It loads no remote Python code, and parses only a JSON list of propositions.
    """

    def __init__(self, model_name: str = "chentong00/propositionizer-wiki-flan-t5-large", *,
                 model=None, tokenizer=None, device: str = "cpu", revision: str | None = None,
                 max_input_tokens: int = 512, max_new_tokens: int = 512):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("TransformersPropositionizer requires cheragh[research]") from exc
        if (model is None) != (tokenizer is None):
            raise ValueError("Inject both model and tokenizer or neither")
        self.model_name = model_name
        self.revision = revision
        self.max_input_tokens = _validate_top_k(max_input_tokens, name="max_input_tokens")
        self.max_new_tokens = _validate_top_k(max_new_tokens, name="max_new_tokens")
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            model_name, revision=revision, trust_remote_code=False)
        self.model = model if model is not None else AutoModelForSeq2SeqLM.from_pretrained(
            model_name, revision=revision, trust_remote_code=False)
        self.device = device
        self.model.to(device).eval()
        self._torch = torch

    def extract(self, document: Document) -> list[str]:
        text = (f"Title: {document.metadata.get('title', '')}. "
                f"Section: {document.metadata.get('section', '')}. Content: {document.content}")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        if inputs["input_ids"].shape[-1] > self.max_input_tokens:
            raise ValueError("Propositionizer input exceeds max_input_tokens; chunk the document before extraction")
        with self._torch.inference_mode():
            output = self.model.generate(**{key: value.to(self.device) for key, value in inputs.items()},
                                         max_new_tokens=self.max_new_tokens, do_sample=False)
        raw = self.tokenizer.decode(output[0], skip_special_tokens=True)
        try:
            propositions = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("The trained propositionizer did not return a JSON list") from exc
        if not isinstance(propositions, list) or any(not isinstance(item, str) or not item.strip() for item in propositions):
            raise ValueError("The trained propositionizer must return a list of non-empty strings")
        return list(dict.fromkeys(item.strip() for item in propositions))

    def get_fingerprint(self) -> str:
        return f"propositionizer::{self.model_name}::{self.revision}::{self.max_input_tokens}::{self.max_new_tokens}"
