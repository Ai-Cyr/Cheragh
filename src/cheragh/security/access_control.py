"""Metadata-based access control for RAG documents and engines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..base import BaseRetriever, Document, LLMClient


_CLASSIFICATION_ORDER = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
    "secret": 4,
}


@dataclass
class Principal:
    """Identity and authorization context for a RAG request."""

    user_id: str
    roles: set[str] = field(default_factory=set)
    tenant_ids: set[str] = field(default_factory=set)
    collection_ids: set[str] = field(default_factory=set)
    attributes: dict[str, Any] = field(default_factory=dict)
    max_classification: str = "internal"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Principal":
        return cls(
            user_id=str(data.get("user_id") or data.get("id") or "anonymous"),
            roles=set(data.get("roles") or []),
            tenant_ids=set(data.get("tenant_ids") or data.get("tenants") or []),
            collection_ids=set(data.get("collection_ids") or data.get("collections") or []),
            attributes=dict(data.get("attributes") or {}),
            max_classification=str(data.get("max_classification", "internal")),
        )


@dataclass
class AccessDecision:
    """Decision returned by :class:`AccessPolicy`."""

    allowed: bool
    reason: str = "allowed"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessPolicy:
    """Metadata-based access policy.

    A document is allowed when all configured checks pass. Supported document
    metadata fields include ``tenant_id``, ``collection_id``, ``allowed_users``,
    ``allowed_roles``, ``denied_users``, ``denied_roles`` and ``classification``.
    """

    require_tenant_match: bool = True
    require_collection_match: bool = False
    allow_public: bool = True
    default_classification: str = "internal"
    metadata_equals: dict[str, Any] = field(default_factory=dict)
    metadata_in: dict[str, set[Any]] = field(default_factory=dict)

    def authorize(self, document: Document, principal: Principal | Mapping[str, Any] | None = None) -> AccessDecision:
        principal_obj = _coerce_principal(principal)
        meta = document.metadata or {}

        denied_users = set(_as_iterable(meta.get("denied_users")))
        denied_roles = set(_as_iterable(meta.get("denied_roles")))
        if principal_obj.user_id in denied_users:
            return AccessDecision(False, "user_explicitly_denied")
        if principal_obj.roles & denied_roles:
            return AccessDecision(False, "role_explicitly_denied")

        allowed_users = set(_as_iterable(meta.get("allowed_users")))
        allowed_roles = set(_as_iterable(meta.get("allowed_roles")))
        if allowed_users and principal_obj.user_id not in allowed_users:
            return AccessDecision(False, "user_not_allowed")
        if allowed_roles and not (principal_obj.roles & allowed_roles):
            return AccessDecision(False, "role_not_allowed")

        tenant_id = meta.get("tenant_id")
        if self.require_tenant_match and tenant_id is not None:
            if tenant_id not in principal_obj.tenant_ids and "admin" not in principal_obj.roles:
                return AccessDecision(False, "tenant_mismatch", {"tenant_id": tenant_id})

        collection_id = meta.get("collection_id")
        if self.require_collection_match and collection_id is not None:
            if collection_id not in principal_obj.collection_ids and "admin" not in principal_obj.roles:
                return AccessDecision(False, "collection_mismatch", {"collection_id": collection_id})

        classification = str(meta.get("classification", self.default_classification)).lower()
        if self.allow_public and classification == "public":
            pass
        elif _CLASSIFICATION_ORDER.get(classification, 1) > _CLASSIFICATION_ORDER.get(principal_obj.max_classification.lower(), 1):
            return AccessDecision(False, "classification_too_high", {"classification": classification})

        for key, expected in self.metadata_equals.items():
            if meta.get(key) != expected:
                return AccessDecision(False, "metadata_equals_failed", {"key": key, "expected": expected})
        for key, allowed_values in self.metadata_in.items():
            if meta.get(key) not in allowed_values:
                return AccessDecision(False, "metadata_in_failed", {"key": key, "allowed": list(allowed_values)})
        return AccessDecision(True)

    def filter_documents(self, documents: Iterable[Document], principal: Principal | Mapping[str, Any] | None = None) -> list[Document]:
        filtered: list[Document] = []
        for doc in documents:
            decision = self.authorize(doc, principal)
            if decision.allowed:
                enriched = Document(doc.content, metadata={**doc.metadata, "access_decision": decision.reason}, doc_id=doc.doc_id, score=doc.score)
                filtered.append(enriched)
        return filtered


def filter_documents_for_principal(
    documents: Iterable[Document],
    principal: Principal | Mapping[str, Any] | None,
    policy: AccessPolicy | None = None,
) -> list[Document]:
    """Filter documents according to ``policy`` and ``principal``."""

    return (policy or AccessPolicy()).filter_documents(documents, principal)


class AccessControlledRetriever(BaseRetriever):
    """Retriever wrapper that filters retrieved documents by access policy."""

    def __init__(
        self,
        retriever: BaseRetriever,
        principal: Principal | Mapping[str, Any] | None,
        policy: AccessPolicy | None = None,
        overfetch_factor: int = 4,
    ):
        self.retriever = retriever
        self.principal = _coerce_principal(principal)
        self.policy = policy or AccessPolicy()
        self.overfetch_factor = max(1, overfetch_factor)
        self.last_denied_count = 0

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        docs = self.retriever.retrieve(query, top_k=top_k * self.overfetch_factor)
        allowed = self.policy.filter_documents(docs, self.principal)
        self.last_denied_count = max(0, len(docs) - len(allowed))
        return allowed[:top_k]


class AccessControlledRAGEngine:
    """RAGEngine wrapper that enforces access control before generation."""

    def __init__(
        self,
        base_engine: Any,
        policy: AccessPolicy | None = None,
        default_principal: Principal | Mapping[str, Any] | None = None,
        overfetch_factor: int = 4,
    ):
        self.base_engine = base_engine
        self.policy = policy or AccessPolicy()
        self.default_principal = _coerce_principal(default_principal)
        self.overfetch_factor = overfetch_factor

    def for_principal(self, principal: Principal | Mapping[str, Any]) -> Any:
        from ..engine import RAGEngine
        retriever = AccessControlledRetriever(
            self.base_engine.retriever,
            principal=principal,
            policy=self.policy,
            overfetch_factor=self.overfetch_factor,
        )
        return RAGEngine(
            retriever=retriever,
            llm_client=self.base_engine.llm_client,
            answer_prompt=self.base_engine.answer_prompt,
            top_k=self.base_engine.top_k,
            strict_grounding=self.base_engine.strict_grounding,
            min_score=self.base_engine.min_score,
            require_citations=self.base_engine.require_citations,
            flag_unsourced_sentences=self.base_engine.flag_unsourced_sentences,
            compressor=self.base_engine.compressor,
            query_transformer=self.base_engine.query_transformer,
            trace_enabled=self.base_engine.trace_enabled,
        )

    def ask(self, query: str, principal: Principal | Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        principal_obj = _coerce_principal(principal) if principal is not None else self.default_principal
        engine = self.for_principal(principal_obj)
        response = engine.ask(query, **kwargs)
        response.metadata.setdefault("access_control", {})
        response.metadata["access_control"].update(
            {
                "enabled": True,
                "user_id": principal_obj.user_id,
                "denied_documents": getattr(engine.retriever, "last_denied_count", 0),
            }
        )
        return response


def _as_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _coerce_principal(principal: Principal | Mapping[str, Any] | None) -> Principal:
    if isinstance(principal, Principal):
        return principal
    if isinstance(principal, Mapping):
        return Principal.from_dict(principal)
    return Principal(user_id="anonymous")
