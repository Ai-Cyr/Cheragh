"""Metadata-based access control for RAG documents and engines."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Iterable, Mapping

from ..base import BaseRetriever, Document, _snapshot_document, _validate_top_k


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
    strict: bool = False

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

        require_tenant = self.require_tenant_match or self.strict
        tenant_id = meta.get("tenant_id")
        if require_tenant:
            if tenant_id is None:
                return AccessDecision(False, "tenant_metadata_missing")
            elif tenant_id not in principal_obj.tenant_ids and "admin" not in principal_obj.roles:
                return AccessDecision(False, "tenant_mismatch", {"tenant_id": tenant_id})

        require_collection = self.require_collection_match or self.strict
        collection_id = meta.get("collection_id")
        if require_collection:
            if collection_id is None:
                return AccessDecision(False, "collection_metadata_missing")
            elif collection_id not in principal_obj.collection_ids and "admin" not in principal_obj.roles:
                return AccessDecision(False, "collection_mismatch", {"collection_id": collection_id})

        classification = str(meta.get("classification", self.default_classification)).lower()
        if classification not in _CLASSIFICATION_ORDER:
            return AccessDecision(False, "unknown_classification", {"classification": classification})
        principal_classification = principal_obj.max_classification.lower()
        if self.allow_public and classification == "public":
            pass
        elif principal_classification not in _CLASSIFICATION_ORDER:
            return AccessDecision(
                False,
                "unknown_principal_classification",
                {"classification": principal_classification},
            )
        elif _CLASSIFICATION_ORDER.get(classification, 1) > _CLASSIFICATION_ORDER.get(principal_classification, 1):
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
                enriched = _snapshot_document(doc)
                enriched.metadata["access_decision"] = decision.reason
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
        max_candidates: int = 10_000,
    ):
        self.retriever = retriever
        self.principal = _coerce_principal(principal)
        self.policy = policy if policy is not None else AccessPolicy()
        self.overfetch_factor = _validate_top_k(overfetch_factor, name="overfetch_factor")
        self.max_candidates = _validate_top_k(max_candidates, name="max_candidates")
        self.last_denied_count = 0
        self.last_scanned_count = 0
        self.last_candidate_limit_reached = False

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        top_k = _validate_top_k(top_k)
        from ..long_rag import LongRAGRetriever

        if isinstance(self.retriever, LongRAGRetriever):
            return [group.as_document() for group in self.retrieve_groups(query, top_k=top_k)]
        # Authorize original evidence before exposing derived summaries, and
        # rebuild the scoped view for each request so revocations take effect.
        authorize_sources = getattr(self.retriever, "for_principal", None)
        authorized = authorize_sources(self.principal, policy=self.policy) if callable(authorize_sources) else None
        if callable(authorize_sources) and not callable(getattr(authorized, "retrieve", None)):
            raise TypeError("for_principal must return an authorized retriever")
        if authorized is not None:
            limit = min(top_k, self.max_candidates)
            documents = [_snapshot_document(document) for document in islice(authorized.retrieve(query, top_k=limit), limit)]
            self.last_scanned_count = len(documents)
            self.last_denied_count = 0
            self.last_candidate_limit_reached = top_k > self.max_candidates
            return documents
        candidate_limit = min(
            self.max_candidates,
            max(top_k, top_k * self.overfetch_factor),
        )
        seen: set[tuple[int, str, str]] = set()
        allowed: list[Document] = []
        self.last_candidate_limit_reached = False

        while True:
            docs = list(islice(self.retriever.retrieve(query, top_k=candidate_limit), candidate_limit))
            unseen: list[Document] = []
            for rank, document in enumerate(docs):
                key = (rank, str(document.doc_id or ""), document.content)
                if key in seen:
                    continue
                seen.add(key)
                unseen.append(document)

            allowed.extend(self.policy.filter_documents(unseen, self.principal))
            if len(allowed) >= top_k:
                break

            # A short page means the ranked collection has been exhausted. A
            # full page with no new ranked candidates means the wrapped
            # retriever cannot expose a deeper prefix through its API.
            if len(docs) < candidate_limit or not unseen:
                break
            if candidate_limit >= self.max_candidates:
                self.last_candidate_limit_reached = True
                break
            candidate_limit = min(
                self.max_candidates,
                max(candidate_limit + 1, candidate_limit * 2),
            )

        self.last_scanned_count = len(seen)
        self.last_denied_count = max(0, self.last_scanned_count - len(allowed))
        return allowed[:top_k]

    def retrieve_groups(self, query: str, top_k: int = 5, **kwargs: Any) -> list[Any]:
        """Apply LongRAG authorization to original sources before regrouping."""
        from ..long_rag import LongRAGRetriever

        if not isinstance(self.retriever, LongRAGRetriever):
            raise TypeError("group retrieval requires a LongRAGRetriever")
        for name in ("principal", "access_policy"):
            if kwargs.pop(name, None) is not None:
                raise ValueError(f"cannot override {name} on an access-controlled retriever")
        top_k = _validate_top_k(top_k)
        limit = min(top_k, self.max_candidates)
        groups = self.retriever.retrieve_groups(
            query, top_k=limit, principal=self.principal, access_policy=self.policy, **kwargs
        )
        self.last_scanned_count = len(groups)
        self.last_denied_count = 0
        self.last_candidate_limit_reached = top_k > self.max_candidates
        return groups


class AccessControlledRAGEngine:
    """RAGEngine wrapper that enforces access control before generation."""

    def __init__(
        self,
        base_engine: Any,
        policy: AccessPolicy | None = None,
        default_principal: Principal | Mapping[str, Any] | None = None,
        overfetch_factor: int = 4,
        max_candidates: int = 10_000,
    ):
        self.base_engine = base_engine
        self.policy = policy if policy is not None else AccessPolicy()
        self.default_principal = _coerce_principal(default_principal)
        self.overfetch_factor = _validate_top_k(overfetch_factor, name="overfetch_factor")
        self.max_candidates = _validate_top_k(max_candidates, name="max_candidates")

    def for_principal(self, principal: Principal | Mapping[str, Any]) -> Any:
        retriever = AccessControlledRetriever(
            self.base_engine.retriever,
            principal=principal,
            policy=self.policy,
            overfetch_factor=self.overfetch_factor,
            max_candidates=self.max_candidates,
        )
        ask_owner = getattr(getattr(self.base_engine, "ask", None), "__self__", None)
        engine_template = self.base_engine
        if (
            ask_owner is not None
            and ask_owner is not self.base_engine
            and not isinstance(ask_owner, type)
            and hasattr(ask_owner, "retriever")
        ):
            engine_template = ask_owner
        # Rebuilding a RAGEngine from a subset of fields silently discarded
        # context packing, cache settings and trace export configuration. A
        # shallow request-local clone preserves the complete engine contract
        # while replacing only the retrieval boundary that needs authorization.
        # A transparent proxy may expose an ``ask`` method bound to its wrapped
        # engine; clone that method owner so the replacement retriever is the one
        # actually used by the request.
        engine = copy(engine_template)
        if engine is engine_template:
            raise TypeError("base_engine must support request-local shallow copying")
        engine.retriever = retriever
        # Corrective RAG may obtain replacement evidence from a second source.
        # It must cross the same authorization boundary before grading or use.
        external_retriever = getattr(engine_template, "external_retriever", None)
        if external_retriever is not None:
            engine.external_retriever = AccessControlledRetriever(
                external_retriever, principal=principal, policy=self.policy,
                overfetch_factor=self.overfetch_factor, max_candidates=self.max_candidates,
            )
        return engine

    def ask(self, query: str, principal: Principal | Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        if kwargs.get("top_k") is not None:
            kwargs["top_k"] = _validate_top_k(kwargs["top_k"])
        principal_obj = _coerce_principal(principal) if principal is not None else self.default_principal
        engine = self.for_principal(principal_obj)
        response = engine.ask(query, **kwargs)
        retrievers = [engine.retriever]
        external_retriever = getattr(engine, "external_retriever", None)
        if isinstance(external_retriever, AccessControlledRetriever):
            retrievers.append(external_retriever)
        if "access_control_enabled" in response.metadata:
            response.metadata["access_control_enabled"] = True
        response.metadata.setdefault("access_control", {})
        response.metadata["access_control"].update(
            {
                "enabled": True,
                "user_id": principal_obj.user_id,
                "denied_documents": sum(getattr(item, "last_denied_count", 0) for item in retrievers),
                "scanned_documents": sum(getattr(item, "last_scanned_count", 0) for item in retrievers),
                "candidate_limit_reached": any(
                    getattr(item, "last_candidate_limit_reached", False) for item in retrievers
                ),
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
