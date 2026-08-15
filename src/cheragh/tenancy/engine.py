"""Multi-tenancy support for RAG deployments."""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Mapping

from ..base import BaseRetriever, Document, _validate_top_k
from ..security import AccessPolicy, Principal, AccessControlledRAGEngine


@dataclass
class TenantConfig:
    """Tenant metadata and defaults."""

    tenant_id: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    default_collection_id: str | None = None
    enabled: bool = True


@dataclass
class CollectionBinding:
    """Engine/retriever bound to a tenant collection."""

    tenant_id: str
    collection_id: str
    target: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class TenantRegistry:
    """In-memory registry of tenant collections and their RAG targets."""

    def __init__(self):
        self.tenants: dict[str, TenantConfig] = {}
        self.collections: dict[tuple[str, str], CollectionBinding] = {}

    def add_tenant(self, tenant_id: str, name: str | None = None, **metadata: Any) -> TenantConfig:
        cfg = TenantConfig(tenant_id=tenant_id, name=name, metadata=metadata)
        self.tenants[tenant_id] = cfg
        return cfg

    def add_collection(
        self,
        tenant_id: str,
        collection_id: str,
        target: Any,
        default: bool = False,
        **metadata: Any,
    ) -> CollectionBinding:
        if tenant_id not in self.tenants:
            self.add_tenant(tenant_id)
        binding = CollectionBinding(tenant_id=tenant_id, collection_id=collection_id, target=target, metadata=metadata)
        self.collections[(tenant_id, collection_id)] = binding
        if default or self.tenants[tenant_id].default_collection_id is None:
            self.tenants[tenant_id].default_collection_id = collection_id
        return binding

    def get_collection(self, tenant_id: str, collection_id: str | None = None) -> CollectionBinding:
        tenant = self.tenants.get(tenant_id)
        if tenant is None:
            raise KeyError(f"Unknown tenant: {tenant_id}")
        if not tenant.enabled:
            raise PermissionError(f"Tenant is disabled: {tenant_id}")
        cid = collection_id or tenant.default_collection_id
        if not cid:
            raise KeyError(f"Tenant {tenant_id!r} has no default collection")
        try:
            return self.collections[(tenant_id, cid)]
        except KeyError as exc:
            raise KeyError(f"Unknown collection {cid!r} for tenant {tenant_id!r}") from exc

    def list_collections(self, tenant_id: str | None = None) -> list[CollectionBinding]:
        if tenant_id is None:
            return list(self.collections.values())
        return [binding for (tid, _), binding in self.collections.items() if tid == tenant_id]


class MultiTenantRAGEngine:
    """Route RAG calls to tenant-specific engines/collections.

    Targets can be ``RAGEngine`` instances, objects exposing ``ask`` or
    ``retrieve``, or bare retrievers. Optional access control is applied when the
    selected target is a ``RAGEngine``.
    """

    def __init__(
        self,
        registry: TenantRegistry | None = None,
        access_policy: AccessPolicy | None = None,
        enforce_access_control: bool = True,
    ):
        self.registry = registry or TenantRegistry()
        self.access_policy = access_policy or AccessPolicy(
            require_tenant_match=True,
            require_collection_match=True,
            strict=True,
        )
        self.enforce_access_control = enforce_access_control

    def add_tenant(self, tenant_id: str, name: str | None = None, **metadata: Any) -> TenantConfig:
        return self.registry.add_tenant(tenant_id, name=name, **metadata)

    def add_collection(
        self,
        tenant_id: str,
        collection_id: str,
        target: Any,
        default: bool = False,
        **metadata: Any,
    ) -> CollectionBinding:
        return self.registry.add_collection(tenant_id, collection_id, target, default=default, **metadata)

    def ask(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None = None,
        principal: Principal | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if kwargs.get("top_k") is not None:
            kwargs["top_k"] = _validate_top_k(kwargs["top_k"])
        binding = self.registry.get_collection(tenant_id, collection_id)
        principal_obj = _tenant_principal(
            principal,
            tenant_id,
            binding.collection_id,
            enforce_access_control=self.enforce_access_control,
        )
        result = self._execute_target(binding, query, principal_obj, **kwargs)
        _attach_tenant_metadata(result, tenant_id, binding.collection_id)
        return result

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None = None,
        principal: Principal | Mapping[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[Document]:
        top_k = _validate_top_k(top_k)
        binding = self.registry.get_collection(tenant_id, collection_id)
        principal_obj = _tenant_principal(
            principal,
            tenant_id,
            binding.collection_id,
            enforce_access_control=self.enforce_access_control,
        )
        target = binding.target
        if hasattr(target, "retriever"):
            target = target.retriever
        if not hasattr(target, "retrieve"):
            response = self.ask(query, tenant_id, collection_id, principal_obj, top_k=top_k)
            return list(getattr(response, "retrieved_documents", []) or [])[:top_k]
        docs = _scope_documents(
            target.retrieve(query, top_k=top_k * 4),
            binding.tenant_id,
            binding.collection_id,
        )
        if self.enforce_access_control:
            docs = self.access_policy.filter_documents(docs, principal_obj)
        return docs[:top_k]

    def stats(self) -> dict[str, Any]:
        return {
            "tenant_count": len(self.registry.tenants),
            "collection_count": len(self.registry.collections),
            "tenants": {
                tenant_id: {
                    "enabled": cfg.enabled,
                    "default_collection_id": cfg.default_collection_id,
                    "collections": [binding.collection_id for binding in self.registry.list_collections(tenant_id)],
                }
                for tenant_id, cfg in self.registry.tenants.items()
            },
        }

    def _execute_target(
        self,
        binding: CollectionBinding,
        query: str,
        principal: Principal,
        **kwargs: Any,
    ) -> Any:
        target = binding.target
        if hasattr(target, "retriever") and hasattr(target, "llm_client") and self.enforce_access_control:
            scoped_target = _ScopedRAGTarget(target, binding.tenant_id, binding.collection_id)
            return AccessControlledRAGEngine(
                scoped_target,
                policy=self.access_policy,
                default_principal=principal,
            ).ask(query, **kwargs)
        if hasattr(target, "ask") and callable(target.ask):
            return _call_with_optional_principal(target.ask, query, principal, kwargs)
        if hasattr(target, "retrieve") and callable(target.retrieve):
            raw_top_k = kwargs.get("top_k", 5)
            top_k = 5 if raw_top_k is None else _validate_top_k(raw_top_k)
            docs = _scope_documents(
                target.retrieve(query, top_k=top_k),
                binding.tenant_id,
                binding.collection_id,
            )
            if self.enforce_access_control:
                docs = self.access_policy.filter_documents(docs, principal)
            return docs
        if callable(target):
            return target(query, tenant_id=principal.tenant_ids, **kwargs)
        raise TypeError(f"Unsupported tenant target type: {type(target).__name__}")


def _call_with_optional_principal(
    ask: Any,
    query: str,
    principal: Principal,
    kwargs: Mapping[str, Any],
) -> Any:
    """Pass a principal only when the target explicitly declares it.

    Catching ``TypeError`` around the call used to retry generators that failed
    internally, potentially executing them twice. It also leaked ``principal``
    into generic ``**generate_kwargs`` accepted by provider-backed engines.
    """

    try:
        parameter = inspect.signature(ask).parameters.get("principal")
    except (TypeError, ValueError):
        parameter = None
    if parameter is None:
        return ask(query, **dict(kwargs))
    if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
        return ask(query, principal, **dict(kwargs))
    return ask(query, principal=principal, **dict(kwargs))


def _tenant_principal(
    principal: Principal | Mapping[str, Any] | None,
    tenant_id: str,
    collection_id: str,
    *,
    enforce_access_control: bool = True,
) -> Principal:
    if isinstance(principal, Principal):
        source = principal
    elif isinstance(principal, Mapping):
        source = Principal.from_dict(principal)
    else:
        source = Principal(user_id="anonymous")

    is_admin = "admin" in source.roles
    if enforce_access_control and not is_admin:
        user_id = str(source.user_id or "").strip()
        if not user_id or user_id.lower() == "anonymous":
            raise PermissionError("An authenticated principal is required for tenant access")
        if tenant_id not in source.tenant_ids:
            raise PermissionError(f"Principal is not authorized for tenant {tenant_id!r}")
        if source.collection_ids and collection_id not in source.collection_ids:
            raise PermissionError(f"Principal is not authorized for collection {collection_id!r}")

    # The scoped principal is a defensive copy. Empty collection_ids means the
    # caller is authorized for every collection in an allowed tenant; the
    # selected collection is added only to this request-local copy.
    return Principal(
        user_id=source.user_id,
        roles=set(source.roles),
        tenant_ids={*source.tenant_ids, tenant_id},
        collection_ids={*source.collection_ids, collection_id},
        attributes=dict(source.attributes),
        max_classification=source.max_classification,
    )


class _ScopedRetriever(BaseRetriever):
    """Attach authoritative registry scope without mutating source documents."""

    def __init__(self, retriever: BaseRetriever, tenant_id: str, collection_id: str):
        self.retriever = retriever
        self.tenant_id = tenant_id
        self.collection_id = collection_id

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        top_k = _validate_top_k(top_k)
        return _scope_documents(
            self.retriever.retrieve(query, top_k=top_k),
            self.tenant_id,
            self.collection_id,
        )


class _ScopedRAGTarget:
    """Proxy a RAG target while scoping documents to its registry binding."""

    def __init__(self, target: Any, tenant_id: str, collection_id: str):
        self._target = target
        self.retriever = _ScopedRetriever(target.retriever, tenant_id, collection_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def _scope_documents(
    documents: Any,
    tenant_id: str,
    collection_id: str,
) -> list[Document]:
    scoped: list[Document] = []
    for doc in documents:
        metadata = dict(doc.metadata or {})
        metadata.setdefault("tenant_id", tenant_id)
        metadata.setdefault("collection_id", collection_id)
        scoped.append(
            Document(
                content=doc.content,
                metadata=metadata,
                doc_id=doc.doc_id,
                score=doc.score,
            )
        )
    return scoped


def _attach_tenant_metadata(result: Any, tenant_id: str, collection_id: str) -> None:
    payload = {"tenant_id": tenant_id, "collection_id": collection_id}
    if hasattr(result, "metadata") and isinstance(result.metadata, dict):
        result.metadata.setdefault("tenant", payload)
    elif isinstance(result, dict):
        result.setdefault("tenant", payload)
