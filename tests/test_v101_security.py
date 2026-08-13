import pickle
import tempfile
import unittest
from pathlib import Path

from cheragh import AccessPolicy, Document, HashingEmbedding, HybridSearchRetriever, MultiTenantRAGEngine, Principal
from cheragh.base import BaseRetriever
from cheragh.cache import load_cache


def _write_marker(path: str) -> dict[str, bool]:
    Path(path).write_text("unpickled", encoding="utf-8")
    return {"loaded": True}


class _PicklePayload:
    def __init__(self, marker_path: str):
        self.marker_path = marker_path

    def __reduce__(self):
        return _write_marker, (self.marker_path,)


class _StaticRetriever(BaseRetriever):
    def __init__(self, documents: list[Document]):
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.documents[:top_k]


class _CountingEmbedding(HashingEmbedding):
    def __init__(self):
        super().__init__(dimension=32)
        self.document_calls = 0

    def embed_documents(self, texts: list[str]):
        self.document_calls += 1
        return super().embed_documents(texts)


class V101SecurityTests(unittest.TestCase):
    def _tenant_engine(self) -> MultiTenantRAGEngine:
        engine = MultiTenantRAGEngine()
        engine.add_collection(
            "tenant-a",
            "contracts",
            _StaticRetriever(
                [
                    Document(
                        "Tenant A contract",
                        metadata={"tenant_id": "tenant-a", "classification": "internal"},
                        doc_id="a1",
                    )
                ]
            ),
            default=True,
        )
        return engine

    def test_tenant_route_rejects_cross_tenant_anonymous_and_wrong_collection(self):
        engine = self._tenant_engine()

        with self.assertRaises(PermissionError):
            engine.retrieve(
                "contract",
                tenant_id="tenant-a",
                principal=Principal(user_id="user-b", tenant_ids={"tenant-b"}),
            )
        with self.assertRaises(PermissionError):
            engine.retrieve("contract", tenant_id="tenant-a")
        with self.assertRaises(PermissionError):
            engine.retrieve(
                "contract",
                tenant_id="tenant-a",
                principal=Principal(
                    user_id="user-a",
                    tenant_ids={"tenant-a"},
                    collection_ids={"invoices"},
                ),
            )

    def test_tenant_route_does_not_mutate_principal_and_allows_explicit_admin(self):
        engine = self._tenant_engine()
        principal = Principal(user_id="user-a", tenant_ids={"tenant-a"})

        documents = engine.retrieve("contract", tenant_id="tenant-a", principal=principal)

        self.assertEqual([document.doc_id for document in documents], ["a1"])
        self.assertEqual(principal.tenant_ids, {"tenant-a"})
        self.assertEqual(principal.collection_ids, set())
        admin_documents = engine.retrieve(
            "contract",
            tenant_id="tenant-a",
            principal=Principal(user_id="root", roles={"admin"}),
        )
        self.assertEqual([document.doc_id for document in admin_documents], ["a1"])

    def test_access_policy_fails_closed_for_required_metadata_and_unknown_labels(self):
        principal = Principal(
            user_id="user-a",
            tenant_ids={"tenant-a"},
            collection_ids={"contracts"},
        )
        strict = AccessPolicy(strict=True)

        missing_tenant = strict.authorize(Document("missing", metadata={"classification": "internal"}), principal)
        self.assertFalse(missing_tenant.allowed)
        self.assertEqual(missing_tenant.reason, "tenant_metadata_missing")

        missing_collection = strict.authorize(
            Document("missing", metadata={"tenant_id": "tenant-a", "classification": "internal"}),
            principal,
        )
        self.assertFalse(missing_collection.allowed)
        self.assertEqual(missing_collection.reason, "collection_metadata_missing")

        unknown = strict.authorize(
            Document(
                "unknown",
                metadata={
                    "tenant_id": "tenant-a",
                    "collection_id": "contracts",
                    "classification": "mystery",
                },
            ),
            principal,
        )
        self.assertFalse(unknown.allowed)
        self.assertEqual(unknown.reason, "unknown_classification")

        tenant_required = AccessPolicy(require_tenant_match=True)
        self.assertFalse(tenant_required.authorize(Document("missing tenant"), principal).allowed)
        self.assertFalse(AccessPolicy().authorize(Document("unknown", metadata={"classification": "mystery"}), principal).allowed)

    def test_legacy_pickle_load_is_disabled_by_default_and_explicitly_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "legacy.pkl"
            marker_path = Path(tmp) / "marker"
            payload = {
                "schema_version": 1,
                "retriever_class": "ExampleRetriever",
                "content_hash": "content",
                "embedder_fingerprint": "embedder",
                "extra_fingerprint": "",
                "state": _PicklePayload(str(marker_path)),
            }
            with cache_path.open("wb") as cache_file:
                pickle.dump(payload, cache_file)

            blocked = load_cache(
                str(cache_path),
                expected_class="ExampleRetriever",
                expected_content_hash="content",
                expected_embedder_fp="embedder",
            )
            self.assertIsNone(blocked)
            self.assertFalse(marker_path.exists())

            loaded = load_cache(
                str(cache_path),
                expected_class="ExampleRetriever",
                expected_content_hash="content",
                expected_embedder_fp="embedder",
                allow_unsafe_pickle=True,
            )
            self.assertEqual(loaded, {"loaded": True})
            self.assertTrue(marker_path.exists())

    def test_hybrid_cache_path_alone_never_unpickles_and_opt_in_reuses_cache(self):
        documents = [Document("safe document", doc_id="safe")]
        with tempfile.TemporaryDirectory() as tmp:
            malicious_path = Path(tmp) / "malicious.pkl"
            marker_path = Path(tmp) / "marker"
            with malicious_path.open("wb") as cache_file:
                pickle.dump(_PicklePayload(str(marker_path)), cache_file)

            HybridSearchRetriever(documents, _CountingEmbedding(), cache_path=str(malicious_path))
            self.assertFalse(marker_path.exists())

            trusted_path = Path(tmp) / "trusted.pkl"
            first_embedder = _CountingEmbedding()
            HybridSearchRetriever(documents, first_embedder, cache_path=str(trusted_path))
            self.assertEqual(first_embedder.document_calls, 1)

            second_embedder = _CountingEmbedding()
            HybridSearchRetriever(
                documents,
                second_embedder,
                cache_path=str(trusted_path),
                allow_unsafe_pickle=True,
            )
            self.assertEqual(second_embedder.document_calls, 0)


if __name__ == "__main__":
    unittest.main()
