import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cheragh import Document, HashingEmbedding
from cheragh.vectorstores.chroma import ChromaVectorStore
from cheragh.vectorstores.qdrant import QdrantVectorStore


class FakeChromaCollection:
    def __init__(self):
        self.upsert_calls = []

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)


class FakeChromaClient:
    def __init__(self):
        self.collection = FakeChromaCollection()

    def get_or_create_collection(self, name):
        del name
        return self.collection


class FakePointStruct:
    def __init__(self, *, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload


class FakeVectorParams:
    def __init__(self, *, size, distance):
        self.size = size
        self.distance = distance


class FakeDistance:
    COSINE = "cosine"


class FakeQdrantModels:
    PointStruct = FakePointStruct
    VectorParams = FakeVectorParams
    Distance = FakeDistance


class FakeQdrantClient:
    def __init__(self):
        self.collections = set()
        self.upsert_calls = []

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in sorted(self.collections)]
        )

    def create_collection(self, *, collection_name, vectors_config):
        del vectors_config
        self.collections.add(collection_name)

    def upsert(self, *, collection_name, points):
        self.upsert_calls.append((collection_name, list(points)))


class VectorStoreAdapterIdTests(unittest.TestCase):
    def setUp(self):
        self.embedding = HashingEmbedding(dimension=8)

    def test_chroma_anonymous_ids_do_not_collide_across_batches(self):
        client = FakeChromaClient()
        store = ChromaVectorStore(self.embedding, client=client)

        store.add_documents([Document("premier")])
        store.add_documents([Document("second")])

        first_id = client.collection.upsert_calls[0]["ids"][0]
        second_id = client.collection.upsert_calls[1]["ids"][0]
        self.assertNotEqual(first_id, second_id)
        self.assertTrue(first_id.startswith("auto-"))
        self.assertTrue(second_id.startswith("auto-"))

    def test_chroma_preserves_explicit_id_for_upsert(self):
        client = FakeChromaClient()
        store = ChromaVectorStore(self.embedding, client=client)

        store.add_documents([Document("version un", doc_id="stable")])
        store.add_documents([Document("version deux", doc_id="stable")])

        self.assertEqual(client.collection.upsert_calls[0]["ids"], ["stable"])
        self.assertEqual(client.collection.upsert_calls[1]["ids"], ["stable"])

    def test_qdrant_anonymous_ids_do_not_collide_across_batches(self):
        client = FakeQdrantClient()
        store = QdrantVectorStore(self.embedding, client=client)

        with patch(
            "cheragh.vectorstores.qdrant.require_qdrant_client",
            return_value=(object, FakeQdrantModels),
        ):
            store.add_documents([Document("premier")])
            store.add_documents([Document("second")])

        first_point = client.upsert_calls[0][1][0]
        second_point = client.upsert_calls[1][1][0]
        self.assertNotEqual(first_point.id, second_point.id)
        self.assertEqual(first_point.payload["doc_id"], first_point.id)
        self.assertEqual(second_point.payload["doc_id"], second_point.id)

    def test_qdrant_metadata_cannot_override_reserved_payload_fields(self):
        client = FakeQdrantClient()
        store = QdrantVectorStore(self.embedding, client=client)
        document = Document(
            "contenu canonique",
            doc_id="document-canonique",
            metadata={
                "content": "contenu usurpé",
                "doc_id": "identifiant usurpé",
                "tenant_id": "tenant-a",
            },
        )

        with patch(
            "cheragh.vectorstores.qdrant.require_qdrant_client",
            return_value=(object, FakeQdrantModels),
        ):
            store.add_documents([document])

        payload = client.upsert_calls[0][1][0].payload
        self.assertEqual(payload["content"], "contenu canonique")
        self.assertEqual(payload["doc_id"], "document-canonique")
        self.assertEqual(payload["tenant_id"], "tenant-a")


if __name__ == "__main__":
    unittest.main()
