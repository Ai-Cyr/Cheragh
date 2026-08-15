import unittest
from types import SimpleNamespace

from cheragh import Document, HashingEmbedding
from cheragh.vectorstores.chroma import ChromaVectorStore
from cheragh.vectorstores.qdrant import QdrantVectorStore


RANKED_RECORDS = [
    ("first", "Premier", {"tenant": "other", "quality": 0.99, "tags": ["legal"]}, 0.01),
    ("second", "Deuxième", {"tenant": "acme", "quality": 0.50, "tags": ["legal"]}, 0.02),
    ("third", "Troisième", {"tenant": "other", "quality": 0.70, "tags": ["demo"]}, 0.03),
    ("fourth", "Quatrième", {"tenant": "acme", "quality": 0.80, "tags": ["demo"]}, 0.04),
    ("wanted", "Résultat", {"tenant": "acme", "quality": 0.95, "tags": ["legal"]}, 0.05),
    ("last", "Dernier", {"tenant": "acme", "quality": 0.91, "tags": ["legal"], "archived": True}, 0.06),
]


class FakeChromaCollection:
    def __init__(self):
        self.query_calls = []
        self.count_calls = 0

    def count(self):
        self.count_calls += 1
        return len(RANKED_RECORDS)

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        records = RANKED_RECORDS[: kwargs["n_results"]]
        return {
            "ids": [[record[0] for record in records]],
            "documents": [[record[1] for record in records]],
            "metadatas": [[record[2] for record in records]],
            "distances": [[record[3] for record in records]],
        }


class FakeChromaClient:
    def __init__(self):
        self.collection = FakeChromaCollection()

    def get_or_create_collection(self, name):
        del name
        return self.collection


class RoundTripChromaCollection:
    def __init__(self):
        self.records = []

    def upsert(self, **kwargs):
        self.records = list(
            zip(kwargs["ids"], kwargs["documents"], kwargs["metadatas"])
        )

    def count(self):
        return len(self.records)

    def query(self, **kwargs):
        records = self.records[: kwargs["n_results"]]
        return {
            "ids": [[record[0] for record in records]],
            "documents": [[record[1] for record in records]],
            "metadatas": [[record[2] for record in records]],
            "distances": [[float(index) for index, _ in enumerate(records)]],
        }


class RoundTripChromaClient:
    def __init__(self):
        self.collection = RoundTripChromaCollection()

    def get_or_create_collection(self, name):
        del name
        return self.collection


class FakeQdrantClient:
    def __init__(self):
        self.search_calls = []
        self.count_calls = []

    def count(self, **kwargs):
        self.count_calls.append(kwargs)
        return SimpleNamespace(count=len(RANKED_RECORDS))

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        hits = []
        for doc_id, content, metadata, distance in RANKED_RECORDS[: kwargs["limit"]]:
            hits.append(
                SimpleNamespace(
                    id=doc_id,
                    payload={**metadata, "content": content, "doc_id": doc_id},
                    score=1.0 - distance,
                )
            )
        return hits


class FakeMatchValue:
    def __init__(self, *, value):
        self.value = value


class FakeMatchAny:
    def __init__(self, *, any):
        self.any = any


class FakeRange:
    def __init__(self, **kwargs):
        self.values = kwargs


class FakeFieldCondition:
    def __init__(self, *, key, match=None, range=None):
        self.key = key
        self.match = match
        self.range = range


class FakeFilter:
    def __init__(self, *, must=None, must_not=None):
        self.must = must or []
        self.must_not = must_not or []


class FakeQdrantFilterModels:
    MatchValue = FakeMatchValue
    MatchAny = FakeMatchAny
    Range = FakeRange
    FieldCondition = FakeFieldCondition
    Filter = FakeFilter


class BackendFilterContractTests(unittest.TestCase):
    def setUp(self):
        self.embedding = HashingEmbedding(dimension=8)

    @staticmethod
    def filters():
        return {
            "tenant": {"$in": ["acme"], "$ne": "blocked"},
            "quality": {"$gte": 0.9, "$lt": 1.0},
            "tags": {"$contains": "legal"},
            "archived": {"$exists": False},
        }

    def test_chroma_uses_canonical_filter_contract_and_preserves_top_k(self):
        client = FakeChromaClient()
        store = ChromaVectorStore(self.embedding, client=client)

        results = store.similarity_search("contrat", top_k=1, filters=self.filters())

        self.assertEqual([document.doc_id for document in results], ["wanted"])
        self.assertEqual([call["n_results"] for call in client.collection.query_calls], [4, 6])
        native_filter = {
            "$and": [
                {"tenant": {"$in": ["acme"]}},
                {"quality": {"$gte": 0.9}},
                {"quality": {"$lt": 1.0}},
            ]
        }
        self.assertTrue(all(call["where"] == native_filter for call in client.collection.query_calls))
        self.assertNotIn("tags", str(native_filter))
        self.assertNotIn("archived", str(native_filter))
        self.assertEqual(client.collection.count_calls, 1)

    def test_qdrant_uses_canonical_filter_contract_and_preserves_top_k(self):
        client = FakeQdrantClient()
        store = QdrantVectorStore(self.embedding, client=client)
        store._models = FakeQdrantFilterModels

        results = store.similarity_search("contrat", top_k=1, filters=self.filters())

        self.assertEqual([document.doc_id for document in results], ["wanted"])
        self.assertEqual([call["limit"] for call in client.search_calls], [4, 6])
        native_filter = client.search_calls[0]["query_filter"]
        self.assertTrue(all(call["query_filter"] is native_filter for call in client.search_calls))
        self.assertEqual([condition.key for condition in native_filter.must], ["tenant", "quality"])
        self.assertEqual(native_filter.must[0].match.any, ["acme"])
        self.assertEqual(native_filter.must[1].range.values, {"gte": 0.9, "lt": 1.0})
        self.assertEqual(native_filter.must_not, [])
        self.assertNotIn("tags", [condition.key for condition in native_filter.must])
        self.assertNotIn("archived", [condition.key for condition in native_filter.must])
        self.assertEqual(len(client.count_calls), 1)
        self.assertIs(client.count_calls[0]["count_filter"], native_filter)
        self.assertEqual(client.count_calls[0]["collection_name"], "cheragh")
        self.assertTrue(client.count_calls[0]["exact"])

    def test_qdrant_does_not_push_negative_predicates_on_array_metadata(self):
        client = FakeQdrantClient()
        store = QdrantVectorStore(self.embedding, client=client)
        store._models = FakeQdrantFilterModels

        results = store.similarity_search(
            "contrat",
            top_k=1,
            filters={"tags": {"$ne": "legal"}},
        )

        # The canonical predicate compares the whole list with the scalar, so
        # ["legal"] != "legal". A Qdrant must_not MatchValue would incorrectly
        # discard this valid first result before local filtering.
        self.assertEqual([document.doc_id for document in results], ["first"])
        self.assertTrue(all(call["query_filter"] is None for call in client.search_calls))

    def test_qdrant_without_models_falls_back_to_exact_local_filtering(self):
        client = FakeQdrantClient()
        store = QdrantVectorStore(self.embedding, client=client)

        results = store.similarity_search("contrat", top_k=1, filters=self.filters())

        self.assertEqual([document.doc_id for document in results], ["wanted"])
        self.assertTrue(all(call["query_filter"] is None for call in client.search_calls))
        self.assertEqual(client.count_calls, [{"collection_name": "cheragh", "exact": True}])

    def test_unfiltered_search_does_not_count_or_overfetch(self):
        chroma_client = FakeChromaClient()
        qdrant_client = FakeQdrantClient()
        chroma = ChromaVectorStore(self.embedding, client=chroma_client)
        qdrant = QdrantVectorStore(self.embedding, client=qdrant_client)

        self.assertEqual(len(chroma.similarity_search("contrat", top_k=2)), 2)
        self.assertEqual(len(qdrant.similarity_search("contrat", top_k=2)), 2)
        self.assertEqual(chroma_client.collection.count_calls, 0)
        self.assertEqual(qdrant_client.count_calls, [])
        self.assertEqual(chroma_client.collection.query_calls[0]["n_results"], 2)
        self.assertEqual(qdrant_client.search_calls[0]["limit"], 2)

    def test_chroma_restores_rich_metadata_before_filtering(self):
        store = ChromaVectorStore(self.embedding, client=RoundTripChromaClient())
        marker_string = "\x1echeragh-json-v1:not-json"
        store.add_documents(
            [
                Document(
                    "Faux positif potentiel",
                    doc_id="illegal",
                    metadata={"tags": ["illegal"], "optional": None, "label": marker_string},
                ),
                Document(
                    "Document juridique",
                    doc_id="legal",
                    metadata={"tags": ["legal"], "optional": None, "label": marker_string},
                ),
            ]
        )

        results = store.similarity_search(
            "contrat",
            top_k=2,
            filters={"tags": {"$contains": "legal"}, "optional": {"$exists": True}},
        )

        self.assertEqual([document.doc_id for document in results], ["legal"])
        self.assertEqual(results[0].metadata["tags"], ["legal"])
        self.assertIsNone(results[0].metadata["optional"])
        self.assertEqual(results[0].metadata["label"], marker_string)

    def test_backends_reject_non_positive_and_boolean_top_k(self):
        stores = [
            ChromaVectorStore(self.embedding, client=FakeChromaClient()),
            QdrantVectorStore(self.embedding, client=FakeQdrantClient()),
        ]
        for store in stores:
            with self.subTest(store=type(store).__name__, top_k=0):
                with self.assertRaises(ValueError):
                    store.similarity_search("contrat", top_k=0)
            with self.subTest(store=type(store).__name__, top_k=True):
                with self.assertRaises(TypeError):
                    store.similarity_search("contrat", top_k=True)


if __name__ == "__main__":
    unittest.main()
