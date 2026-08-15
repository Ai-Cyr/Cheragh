import unittest

import numpy as np

from cheragh import StaticLLMClient
from cheragh.multimodal import (
    CallableMultimodalEmbedding,
    MultimodalDocument,
    MultimodalRAGEngine,
    MultimodalRetriever,
)


class MultimodalAPIConsistencyTests(unittest.TestCase):
    def _embedding(self):
        vectors = {"first": [1.0, 0.0], "second": [0.0, 1.0]}
        return CallableMultimodalEmbedding(
            lambda documents: np.asarray([vectors[document.doc_id] for document in documents]),
            lambda query: np.asarray([1.0, 0.0]),
        )

    def test_index_takes_a_snapshot_of_caller_owned_media_documents(self):
        metadata = {"nested": {"value": "original"}}
        source = MultimodalDocument("original content", metadata=metadata, doc_id="first")
        retriever = MultimodalRetriever([source], self._embedding())

        source.content = "mutated content"
        metadata["nested"]["value"] = "mutated"

        result = retriever.retrieve("find", top_k=1)[0]
        self.assertEqual(result.content, "original content")
        self.assertEqual(result.metadata["nested"]["value"], "original")

    def test_engine_forwards_filters_and_finishes_trace(self):
        documents = [
            MultimodalDocument("blocked", metadata={"access": "blocked"}, doc_id="first"),
            MultimodalDocument("allowed", metadata={"access": "allowed"}, doc_id="second"),
        ]
        engine = MultimodalRAGEngine(
            MultimodalRetriever(documents, self._embedding()),
            StaticLLMClient("answer [source: second]"),
        )

        response = engine.ask("find", filters={"access": "allowed"})

        self.assertEqual([document.doc_id for document in response.retrieved_documents], ["second"])
        self.assertEqual(response.metadata["top_k"], 5)
        self.assertEqual(response.trace.query, "find")
        self.assertIsNotNone(response.trace.ended_at_unix)

    def test_top_k_contract_rejects_non_positive_and_boolean_values(self):
        retriever = MultimodalRetriever(
            [MultimodalDocument("content", doc_id="first")],
            self._embedding(),
        )
        engine = MultimodalRAGEngine(retriever)

        for invalid in (0, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    retriever.retrieve("find", top_k=invalid)
                with self.assertRaises(ValueError):
                    engine.ask("find", top_k=invalid)
        with self.assertRaises(TypeError):
            engine.ask("find", top_k=True)


if __name__ == "__main__":
    unittest.main()
