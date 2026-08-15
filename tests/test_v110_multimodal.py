import unittest

import numpy as np

from cheragh.base import StaticLLMClient
from cheragh.multimodal import (
    CallableMultimodalEmbedding,
    Modality,
    MultimodalDocument,
    MultimodalQuery,
    MultimodalRAGEngine,
    MultimodalRetriever,
)


class V110MultimodalTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            MultimodalDocument("Un phare rouge sur la côte.", doc_id="image-1", modality="image", uri="phare.png"),
            MultimodalDocument("Le bilan financier annuel.", doc_id="table-1", modality="table"),
            MultimodalDocument("Guide des ports.", doc_id="text-1", modality="text"),
        ]
        vectors = {
            "image-1": [1.0, 0.0, 0.0],
            "table-1": [0.0, 1.0, 0.0],
            "text-1": [0.0, 0.0, 1.0],
        }
        self.encoder = CallableMultimodalEmbedding(
            lambda docs: np.asarray([vectors[doc.doc_id] for doc in docs]),
            lambda query: np.asarray([1.0, 0.0, 0.0] if "phare" in query.text else [0.0, 1.0, 0.0]),
        )

    def test_cross_modal_text_query_retrieves_image(self):
        retriever = MultimodalRetriever(self.documents, self.encoder)
        result = retriever.retrieve("Montre le phare", top_k=1)
        self.assertEqual(result[0].doc_id, "image-1")
        self.assertEqual(result[0].metadata["modality"], "image")

    def test_modality_filter_and_multimodal_query(self):
        retriever = MultimodalRetriever(self.documents, self.encoder)
        result = retriever.retrieve_multimodal(
            MultimodalQuery(text="phare"), top_k=3, modalities=[Modality.TEXT, Modality.TABLE]
        )
        self.assertNotIn("image-1", [doc.doc_id for doc in result])

    def test_engine_preserves_media_provenance_and_citations(self):
        retriever = MultimodalRetriever(self.documents, self.encoder)
        engine = MultimodalRAGEngine(
            retriever,
            llm_client=StaticLLMClient("Le phare est rouge. [source: image-1]"),
        )
        response = engine.ask("Que montre l'image du phare ?", top_k=1)
        self.assertEqual(response.sources[0].doc_id, "image-1")
        self.assertEqual(response.metadata["architecture"], "multimodal_rag")
        self.assertEqual(response.citations, ["image-1"])

    def test_invalid_embedding_shape_is_rejected(self):
        encoder = CallableMultimodalEmbedding(lambda docs: np.asarray([1.0, 2.0]), lambda query: [1.0, 2.0])
        with self.assertRaises(ValueError):
            MultimodalRetriever(self.documents, encoder)


if __name__ == "__main__":
    unittest.main()
