from __future__ import annotations

import unittest

import numpy as np

from cheragh.multimodal import Modality, MultimodalDocument
from cheragh.multimodal.colpali import CallableVisualLateInteractionEncoder, ColPaliRetriever


def _page(doc_id: str, vectors, **metadata) -> MultimodalDocument:
    return MultimodalDocument(
        content=f"page {doc_id}",
        doc_id=doc_id,
        metadata={"vectors": vectors, **metadata},
        modality=Modality.IMAGE,
        uri=f"{doc_id}.png",
        mime_type="image/png",
    )


class ColPaliRetrieverTests(unittest.TestCase):
    def _encoder(self, query_vectors) -> CallableVisualLateInteractionEncoder:
        return CallableVisualLateInteractionEncoder(
            page_encoder=lambda pages: [page.metadata["vectors"] for page in pages],
            query_encoder=lambda queries: [query_vectors for _ in queries],
            fingerprint="test-colpali",
        )

    def test_exact_maxsim_returns_patch_provenance(self):
        query = np.array([[1.0, 0.0], [0.0, 1.0]])
        retriever = ColPaliRetriever(
            [
                _page("best", [[1.0, 0.0], [0.0, 1.0]]),
                _page("weak", [[0.5, 0.5]]),
            ],
            self._encoder(query),
            normalize_vectors=False,
        )

        results = retriever.retrieve_pages("find the page", top_k=2)

        self.assertEqual([doc.doc_id for doc in results], ["best", "weak"])
        self.assertEqual(results[0].score, 2.0)
        self.assertEqual(results[1].score, 1.0)
        self.assertEqual(results[0].metadata["maxsim_patch_indices"], [0, 1])
        self.assertEqual(results[0].metadata["maxsim_token_scores"], [1.0, 1.0])
        self.assertEqual(results[0].metadata["retrieval_method"], "colpali-maxsim")
        self.assertEqual(results[0].metadata["visual_encoder"], "test-colpali")

    def test_filters_snapshots_and_base_retriever_contract(self):
        page = _page("legal", [[1.0, 0.0]], tenant="a")
        retriever = ColPaliRetriever(
            [page, _page("finance", [[0.8, 0.0]], tenant="b")],
            self._encoder([[1.0, 0.0]]),
            normalize_vectors=False,
        )
        page.content = "mutated"
        page.metadata["tenant"] = "b"

        filtered = retriever.retrieve_pages("policy", top_k=5, filters={"tenant": "a"})
        as_documents = retriever.retrieve("policy", top_k=1)

        self.assertEqual([doc.doc_id for doc in filtered], ["legal"])
        self.assertEqual(filtered[0].content, "page legal")
        self.assertEqual(filtered[0].metadata["tenant"], "a")
        self.assertIsInstance(as_documents[0], MultimodalDocument)

    def test_normalization_and_query_length_normalization(self):
        page = _page("page", [[2.0, 0.0], [0.0, 3.0]])
        query = [[4.0, 0.0], [0.0, 5.0]]
        summed = ColPaliRetriever([page], self._encoder(query))
        averaged = ColPaliRetriever(
            [page],
            self._encoder(query),
            normalize_by_query_tokens=True,
        )

        self.assertAlmostEqual(summed.retrieve("x", top_k=1)[0].score, 2.0)
        self.assertAlmostEqual(averaged.retrieve("x", top_k=1)[0].score, 1.0)

    def test_validates_top_k_query_modality_and_embedding_shapes(self):
        encoder = self._encoder([[1.0, 0.0]])
        retriever = ColPaliRetriever([_page("page", [[1.0, 0.0]])], encoder)

        for invalid in (0, -1):
            with self.subTest(top_k=invalid), self.assertRaises(ValueError):
                retriever.retrieve("query", top_k=invalid)
        for invalid in (True, 1.5):
            with self.subTest(top_k=invalid), self.assertRaises(TypeError):
                retriever.retrieve("query", top_k=invalid)
        with self.assertRaises(ValueError):
            retriever.retrieve("   ", top_k=1)
        with self.assertRaises(ValueError):
            ColPaliRetriever(
                [
                    MultimodalDocument(
                        "text",
                        doc_id="text",
                        modality=Modality.TEXT,
                    )
                ],
                encoder,
            )

        bad_count = CallableVisualLateInteractionEncoder(
            page_encoder=lambda pages: [],
            query_encoder=lambda queries: [[[1.0, 0.0]]],
        )
        with self.assertRaises(ValueError):
            ColPaliRetriever([_page("page", [[1.0, 0.0]])], bad_count)

        bad_query = CallableVisualLateInteractionEncoder(
            page_encoder=lambda pages: [[[1.0, 0.0]]],
            query_encoder=lambda queries: [[[1.0, 0.0, 0.0]]],
        )
        with self.assertRaises(ValueError):
            ColPaliRetriever([_page("page", [[1.0, 0.0]])], bad_query).retrieve("query", top_k=1)

    def test_non_finite_embeddings_are_rejected(self):
        with self.assertRaises(ValueError):
            ColPaliRetriever(
                [_page("page", [[float("nan"), 0.0]])],
                self._encoder([[1.0, 0.0]]),
            )

    def test_failed_add_pages_is_transactional(self):
        encoder = CallableVisualLateInteractionEncoder(
            page_encoder=lambda pages: [page.metadata["vectors"] for page in pages],
            query_encoder=lambda queries: [[[1.0, 0.0, 0.0]]],
        )
        retriever = ColPaliRetriever([], encoder)

        with self.assertRaises(ValueError):
            retriever.add_pages(
                [
                    _page("two", [[1.0, 0.0]]),
                    _page("three", [[1.0, 0.0, 0.0]]),
                ]
            )

        self.assertIsNone(retriever.dimension)
        self.assertEqual(retriever.pages, [])
        retriever.add_pages([_page("valid", [[1.0, 0.0, 0.0]])])
        self.assertEqual(retriever.dimension, 3)
        self.assertEqual(retriever.retrieve("query", top_k=1)[0].doc_id, "valid")

    def test_encoder_cannot_mutate_indexed_page_snapshots(self):
        def page_encoder(pages):
            vectors = [page.metadata["vectors"] for page in pages]
            pages[0].content = "encoder mutation"
            pages[0].metadata["nested"]["owner"] = "encoder"
            return vectors

        encoder = CallableVisualLateInteractionEncoder(
            page_encoder=page_encoder,
            query_encoder=lambda queries: [[[1.0, 0.0]]],
        )
        page = _page("page", [[1.0, 0.0]], nested={"owner": "source"})
        retriever = ColPaliRetriever([page], encoder)

        result = retriever.retrieve("query", top_k=1)[0]

        self.assertEqual(result.content, "page page")
        self.assertEqual(result.metadata["nested"]["owner"], "source")

    def test_index_owns_a_copy_of_provider_embedding_arrays(self):
        provider_array = np.array([[1.0, 0.0]], dtype=float)
        encoder = CallableVisualLateInteractionEncoder(
            page_encoder=lambda pages: [provider_array],
            query_encoder=lambda queries: [[[1.0, 0.0]]],
        )
        retriever = ColPaliRetriever(
            [_page("page", [[0.0, 0.0]])],
            encoder,
            normalize_vectors=False,
        )

        provider_array[:] = 0.0

        self.assertEqual(retriever.retrieve("query", top_k=1)[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()
