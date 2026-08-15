import unittest

import numpy as np

from cheragh.base import Document
from cheragh.retrieval import ColBERTRetriever, LearnedSparseRetriever, SPLADERetriever


class FakeSparseEncoder:
    def __init__(self):
        self.document_batch_sizes = []

    def encode_documents(self, texts):
        self.document_batch_sizes.append(len(texts))
        values = {
            "a cat naps": {1: 1.0, 4: 0.2},
            "a dog runs": {2: 1.0},
            "another cat": {1: 1.0},
        }
        return [values[text] for text in texts]

    def encode_queries(self, texts):
        values = {"feline": {1: 2.0}, "canine": {2: 1.5}}
        return [values[text] for text in texts]


class FakeTokenEncoder:
    def __init__(self):
        self.document_batch_sizes = []
        self.vectors = {
            "perfect": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "partial": np.array([[0.8, 0.6], [0.8, 0.6]]),
            "opposite": np.array([[-1.0, 0.0], [0.0, -1.0]]),
            "query": np.array([[1.0, 0.0], [0.0, 1.0]]),
        }

    def encode_documents(self, texts):
        self.document_batch_sizes.append(len(texts))
        return [self.vectors[text] for text in texts]

    def encode_queries(self, texts):
        return [self.vectors[text] for text in texts]


class LearnedSparseRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document("a cat naps", metadata={"kind": "cat"}, doc_id="cat", score=99.0),
            Document("a dog runs", metadata={"kind": "dog"}, doc_id="dog"),
            Document("another cat", metadata={"kind": "cat-2"}, doc_id="cat-2"),
        ]

    def test_splade_ranks_by_sparse_dot_product_and_batches(self):
        encoder = FakeSparseEncoder()
        retriever = SPLADERetriever(self.documents, encoder=encoder, batch_size=2)

        results = retriever.retrieve("feline", top_k=3)

        self.assertIs(SPLADERetriever, LearnedSparseRetriever)
        self.assertEqual(encoder.document_batch_sizes, [2, 1])
        self.assertEqual([result.doc_id for result in results], ["cat", "cat-2", "dog"])
        self.assertEqual([result.score for result in results], [2.0, 2.0, 0.0])
        self.assertEqual(results[0].metadata, {"kind": "cat", "retrieval_method": "learned_sparse"})
        self.assertEqual(self.documents[0].metadata, {"kind": "cat"})
        self.assertEqual(self.documents[0].score, 99.0)

    def test_sparse_retriever_accepts_dense_array_encoder_and_add_documents(self):
        class DenseEncoder:
            def encode(self, texts):
                values = {"first": [1.0, 0.0], "second": [0.0, 1.0], "find second": [0.0, 2.0]}
                return np.asarray([values[text] for text in texts])

        retriever = LearnedSparseRetriever([Document("first", doc_id="first")], encoder=DenseEncoder())
        retriever.add_documents([Document("second", doc_id="second")])
        self.assertEqual(retriever.retrieve("find second", top_k=1)[0].doc_id, "second")

    def test_sparse_validation_rejects_invalid_inputs_and_outputs(self):
        with self.assertRaises(ValueError):
            LearnedSparseRetriever([], encoder=FakeSparseEncoder(), batch_size=0)

        retriever = LearnedSparseRetriever(self.documents, encoder=FakeSparseEncoder())
        with self.assertRaises(ValueError):
            retriever.retrieve(" ")
        with self.assertRaises(ValueError):
            retriever.retrieve("feline", top_k=0)

        class WrongCountEncoder:
            def encode(self, texts):
                return [{0: 1.0}]

        with self.assertRaises(ValueError):
            LearnedSparseRetriever(self.documents, encoder=WrongCountEncoder())


class ColBERTRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document("perfect", metadata={"rank": 1}, doc_id="perfect"),
            Document("partial", metadata={"rank": 2}, doc_id="partial"),
            Document("opposite", metadata={"rank": 3}, doc_id="opposite"),
        ]

    def test_colbert_computes_exact_maxsim_and_batches(self):
        encoder = FakeTokenEncoder()
        retriever = ColBERTRetriever(self.documents, token_encoder=encoder, batch_size=2)

        results = retriever.retrieve("query", top_k=3)

        self.assertEqual(encoder.document_batch_sizes, [2, 1])
        self.assertEqual([result.doc_id for result in results], ["perfect", "partial", "opposite"])
        self.assertAlmostEqual(results[0].score, 2.0)
        self.assertAlmostEqual(results[1].score, 1.4)
        # Each query token can select the orthogonal document token, so the
        # MaxSim score is zero rather than the diagonal score of -2.
        self.assertAlmostEqual(results[2].score, 0.0)
        self.assertEqual(results[0].metadata, {"rank": 1, "retrieval_method": "colbert_maxsim"})
        self.assertEqual(self.documents[0].metadata, {"rank": 1})

    def test_colbert_supports_batched_tensor_and_attention_mask(self):
        class MaskedEncoder:
            def encode(self, texts):
                embeddings = np.asarray([[[1.0, 0.0], [50.0, 50.0]] for _ in texts])
                masks = np.asarray([[1, 0] for _ in texts])
                return embeddings, masks

        retriever = ColBERTRetriever(
            [Document("doc", doc_id="doc")],
            token_encoder=MaskedEncoder(),
            normalize=False,
        )
        result = retriever.retrieve("query", top_k=1)[0]
        self.assertEqual(result.score, 1.0)

    def test_colbert_accepts_tuple_of_variable_length_token_matrices(self):
        class TupleEncoder:
            def encode(self, texts):
                return tuple(np.ones((index + 1, 2)) for index, _ in enumerate(texts))

        retriever = ColBERTRetriever(
            [Document("first"), Document("second")],
            token_encoder=TupleEncoder(),
            batch_size=2,
        )
        self.assertEqual(len(retriever.retrieve("query", top_k=2)), 2)

    def test_colbert_validation_rejects_dimension_changes(self):
        class InconsistentEncoder:
            def encode_documents(self, texts):
                return [np.ones((2, 2)) for _ in texts]

            def encode_queries(self, texts):
                return [np.ones((2, 3)) for _ in texts]

        retriever = ColBERTRetriever([Document("doc")], token_encoder=InconsistentEncoder())
        with self.assertRaises(ValueError):
            retriever.retrieve("query")


if __name__ == "__main__":
    unittest.main()
