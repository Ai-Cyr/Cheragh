from __future__ import annotations

import math
import unittest

import numpy as np

from cheragh import BaseRetriever, Document
from cheragh.training import (
    DistilledRetrievalExample,
    HardNegativeMiner,
    RAFTDatasetBuilder,
    RAFTTrainingRecord,
    RetrievalTrainingExample,
    RetrievalTrainingPipeline,
    TeacherScoreDistiller,
    contrastive_retrieval_loss,
)


class _Retriever(BaseRetriever):
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def retrieve(self, query: str, top_k: int = 5):
        self.calls.append((query, top_k))
        return self.documents[:top_k]


class _Trainer:
    def __init__(self):
        self.examples = None
        self.kwargs = None

    def fit(self, examples, **kwargs):
        self.examples = list(examples)
        self.kwargs = kwargs
        return {"examples": len(self.examples)}


class RetrievalTrainingTests(unittest.TestCase):
    def test_example_snapshots_and_rejects_overlap(self):
        positive = Document("positive", doc_id="p", metadata={"nested": {"x": 1}})
        negative = Document("negative", doc_id="n")
        example = RetrievalTrainingExample("query", (positive,), (negative,), answer="answer")
        positive.content = "changed"
        positive.metadata["nested"]["x"] = 2

        self.assertEqual(example.positive_documents[0].content, "positive")
        self.assertEqual(example.positive_documents[0].metadata["nested"]["x"], 1)
        self.assertEqual(example.positive_doc_ids, ("p",))
        serialized = example.to_dict()
        self.assertEqual(serialized["answer"], "answer")
        serialized["metadata"]["nested"] = {"changed": True}
        self.assertNotIn("nested", example.metadata)
        with self.assertRaises(ValueError):
            RetrievalTrainingExample("query", (Document("a", doc_id="same"),), (Document("b", doc_id="same"),))
        with self.assertRaises(TypeError):
            RetrievalTrainingExample("query", (positive,), answer=42)

    def test_hard_negative_mining_excludes_positives_duplicates_and_filter(self):
        positive = Document("oracle", doc_id="p")
        retriever = _Retriever(
            [
                Document("oracle copy", doc_id="p", score=1.0),
                Document("answer-bearing", doc_id="false", score=0.9),
                Document("hard one", doc_id="n1", score=0.8),
                Document("hard duplicate", doc_id="n1", score=0.7),
                Document("hard two", doc_id="n2", score=0.6),
            ]
        )
        miner = HardNegativeMiner(
            retriever,
            candidate_top_k=5,
            negatives_per_query=2,
            exclusion_filter=lambda query, doc: doc.doc_id == "false",
        )

        example = miner.mine("question", [positive])

        self.assertEqual([doc.doc_id for doc in example.negative_documents], ["n1", "n2"])
        self.assertEqual(example.metadata["mined_negatives"], 2)
        self.assertEqual(retriever.calls, [("question", 5)])

    def test_teacher_distillation_is_aligned_and_stable(self):
        example = RetrievalTrainingExample(
            "query",
            (Document("positive", doc_id="p"),),
            (Document("negative", doc_id="n"),),
        )
        distilled = TeacherScoreDistiller(
            lambda query, docs: [1001.0, 1000.0],
            temperature=1.0,
        ).distill(example)

        self.assertAlmostEqual(sum(distilled.document_probabilities), 1.0)
        self.assertGreater(distilled.document_probabilities[0], distilled.document_probabilities[1])
        self.assertEqual(distilled.teacher_scores, (1001.0, 1000.0))
        with self.assertRaises(ValueError):
            TeacherScoreDistiller(lambda query, docs: [1.0]).distill(example)

        def mutating_scorer(query, docs):
            docs[0].content = "mutated by scorer"
            return [1.0, 0.0]

        TeacherScoreDistiller(mutating_scorer).distill(example)
        self.assertEqual(example.positive_documents[0].content, "positive")
        with self.assertRaises(ValueError):
            DistilledRetrievalExample(example, (1.1, -0.1), (1.0, 0.0), 1.0)
        with self.assertRaises(ValueError):
            DistilledRetrievalExample(example, (0.5, 0.5), (1.0, 0.0), 0.0)

    def test_reference_contrastive_loss_and_validation(self):
        loss = contrastive_retrieval_loss(
            np.array([[1.0, 0.0]]),
            np.array([[1.0, 0.0]]),
            np.array([[[0.0, 1.0]]]),
        )
        self.assertAlmostEqual(loss, math.log1p(math.exp(-1.0)))
        with self.assertRaises(ValueError):
            contrastive_retrieval_loss([[1.0, 0.0]], [[1.0, 0.0]], np.zeros((1, 0, 2)))
        with self.assertRaises(ValueError):
            contrastive_retrieval_loss([[1.0, 0.0]], [[1.0, 0.0]], [[[0.0, 1.0]]], temperature=0)
        with self.assertRaises(ValueError):
            contrastive_retrieval_loss(
                np.zeros((0, 2)),
                np.zeros((0, 2)),
                np.zeros((0, 1, 2)),
            )
        with self.assertRaises(ValueError):
            contrastive_retrieval_loss(
                np.zeros((1, 0)),
                np.zeros((1, 0)),
                np.zeros((1, 1, 0)),
            )

    def test_raft_records_include_or_drop_oracles_deterministically(self):
        example = RetrievalTrainingExample(
            "question",
            (Document("oracle", doc_id="p"),),
            (Document("distractor", doc_id="n"),),
            answer="grounded answer [source: p]",
        )

        included = RAFTDatasetBuilder(oracle_probability=1.0).build([example])[0]
        excluded = RAFTDatasetBuilder(oracle_probability=0.0).build([example])[0]

        self.assertTrue(included.oracle_included)
        self.assertEqual([doc.doc_id for doc in included.documents], ["p", "n"])
        self.assertFalse(excluded.oracle_included)
        self.assertEqual([doc.doc_id for doc in excluded.documents], ["n"])
        self.assertIn("[source: p]", included.render_prompt())
        self.assertEqual(included.to_dict()["metadata"]["recipe"], "raft")
        with self.assertRaises(TypeError):
            RAFTTrainingRecord("q", "a", (example.positive_documents[0],), ("p",), 1)
        with self.assertRaises(ValueError):
            RAFTTrainingRecord("q", "a", (), ("p",), True)
        with self.assertRaises(ValueError):
            RAFTTrainingRecord("q", "a", (example.positive_documents[0],), ("p",), False)

    def test_pipeline_mines_distills_and_invokes_adapter(self):
        source = RetrievalTrainingExample(
            "query",
            (Document("positive", doc_id="p"),),
            answer="answer",
            metadata={"split": "train"},
        )
        miner = HardNegativeMiner(
            _Retriever([Document("negative", doc_id="n")]),
            candidate_top_k=2,
            negatives_per_query=1,
        )
        distiller = TeacherScoreDistiller(lambda query, docs: [2.0, 1.0])
        trainer = _Trainer()

        result = RetrievalTrainingPipeline(miner, distiller).fit([source], trainer, epochs=2)

        self.assertEqual(result, {"examples": 1})
        self.assertEqual(trainer.kwargs, {"epochs": 2})
        self.assertEqual(trainer.examples[0].example.metadata["split"], "train")
        self.assertEqual(trainer.examples[0].example.metadata["mined_negatives"], 1)
        with self.assertRaises(ValueError):
            RetrievalTrainingPipeline().fit([], trainer)
        with self.assertRaises(TypeError):
            RetrievalTrainingPipeline().fit([source], type("BadTrainer", (), {"fit": 3})())


if __name__ == "__main__":
    unittest.main()
