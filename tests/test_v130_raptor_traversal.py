import unittest

import numpy as np

from cheragh import Document, EmbeddingModel, StaticLLMClient
from cheragh.raptor_engine import RAPTOREngine, RAPTORIndex, RAPTORNode, RAPTORRetrieverV2


class _KeywordEmbedding(EmbeddingModel):
    """Tiny normalized encoder with completely predictable similarities."""

    vocabulary = ("apple", "pear", "quantum", "photon")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed_query(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        tokens = set(text.lower().split())
        vector = np.asarray(
            [1.0 if token in tokens else 0.0 for token in self.vocabulary],
            dtype=np.float32,
        )
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


def _tree_index() -> RAPTORIndex:
    leaves = [
        RAPTORNode(
            Document("apple", metadata={"nested": {"owner": "original"}}, doc_id="fruit-a"),
            level=0,
        ),
        RAPTORNode(Document("pear", doc_id="fruit-b"), level=0),
        RAPTORNode(Document("quantum", doc_id="physics-a"), level=0),
        RAPTORNode(Document("photon", doc_id="physics-b"), level=0),
    ]
    summaries = [
        RAPTORNode(
            Document("apple pear", metadata={"node_type": "summary"}, doc_id="summary-fruit"),
            level=1,
            child_ids=["fruit-a", "fruit-b"],
        ),
        RAPTORNode(
            Document("quantum photon", metadata={"node_type": "summary"}, doc_id="summary-physics"),
            level=1,
            child_ids=["physics-a", "physics-b"],
        ),
    ]
    root = RAPTORNode(
        Document("apple pear quantum photon", metadata={"node_type": "summary"}, doc_id="root"),
        level=2,
        child_ids=["summary-fruit", "summary-physics"],
    )
    # Deliberately store children before parents: traversal must use edges and
    # levels rather than depending on insertion order.
    return RAPTORIndex(nodes=[*leaves, *summaries, root])


class RAPTORTraversalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding = _KeywordEmbedding()

    def test_tree_mode_follows_summary_path_to_relevant_leaf(self) -> None:
        retriever = RAPTORRetrieverV2(
            _tree_index(),
            embedding_model=self.embedding,
            retrieval_mode="tree_traversal",
            beam_width=1,
            traversal_budget=5,
        )

        documents = retriever.retrieve("apple", top_k=1)

        self.assertEqual([document.doc_id for document in documents], ["fruit-a"])
        document = documents[0]
        self.assertEqual(document.metadata["raptor_retrieval_mode"], "tree")
        self.assertEqual(
            document.metadata["raptor_path"],
            ["root", "summary-fruit", "fruit-a"],
        )
        self.assertEqual(document.metadata["raptor_path_levels"], [2, 1, 0])
        self.assertEqual(len(document.metadata["raptor_path_scores"]), 3)
        self.assertTrue(document.metadata["raptor_terminal"])
        self.assertEqual(document.metadata["raptor_visited_nodes"], 3)
        self.assertEqual(document.metadata["raptor_scored_nodes"], 5)
        self.assertLessEqual(
            document.metadata["raptor_scored_nodes"],
            document.metadata["raptor_traversal_budget"],
        )
        self.assertLess(document.metadata["raptor_scored_nodes"], len(_tree_index().nodes))
        self.assertLessEqual(len(documents), 1)

    def test_budget_returns_deepest_frontier_without_exceeding_limit(self) -> None:
        retriever = RAPTORRetrieverV2(
            _tree_index(),
            embedding_model=self.embedding,
            retrieval_mode="tree",
            beam_width=1,
            traversal_budget=2,
        )

        documents = retriever.retrieve("apple", top_k=5)

        self.assertEqual([document.doc_id for document in documents], ["summary-fruit"])
        self.assertEqual(documents[0].metadata["raptor_visited_nodes"], 2)
        self.assertEqual(documents[0].metadata["raptor_scored_nodes"], 2)
        self.assertEqual(documents[0].metadata["raptor_traversal_budget"], 2)
        self.assertFalse(documents[0].metadata["raptor_terminal"])

    def test_beam_and_top_k_are_hard_limits(self) -> None:
        retriever = RAPTORRetrieverV2(
            _tree_index(),
            embedding_model=self.embedding,
            retrieval_mode="tree",
            beam_width=2,
            traversal_budget=7,
        )

        documents = retriever.retrieve("apple quantum", top_k=2)

        self.assertLessEqual(len(documents), 2)
        self.assertTrue(all(document.metadata["raptor_beam_width"] == 2 for document in documents))
        self.assertTrue(all(document.metadata["raptor_visited_nodes"] <= 7 for document in documents))
        self.assertTrue(all(document.metadata["raptor_terminal"] for document in documents))

    def test_equal_scores_have_stable_id_tie_breaking(self) -> None:
        retriever = RAPTORRetrieverV2(
            _tree_index(),
            embedding_model=self.embedding,
            retrieval_mode="tree",
            beam_width=1,
            traversal_budget=5,
        )

        first = retriever.retrieve("unknown", top_k=1)
        second = retriever.retrieve("unknown", top_k=1)

        self.assertEqual(first[0].doc_id, "fruit-a")
        self.assertEqual(first[0].doc_id, second[0].doc_id)
        self.assertEqual(first[0].score, second[0].score)

    def test_flat_mode_remains_default_and_includes_tree_provenance(self) -> None:
        retriever = RAPTORRetrieverV2(_tree_index(), embedding_model=self.embedding)

        documents = retriever.retrieve("apple", top_k=1)

        self.assertEqual(documents[0].doc_id, "fruit-a")
        self.assertEqual(documents[0].metadata["raptor_retrieval_mode"], "flat")
        self.assertEqual(
            documents[0].metadata["raptor_path"],
            ["root", "summary-fruit", "fruit-a"],
        )

    def test_index_and_result_boundaries_are_defensive_snapshots(self) -> None:
        index = _tree_index()
        retriever = RAPTORRetrieverV2(
            index,
            embedding_model=self.embedding,
            retrieval_mode="tree",
            beam_width=1,
            traversal_budget=5,
        )
        index.nodes[0].document.content = "mutated"
        index.nodes[0].document.metadata["nested"]["owner"] = "mutated"
        index.nodes[-1].child_ids.clear()

        first = retriever.retrieve("apple", top_k=1)
        first[0].content = "changed result"
        first[0].metadata["nested"]["owner"] = "changed result"
        first[0].metadata["raptor_path"].clear()
        second = retriever.retrieve("apple", top_k=1)

        self.assertEqual(second[0].content, "apple")
        self.assertEqual(second[0].metadata["nested"]["owner"], "original")
        self.assertEqual(second[0].metadata["raptor_path"], ["root", "summary-fruit", "fruit-a"])

    def test_controls_use_strict_positive_integer_contract(self) -> None:
        invalid_values = ((0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError))
        for value, expected_error in invalid_values:
            with self.subTest(control="beam_width", value=value):
                with self.assertRaises(expected_error):
                    RAPTORRetrieverV2(_tree_index(), beam_width=value)  # type: ignore[arg-type]
            with self.subTest(control="traversal_budget", value=value):
                with self.assertRaises(expected_error):
                    RAPTORRetrieverV2(_tree_index(), traversal_budget=value)  # type: ignore[arg-type]
            with self.subTest(control="request beam_width", value=value):
                retriever = RAPTORRetrieverV2(_tree_index(), embedding_model=self.embedding)
                with self.assertRaises(expected_error):
                    retriever.retrieve("apple", beam_width=value)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            RAPTORRetrieverV2(_tree_index(), retrieval_mode="not-a-mode")
        with self.assertRaises(TypeError):
            RAPTORRetrieverV2(_tree_index(), retrieval_mode=1)  # type: ignore[arg-type]

    def test_index_graph_and_numeric_boundaries_fail_closed(self) -> None:
        with self.assertRaises(TypeError):
            RAPTORIndex(nodes="not nodes")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RAPTORNode(Document("root", doc_id="root"), level=1, child_ids="leaf")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            RAPTORRetrieverV2(
                RAPTORIndex(
                    [
                        RAPTORNode(
                            Document("root", doc_id="root"),
                            level=1,
                            child_ids=["missing"],
                        )
                    ]
                )
            )
        with self.assertRaises(ValueError):
            RAPTORRetrieverV2(
                RAPTORIndex(
                    [
                        RAPTORNode(
                            Document("root", doc_id="root"),
                            level=1,
                            child_ids=["same-level"],
                        ),
                        RAPTORNode(Document("child", doc_id="same-level"), level=1),
                    ]
                )
            )
        with self.assertRaises(ValueError):
            RAPTORNode(Document("bad", doc_id="bad", score=float("nan")), level=0)

        retriever = RAPTORRetrieverV2(_tree_index(), embedding_model=self.embedding)
        for invalid, expected_error in (("", ValueError), ("   ", ValueError), (1, TypeError)):
            with self.subTest(query=invalid):
                with self.assertRaises(expected_error):
                    retriever.retrieve(invalid)  # type: ignore[arg-type]

    def test_engine_can_enable_tree_mode_without_changing_ask_api(self) -> None:
        documents = [
            Document("apple", doc_id="a"),
            Document("pear", doc_id="b"),
        ]
        engine = RAPTOREngine.from_documents(
            documents,
            embedding_model=self.embedding,
            llm_client=StaticLLMClient("apple pear"),
            levels=1,
            branching_factor=2,
            retrieval_mode="tree",
            beam_width=2,
            traversal_budget=3,
            top_k=1,
        )

        response = engine.ask("apple")

        self.assertEqual(response.metadata["architecture"], "raptor")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].metadata["raptor_retrieval_mode"], "tree")


if __name__ == "__main__":
    unittest.main()
