"""Mechanism and boundary tests; these do not claim paper benchmark parity."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from cheragh import Document, EmbeddingModel, StaticLLMClient
from cheragh.raptor_engine import (
    RAPTORClusteringConfig, RAPTOREngine, RAPTORIndex, RAPTORNode,
    RAPTORRetrieverV2, UMAPGMMClusterer,
)


class _Vectors(EmbeddingModel):
    def __init__(self, vectors=None):
        self.vectors = vectors or {}

    def embed_documents(self, texts):
        return np.asarray([self.embed_query(text) for text in texts])

    def embed_query(self, text):
        return np.asarray(self.vectors.get(text, [1.0, 0.0]), dtype=np.float64)


class _IdentityUMAP:
    calls = []

    def __init__(self, **kwargs):
        self.calls.append(kwargs)

    def fit_transform(self, matrix):
        return matrix


class _FixedMixture:
    """Controlled posterior and BIC so integration can test exact memberships."""
    components = []

    def __init__(self, n_components, **kwargs):
        self.n_components = n_components
        self.converged_ = True
        self.components.append(n_components)

    def fit(self, matrix):
        return self

    def bic(self, matrix):
        return 0.0 if self.n_components == 2 else 100.0

    def predict_proba(self, matrix):
        if self.n_components == 1:
            return np.ones((len(matrix), 1))
        return np.asarray([[0.9, 0.1] if row[0] < 0 else [0.1, 0.9] if row[0] > 0 else [0.5, 0.5]
                           for row in matrix])


class RAPTORSoftClusteringTests(unittest.TestCase):
    def test_global_memberships_overlap_and_preserve_row_identity(self):
        matrix = np.asarray([[-1., 0.], [-1., 0.], [0., 1.], [1., 0.]])
        _IdentityUMAP.calls = []
        _FixedMixture.components = []
        clusterer = UMAPGMMClusterer(RAPTORClusteringConfig(membership_threshold=0.2))
        with patch.object(clusterer, "_load_dependencies", return_value=(_IdentityUMAP, _FixedMixture, UserWarning)):
            groups = clusterer.cluster(matrix)
        self.assertEqual(groups, [[0, 1, 2], [2, 3]])
        self.assertEqual(_FixedMixture.components, [1, 2, 3])
        self.assertEqual(_IdentityUMAP.calls[0]["random_state"], 224)
        self.assertEqual(_IdentityUMAP.calls[0]["n_jobs"], 1)
        self.assertEqual(_IdentityUMAP.calls[0]["n_neighbors"], 2)

    def test_global_then_local_stages_use_distinct_neighbors(self):
        clusterer = UMAPGMMClusterer(RAPTORClusteringConfig(reduction_dimension=2))
        matrix = np.arange(40, dtype=float).reshape(20, 2)
        _IdentityUMAP.calls = []
        with patch.object(clusterer, "_load_dependencies", return_value=(_IdentityUMAP, _FixedMixture, UserWarning)), \
             patch.object(clusterer, "_soft_cluster", side_effect=[[[0]] * 20, [[i % 2] for i in range(20)]]):
            groups = clusterer.cluster(matrix)
        self.assertEqual([call["n_neighbors"] for call in _IdentityUMAP.calls], [4, 10])
        self.assertEqual(groups, [list(range(0, 20, 2)), list(range(1, 20, 2))])

    def test_high_threshold_retains_argmax_membership(self):
        clusterer = UMAPGMMClusterer(RAPTORClusteringConfig(membership_threshold=0.99))
        memberships = clusterer._soft_cluster(np.asarray([[-1.], [0.], [1.]]), _FixedMixture, UserWarning)
        self.assertEqual(memberships, [[0], [0], [1]])

    def test_tiny_and_identical_inputs_do_not_load_optional_dependencies(self):
        clusterer = UMAPGMMClusterer()
        with patch.object(clusterer, "_load_dependencies", side_effect=AssertionError("eager import")):
            self.assertEqual(clusterer.cluster(np.zeros((0, 3))), [])
            self.assertEqual(clusterer.cluster([[1., 0.]]), [[0]])
            self.assertEqual(clusterer.cluster([[1., 0.], [0., 1.]]), [[0, 1]])
            self.assertEqual(clusterer.cluster(np.ones((12, 3))), [list(range(12))])

    def test_optional_dependency_error_is_actionable(self):
        import builtins
        real_import = builtins.__import__

        def missing_umap(name, *args, **kwargs):
            if name == "umap":
                raise ImportError("optional module absent")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing_umap), self.assertRaisesRegex(ImportError, r"cheragh\[raptor\]"):
            UMAPGMMClusterer().cluster([[0., 1.], [1., 0.], [0.5, 0.5]])

    def test_invalid_or_excessive_embeddings_fail_before_models(self):
        clusterer = UMAPGMMClusterer(RAPTORClusteringConfig(max_cluster_points=3))
        for matrix in ([1, 2], [[1], [1, 2]], [[np.nan]], [[np.inf]], np.ones((4, 2)), np.empty((2, 0))):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                clusterer.cluster(matrix)

    def test_config_validation(self):
        for kwargs in (
            {"random_state": -1}, {"random_state": 2**32}, {"random_state": True},
            {"membership_threshold": 1.0}, {"membership_threshold": float("nan")},
            {"membership_threshold": True}, {"reg_covar": 0}, {"max_iter": 0},
            {"local_neighbors": 1}, {"n_init": 1.5}, {"max_cluster_points": False},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                RAPTORClusteringConfig(**kwargs)

    def test_all_nonconverged_candidates_fail_explicitly(self):
        class Nonconverged(_FixedMixture):
            def fit(self, matrix):
                self.converged_ = False
                return self

        with self.assertRaisesRegex(ValueError, "converged"):
            UMAPGMMClusterer()._soft_cluster(np.asarray([[-1.], [0.], [1.]]), Nonconverged, UserWarning)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "requires cheragh[raptor]")
    def test_real_gmm_can_assign_multiple_memberships(self):
        from sklearn.mixture import GaussianMixture
        from sklearn.exceptions import ConvergenceWarning

        random = np.random.default_rng(7)
        left = random.normal(-2, 0.6, (60, 1))
        matrix = np.vstack([left, -left, [[0.0]]])
        clusterer = UMAPGMMClusterer(RAPTORClusteringConfig(max_clusters=2, membership_threshold=0.1))
        labels = clusterer._soft_cluster(matrix, GaussianMixture, ConvergenceWarning)
        self.assertEqual(len(labels[-1]), 2)
        self.assertTrue(all(labels))

    @unittest.skipUnless(importlib.util.find_spec("umap"), "requires cheragh[raptor]")
    def test_real_umap_gmm_is_seeded_and_cover_complete(self):
        random = np.random.default_rng(4)
        matrix = random.normal(size=(24, 5))
        clusterer = UMAPGMMClusterer(RAPTORClusteringConfig(reduction_dimension=2, max_clusters=3))
        first = clusterer.cluster(matrix)
        second = clusterer.cluster(matrix)
        self.assertEqual(first, second)
        self.assertEqual(set(index for group in first for index in group), set(range(24)))


class RAPTORPaperTreeTests(unittest.TestCase):
    def test_soft_groups_create_shared_child_edges_and_retrieval_deduplicates(self):
        documents = [Document(str(i), doc_id=str(i)) for i in range(4)]
        with patch.object(UMAPGMMClusterer, "cluster", return_value=[[0, 1, 2], [2, 3]]):
            engine = RAPTOREngine(
                documents, embedding_model=_Vectors(), llm_client=StaticLLMClient("summary"),
                clustering_mode="umap_gmm", levels=1, retrieval_mode="paper_tree", beam_width=4,
            )
        summaries = engine.index.levels()[1]
        self.assertEqual([node.child_ids for node in summaries], [["0", "1", "2"], ["2", "3"]])
        outputs = engine.retrieve("question", top_k=10)
        self.assertEqual(len(outputs), 6)
        self.assertEqual(len({doc.doc_id for doc in outputs}), 6)
        self.assertEqual(outputs[0].metadata["raptor_level"], 1)
        self.assertEqual(outputs[-1].metadata["raptor_level"], 0)

    def test_paper_tree_returns_ancestors_and_ranks_by_own_cosine(self):
        # Root A is more relevant, but root B's child is the more relevant leaf.
        nodes = [
            RAPTORNode(Document("root-a", doc_id="root-a"), 1, ["leaf-a"]),
            RAPTORNode(Document("root-b", doc_id="root-b"), 1, ["leaf-b"]),
            RAPTORNode(Document("leaf-a", doc_id="leaf-a"), 0),
            RAPTORNode(Document("leaf-b", doc_id="leaf-b"), 0),
        ]
        embedding = _Vectors({"root-a": [1, 0], "root-b": [0.1, 1], "leaf-a": [0.5, 1], "leaf-b": [0.6, 1]})
        retriever = RAPTORRetrieverV2(RAPTORIndex(nodes), embedding, retrieval_mode="paper_tree", beam_width=2)
        result = retriever.retrieve("query", top_k=4)
        self.assertEqual([doc.doc_id for doc in result], ["root-a", "root-b", "leaf-b", "leaf-a"])
        self.assertGreater(result[2].score, result[3].score)
        self.assertEqual(result[2].metadata["raptor_path"], ["root-b", "leaf-b"])
        bounded = retriever.retrieve("query", top_k=4, traversal_budget=2)
        self.assertEqual(len(bounded), 2)
        self.assertEqual(bounded[0].metadata["raptor_scored_nodes"], 2)

    def test_collapsed_token_budget_counts_concatenation_and_stops(self):
        nodes = [RAPTORNode(Document(text, doc_id=str(i)), 0) for i, text in enumerate(["abc", "defg", "h"])]
        retriever = RAPTORRetrieverV2(RAPTORIndex(nodes), _Vectors(), token_estimator=len, retrieval_token_budget=8)
        result = retriever.retrieve("q", top_k=3)
        self.assertEqual([doc.content for doc in result], ["abc"])
        self.assertEqual(result[0].metadata["raptor_context_tokens_so_far"], 3)
        result = retriever.retrieve("q", top_k=3, retrieval_token_budget=9)
        self.assertEqual([doc.content for doc in result], ["abc", "defg"])
        self.assertEqual(result[1].metadata["raptor_context_tokens_so_far"], 9)

    def test_budget_validation_even_on_empty_index(self):
        retriever = RAPTORRetrieverV2(RAPTORIndex())
        with self.assertRaises(ValueError):
            retriever.retrieve("q", retrieval_token_budget=0)
        retriever = RAPTORRetrieverV2(RAPTORIndex([RAPTORNode(Document("a"), 0)]), token_estimator=lambda _: True)
        with self.assertRaises(TypeError):
            retriever.retrieve("q", retrieval_token_budget=2)

    def test_reclustering_terminates_and_summary_prompt_never_truncates(self):
        documents = [Document((str(i) + " ") * 12, doc_id=str(i)) for i in range(4)]
        prompts = []

        class CaptureLLM:
            def generate(self, prompt, **kwargs):
                prompts.append(prompt)
                return "summary"

        # Budget allows pairs but not the entire set. A non-shrinking cluster
        # must terminate via bisection with no source text silently sliced.
        with patch.object(UMAPGMMClusterer, "cluster", side_effect=lambda matrix: [list(range(len(matrix)))]):
            engine = RAPTOREngine(
                documents, embedding_model=_Vectors(), llm_client=CaptureLLM(),
                clustering_mode="umap_gmm", levels=1, token_estimator=len,
                summary_input_token_budget=280,
            )
        self.assertEqual(len(prompts), 2)
        self.assertTrue(all(len(prompt) <= 280 for prompt in prompts))
        self.assertEqual([node.child_ids for node in engine.index.levels()[1]], [["0", "1"], ["2", "3"]])
        for document in documents:
            self.assertTrue(any(document.content in prompt for prompt in prompts))

    def test_oversized_single_node_fails_instead_of_losing_content(self):
        with self.assertRaisesRegex(ValueError, "smaller chunks"):
            RAPTOREngine(
                [Document("x" * 500), Document("short")], embedding_model=_Vectors(),
                clustering_mode="umap_gmm", summary_input_token_budget=250, token_estimator=len,
            )

    def test_paper_mode_rejects_empty_generated_summary(self):
        with self.assertRaisesRegex(ValueError, "empty RAPTOR summary"):
            RAPTOREngine(
                [Document("a"), Document("b")], embedding_model=_Vectors(),
                llm_client=StaticLLMClient(""), clustering_mode="umap_gmm",
            )

    def test_legacy_mode_does_not_invoke_optional_clustering(self):
        with patch.object(UMAPGMMClusterer, "cluster", side_effect=AssertionError("unexpected model")):
            engine = RAPTOREngine([Document("one"), Document("two")], llm_client=StaticLLMClient("summary"))
        self.assertEqual(engine.clustering_mode, "greedy")
        self.assertEqual(len(engine.index.levels()[1]), 1)

    def test_configuration_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "requires clustering_mode"):
            RAPTOREngine([], clustering_config=RAPTORClusteringConfig())


if __name__ == "__main__":
    unittest.main()
