"""RAPTOR summary coverage and layer selection against the official mechanism."""
from unittest.mock import patch

import pytest

from cheragh import Document, HashingEmbedding, StaticLLMClient
from cheragh.raptor_engine import RAPTOREngine, RAPTORIndex, RAPTORNode, RAPTORRetrieverV2, UMAPGMMClusterer


def test_singleton_soft_cluster_remains_represented_at_every_coarser_level():
    def clusters(matrix):
        return [[0, 1], [2]] if len(matrix) == 3 else [[0, 1]]

    with patch.object(UMAPGMMClusterer, "cluster", side_effect=clusters):
        engine = RAPTOREngine(
            [Document("shared fact A", doc_id="a"), Document("shared fact B", doc_id="b"),
             Document("unique critical fact C", doc_id="c")],
            clustering_mode="umap_gmm", levels=2, llm_client=StaticLLMClient("A factual summary"),
        )
    assert [node.child_ids for node in engine.index.levels()[1]] == [["a", "b"], ["c"]]
    assert len(engine.index.levels()[2]) == 1
    parent = engine.index.levels()[2][0]
    assert set(parent.child_ids) == {node.document.doc_id for node in engine.index.levels()[1]}


def test_greedy_summaries_preserve_full_content_and_forward_the_output_limit():
    class Summarizer:
        calls = []

        def generate(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return "summary"

    llm = Summarizer()
    documents = [Document(str(i) * 100, doc_id=str(i)) for i in range(4)]
    engine = RAPTOREngine(documents, llm_client=llm, token_estimator=len,
                         summary_input_token_budget=350, summary_max_tokens=40)
    assert all(len(prompt) <= 350 for prompt, _ in llm.calls)
    assert all(kwargs == {"max_tokens": 40} for _, kwargs in llm.calls)
    assert all(any(document.content in prompt for prompt, _ in llm.calls) for document in documents)
    assert {child for node in engine.index.levels()[1] for child in node.child_ids} == {"0", "1", "2", "3"}


def index():
    return RAPTORIndex([
        RAPTORNode(Document("astronomy leaf", doc_id="a"), level=0),
        RAPTORNode(Document("astronomy summary", doc_id="b"), level=1, child_ids=["a"]),
        RAPTORNode(Document("astronomy overview", doc_id="c"), level=2, child_ids=["b"]),
    ])


def test_paper_traversal_starts_at_selected_layer_and_limits_layer_count():
    retriever = RAPTORRetrieverV2(index(), HashingEmbedding(), retrieval_mode="paper_tree")
    selected = retriever.retrieve("astronomy", top_k=10, start_level=1, num_levels=1)
    assert [doc.doc_id for doc in selected] == ["b"]
    selected = retriever.retrieve("astronomy", top_k=10, start_level=1, num_levels=2)
    assert [doc.doc_id for doc in selected] == ["b", "a"]
    assert selected[1].metadata["raptor_path"] == ["b", "a"]


@pytest.mark.parametrize("options", [
    {"start_level": 3}, {"num_levels": 4}, {"start_level": 0, "num_levels": 2},
    {"start_level": True}, {"num_levels": 0},
])
def test_invalid_layer_selection_is_rejected(options):
    retriever = RAPTORRetrieverV2(index(), retrieval_mode="paper_tree")
    with pytest.raises((ValueError, TypeError)):
        retriever.retrieve("astronomy", **options)


def test_layer_controls_cannot_silently_affect_collapsed_retrieval():
    with pytest.raises(ValueError, match="paper_tree"):
        RAPTORRetrieverV2(index(), start_level=0)
