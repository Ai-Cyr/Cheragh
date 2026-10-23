from itertools import repeat

import pytest

from cheragh import Document, RAGEngine, StaticLLMClient
from cheragh.evaluation import evaluate_generation, evaluate_pipeline, evaluate_retrieval
from cheragh.feedback import FeedbackLoop, FeedbackRecord, InMemoryFeedbackStore
from cheragh.training import HardNegativeMiner, RetrievalTrainingExample, TeacherScoreDistiller
from cheragh.tracing import RAGTrace


class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        return iter(self.documents)


def test_miner_honors_candidate_budget_even_when_every_result_is_excluded():
    positive = Document("oracle", doc_id="positive")
    inspected = []

    def candidates():
        for index in range(3):
            inspected.append(index)
            yield positive
        raise AssertionError("miner exceeded its candidate budget")

    miner = HardNegativeMiner(Retriever(candidates()), candidate_top_k=3, negatives_per_query=1)
    example = miner.mine("question", [positive])
    assert example.negative_documents == ()
    assert inspected == [0, 1, 2]


def test_mining_filter_cannot_mutate_corpus_or_training_evidence():
    negative = Document("original", doc_id="negative", metadata={"tenant": {"id": "a"}})

    def mutating_filter(query, document):
        document.content = "changed"
        document.metadata["tenant"]["id"] = "b"
        return False

    miner = HardNegativeMiner(Retriever([negative]), negatives_per_query=1, exclusion_filter=mutating_filter)
    example = miner.mine("question", [Document("oracle", doc_id="positive")])
    assert negative.content == example.negative_documents[0].content == "original"
    assert negative.metadata == example.negative_documents[0].metadata == {"tenant": {"id": "a"}}


def test_miner_rejects_invalid_query_before_calling_provider():
    retriever = Retriever([])
    miner = HardNegativeMiner(retriever)
    with pytest.raises(ValueError, match="query"):
        miner.mine("  ", [Document("oracle", doc_id="positive")])
    assert retriever.calls == []


@pytest.mark.parametrize("scores,expected", [([1e308, 1e308], (0.5, 0.5)), ([1e308, -1e308], (1., 0.))])
def test_teacher_softmax_preserves_finite_scores_at_small_temperatures(scores, expected):
    example = RetrievalTrainingExample("query", (Document("p", doc_id="p"),), (Document("n", doc_id="n"),))
    distilled = TeacherScoreDistiller(lambda query, docs: scores, temperature=1e-300).distill(example)
    assert distilled.document_probabilities == expected


def test_feedback_store_snapshots_input_output_and_public_records():
    store = InMemoryFeedbackStore()
    record = FeedbackRecord("question", "positive", tenant_id="a", metadata={"nested": {"label": "original"}})
    store.append(record)
    record.tenant_id = "b"
    record.metadata["nested"]["label"] = "input mutation"
    returned = store.list(tenant_id="a")
    assert len(returned) == 1
    returned[0].metadata["nested"]["label"] = "output mutation"
    store.records[0].tenant_id = "c"
    assert store.list(tenant_id="b") == []
    assert store.list(tenant_id="a")[0].metadata == {"nested": {"label": "original"}}


def test_feedback_return_value_cannot_rewrite_recorded_rating():
    loop = FeedbackLoop()
    feedback = loop.log_feedback("question", "positive")
    feedback.rating = "negative"
    assert loop.summary().positive_rate == 1.0


def test_pipeline_evaluates_actual_transformed_evidence_with_one_call():
    useful = Document("The answer is blue.", doc_id="useful")

    class QueryTransformer:
        def transform(self, query):
            return ["rewritten"]

    class RoutedRetriever(Retriever):
        def retrieve(self, query, top_k=5):
            self.calls.append((query, top_k))
            return [useful] if query == "rewritten" else []

    retriever = RoutedRetriever([])
    engine = RAGEngine(retriever, StaticLLMClient("blue [source: useful]"), query_transformer=QueryTransformer())
    result = evaluate_pipeline(engine, [{"question": "original", "expected": ["useful"]}], top_k=1)
    assert result.retrieval_metrics["hit_rate@1"] == 1.0
    assert result.generation_metrics["citation_coverage"] == 1.0
    assert retriever.calls == [("rewritten", 1)]


def test_generation_distinguishes_expected_coverage_from_available_accuracy():
    result = evaluate_generation([{
        "query": "question", "answer": "answer [source: a]", "contexts": ["answer"],
        "source_ids": ["a"], "expected_doc_ids": ["a", "b"],
    }])
    assert result.metrics["citation_accuracy"] == 1.0
    assert result.metrics["citation_coverage"] == 0.5


def test_retrieval_evaluation_bounds_generator_results():
    retriever = Retriever(repeat(Document("answer", doc_id="a")))
    result = evaluate_retrieval([{"query": "question", "expected_doc_ids": ["a"]}], retriever, top_k=2)
    assert result.rows[0]["retrieved_doc_ids"] == ["a", "a"]


def test_pipeline_validates_top_k_even_for_an_empty_dataset():
    with pytest.raises(TypeError):
        evaluate_pipeline(object(), [], top_k=True)


def test_trace_history_isolated_from_document_and_serialization_mutations():
    trace = RAGTrace()
    document = Document("evidence", metadata={"acl": {"tenant": "a"}})
    details = {"nested": {"status": "original"}}
    step = trace.start_step("retrieval", details=details)
    trace.add_retrieval("question", [document])
    step.finish(details=details)
    trace.finish(details=details)
    document.metadata["acl"]["tenant"] = "b"
    details["nested"]["status"] = "changed"
    serialized = trace.to_dict()
    assert serialized["retrieval"][0]["documents"][0]["metadata"]["acl"] == {"tenant": "a"}
    assert serialized["steps"][0]["metadata"]["details"]["nested"] == {"status": "original"}
    assert serialized["metadata"]["details"]["nested"] == {"status": "original"}
    serialized["retrieval"].clear()
    serialized["metadata"]["details"].clear()
    step.to_dict()["metadata"].clear()
    assert trace.retrieval
    assert trace.metadata["details"]
    assert step.metadata
