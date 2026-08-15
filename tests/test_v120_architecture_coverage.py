from __future__ import annotations

import re
import unittest
from collections.abc import Callable, Sequence

import numpy as np

from cheragh.base import BaseRetriever, Document, EmbeddingModel, LLMClient
from cheragh.adaptive import AdaptiveRetriever, GateDecision
from cheragh.chain_of_note import ChainOfNoteRetriever
from cheragh.contextual_compression import ContextualCompressionRetriever
from cheragh.flare import FLAREPipeline
from cheragh.hyde import HyDERetriever
from cheragh.hyqe import HyQERetriever
from cheragh.mmr import MMRRetriever
from cheragh.propositional import PropositionalRetriever
from cheragh.query_decomposition import QueryDecompositionRetriever
from cheragh.rag_fusion import RAGFusionRetriever
from cheragh.reranking import (
    CrossEncoderReranker,
    ReciprocalRankFusionReranker,
    ReciprocalRankFusionRetriever,
)
from cheragh.self_query import SelfQueryRetriever
from cheragh.sentence_window import SentenceWindowRetriever
from cheragh.step_back import StepBackRetriever


class _TokenEmbedding(EmbeddingModel):
    """Tiny deterministic embedding used to keep architecture tests offline."""

    def __init__(self, vocabulary: Sequence[str]) -> None:
        self.vocabulary = tuple(token.lower() for token in vocabulary)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.vocabulary)), dtype=np.float32)
        return np.vstack([self.embed_query(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        tokens = set(re.findall(r"\w+", text.lower()))
        vector = np.asarray(
            [1.0 if token in tokens else 0.0 for token in self.vocabulary],
            dtype=np.float32,
        )
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


class _TableEmbedding(EmbeddingModel):
    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = {
            text: np.asarray(vector, dtype=np.float32) for text, vector in vectors.items()
        }
        self.dimension = len(next(iter(self.vectors.values())))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self.embed_query(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self.vectors[text].copy()


class _QueueLLM(LLMClient):
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError(f"unexpected LLM call: {prompt[:80]}")
        return self.responses.pop(0)


class _RoutingRetriever(BaseRetriever):
    def __init__(
        self,
        routes: dict[str, Sequence[Document]] | None = None,
        *,
        default: Sequence[Document] = (),
    ) -> None:
        self.routes = {query: list(documents) for query, documents in (routes or {}).items()}
        self.default = list(default)
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        self.calls.append((query, top_k))
        return self.routes.get(query, self.default)[:top_k]


class _FakeCrossEncoder:
    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.pairs = list(pairs)
        return self.scores[: len(pairs)]


class ArchitectureCoverageTests(unittest.TestCase):
    def assert_strict_top_k(self, invoke: Callable[[object], object]) -> None:
        for value, error in (
            (0, ValueError),
            (-1, ValueError),
            (True, TypeError),
            (1.5, TypeError),
        ):
            with self.subTest(top_k=value), self.assertRaises(error):
                invoke(value)

    def test_internal_candidate_and_iteration_limits_are_strict(self) -> None:
        base = _RoutingRetriever()
        llm = _QueueLLM([])
        embedding = _TokenEmbedding(["token"])
        factories = {
            "mmr.fetch_k": lambda value: MMRRetriever(
                embedding,
                documents=[],
                fetch_k=value,
            ),
            "rag_fusion.n_queries": lambda value: RAGFusionRetriever(
                base,
                llm,
                n_queries=value,
            ),
            "rag_fusion.rrf_k": lambda value: RAGFusionRetriever(
                base,
                llm,
                rrf_k=value,
            ),
            "rag_fusion.per_query_top_k": lambda value: RAGFusionRetriever(
                base,
                llm,
                per_query_top_k=value,
            ),
            "step_back.n_original": lambda value: StepBackRetriever(
                base,
                llm,
                n_original=value,
            ),
            "step_back.n_stepback": lambda value: StepBackRetriever(
                base,
                llm,
                n_stepback=value,
            ),
            "decomposition.max_subquestions": lambda value: QueryDecompositionRetriever(
                base,
                llm,
                max_subquestions=value,
            ),
            "decomposition.per_subquestion_top_k": lambda value: QueryDecompositionRetriever(
                base,
                llm,
                per_subquestion_top_k=value,
            ),
            "chain_of_note.fetch_multiplier": lambda value: ChainOfNoteRetriever(
                base,
                llm,
                fetch_multiplier=value,
            ),
            "flare.max_iterations": lambda value: FLAREPipeline(
                base,
                llm,
                max_iterations=value,
            ),
            "flare.retrieval_top_k": lambda value: FLAREPipeline(
                base,
                llm,
                retrieval_top_k=value,
            ),
            "hyqe.n_questions_per_doc": lambda value: HyQERetriever(
                [],
                embedding,
                llm,
                n_questions_per_doc=value,
            ),
        }
        for name, factory in factories.items():
            for value, error in ((0, ValueError), (True, TypeError), (1.5, TypeError)):
                with self.subTest(parameter=name, value=value), self.assertRaises(error):
                    factory(value)

    def test_sentence_window_retrieves_local_context_around_best_sentence(self) -> None:
        source = Document(
            "Alpha introduction. Beta target fact. Gamma conclusion.",
            metadata={"source": "guide"},
            doc_id="guide",
        )
        retriever = SentenceWindowRetriever(
            [source], _TokenEmbedding(["alpha", "target", "gamma"]), window_size=1
        )

        result = retriever.retrieve("target", top_k=1)

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].content,
            "Alpha introduction. Beta target fact. Gamma conclusion.",
        )
        self.assertEqual(result[0].metadata["matched_sentence"], "Beta target fact.")
        self.assertEqual(result[0].metadata["window_start"], 0)
        self.assertEqual(result[0].metadata["window_end"], 3)
        self.assertEqual(result[0].doc_id, "guide::win::0-3")
        self.assert_strict_top_k(lambda value: retriever.retrieve("target", top_k=value))

    def test_adaptive_gate_rephrases_or_skips_and_snapshots_results(self) -> None:
        source = Document("policy", doc_id="policy", metadata={"nested": {"owner": "base"}})
        base = _RoutingRetriever(routes={"precise policy": [source]})
        retriever = AdaptiveRetriever(base, _QueueLLM(["REPHRASE", "precise policy"]))

        result = retriever.retrieve("it?", top_k=1)

        self.assertEqual(retriever.last_decision, GateDecision.REPHRASE)
        self.assertEqual(retriever.last_used_query, "precise policy")
        self.assertEqual(result[0].metadata["gate_used_query"], "precise policy")
        result[0].metadata["nested"]["owner"] = "caller"
        self.assertEqual(source.metadata["nested"]["owner"], "base")

        skipped = AdaptiveRetriever(base, _QueueLLM(["NO_RETRIEVE"]))
        self.assertEqual(skipped.retrieve("bonjour", top_k=1), [])
        self.assertEqual(skipped.last_decision, GateDecision.NO_RETRIEVE)
        with self.assertRaises(TypeError):
            AdaptiveRetriever(base, _QueueLLM([]), allow_rephrase=1)

    def test_contextual_compression_filters_and_preserves_source_metadata(self) -> None:
        relevant = Document(
            "Long relevant source text.",
            doc_id="relevant",
            metadata={"nested": {"owner": "base"}},
        )
        irrelevant = Document("Noise only.", doc_id="noise")
        base = _RoutingRetriever(default=[relevant, irrelevant])
        retriever = ContextualCompressionRetriever(
            base,
            _QueueLLM(["Relevant extracted sentence.", "NO_OUTPUT"]),
            min_compressed_length=5,
        )

        result = retriever.retrieve("question", top_k=2)

        self.assertEqual([document.doc_id for document in result], ["relevant"])
        self.assertEqual(result[0].content, "Relevant extracted sentence.")
        self.assertTrue(result[0].metadata["was_compressed"])
        result[0].metadata["nested"]["owner"] = "caller"
        self.assertEqual(relevant.metadata["nested"]["owner"], "base")
        self.assert_strict_top_k(lambda value: retriever.retrieve("question", top_k=value))
        with self.assertRaises(TypeError):
            ContextualCompressionRetriever(base, _QueueLLM([]), min_compressed_length=True)

    def test_propositional_retrieval_indexes_atomic_facts_then_returns_parent(self) -> None:
        documents = [
            Document("Alpha report.", doc_id="alpha"),
            Document("Beta report.", doc_id="beta"),
        ]
        llm = _QueueLLM(
            [
                "Alpha revenue is 100 euros.\nAlpha uses product A.",
                "Beta revenue is 200 euros.\nBeta uses product B.",
            ]
        )
        retriever = PropositionalRetriever(
            documents,
            _TokenEmbedding(["alpha", "beta", "100", "200", "product"]),
            llm,
        )

        result = retriever.retrieve("revenue 200", top_k=1)

        self.assertEqual(result[0].doc_id, "beta")
        self.assertEqual(result[0].content, "Beta report.")
        self.assertEqual(result[0].metadata["matched_proposition"], "Beta revenue is 200 euros.")
        self.assertEqual(len(llm.prompts), 2)
        self.assert_strict_top_k(lambda value: retriever.retrieve("revenue 200", top_k=value))

    def test_cross_encoder_reranks_with_a_fake_model_and_snapshots_scores(self) -> None:
        documents = [
            Document("weak evidence", metadata={"nested": {"owner": "source"}}, doc_id="weak", score=0.8),
            Document("best evidence", doc_id="best", score=0.2),
        ]
        model = _FakeCrossEncoder([0.1, 0.95])
        reranker = CrossEncoderReranker(model=model)

        result = reranker.rerank("question", documents, top_k=1)

        self.assertEqual(result[0].doc_id, "best")
        self.assertEqual(result[0].score, 0.95)
        self.assertEqual(result[0].metadata["first_stage_score"], 0.2)
        self.assertEqual(result[0].metadata["rerank_score"], 0.95)
        self.assertEqual(model.pairs, [("question", "weak evidence"), ("question", "best evidence")])
        self.assertEqual(documents[1].metadata, {})
        self.assert_strict_top_k(lambda value: reranker.rerank("question", documents, top_k=value))

    def test_reciprocal_rank_fusion_covers_reranker_and_multi_source_retriever(self) -> None:
        alpha = Document("alpha", doc_id="alpha", score=0.9)
        beta = Document("beta", doc_id="beta", score=0.8)
        gamma = Document("gamma", doc_id="gamma", score=0.7)
        fuser = ReciprocalRankFusionReranker(k=60)

        fused = fuser.fuse([[alpha, beta], [beta, gamma]], top_k=2)
        single_list = fuser.rerank("unused", [alpha, beta], top_k=1)

        self.assertEqual([doc.doc_id for doc in fused], ["beta", "alpha"])
        self.assertAlmostEqual(fused[0].score, 1 / 62 + 1 / 61)
        self.assertEqual(fused[0].metadata["first_stage_score"], 0.8)
        self.assertEqual(single_list[0].doc_id, "alpha")

        first = _RoutingRetriever(default=[alpha, beta])
        second = _RoutingRetriever(default=[beta, gamma])
        retriever = ReciprocalRankFusionRetriever(
            [first, second], candidate_top_k=4, k=60
        )
        result = retriever.retrieve("question", top_k=1)

        self.assertEqual(result[0].doc_id, "beta")
        self.assertEqual(result[0].metadata["retrieval_method"], "reciprocal-rank-fusion")
        self.assertEqual(result[0].metadata["rrf_sources"], 2)
        self.assertEqual(first.calls, [("question", 4)])
        self.assertEqual(second.calls, [("question", 4)])
        self.assert_strict_top_k(lambda value: fuser.fuse([[alpha]], top_k=value))
        self.assert_strict_top_k(lambda value: fuser.rerank("q", [alpha], top_k=value))
        self.assert_strict_top_k(lambda value: retriever.retrieve("q", top_k=value))

    def test_mmr_balances_relevance_and_diversity(self) -> None:
        documents = [
            Document("near query", doc_id="near"),
            Document("redundant", doc_id="redundant"),
            Document("diverse", doc_id="diverse"),
        ]
        embedding = _TableEmbedding(
            {
                "query": [1.0, 0.0],
                "near query": [0.9, 0.4358899],
                "redundant": [0.85, 0.5267827],
                "diverse": [0.5, -0.8660254],
            }
        )
        retriever = MMRRetriever(
            embedding, documents=documents, lambda_mult=0.5, fetch_k=3
        )

        result = retriever.retrieve("query", top_k=2)

        self.assertEqual([doc.doc_id for doc in result], ["near", "diverse"])
        self.assertEqual([doc.metadata["mmr_rank"] for doc in result], [0, 1])
        self.assertGreater(result[0].metadata["relevance_to_query"], result[1].metadata["relevance_to_query"])
        self.assert_strict_top_k(lambda value: retriever.retrieve("query", top_k=value))

    def test_hyde_retrieves_from_generated_hypothetical_answer(self) -> None:
        documents = [
            Document("Cats purr softly.", doc_id="cats"),
            Document("Dogs bark loudly.", doc_id="dogs"),
        ]
        llm = _QueueLLM(["Dogs bark loudly.", "Dogs bark loudly."])
        retriever = HyDERetriever(
            documents,
            _TokenEmbedding(["cats", "purr", "dogs", "bark"]),
            llm,
            n_hypotheses=2,
        )

        result = retriever.retrieve("Which animal makes this sound?", top_k=1)

        self.assertEqual(result[0].doc_id, "dogs")
        self.assertEqual(result[0].metadata["hypothetical_doc_preview"], "Dogs bark loudly.")
        self.assertEqual(len(llm.prompts), 2)
        self.assertTrue(all("Which animal" in prompt for prompt in llm.prompts))
        self.assert_strict_top_k(
            lambda value: retriever.retrieve("Which animal makes this sound?", top_k=value)
        )

    def test_hyqe_indexes_generated_questions_and_maps_hit_back_to_document(self) -> None:
        documents = [
            Document("Cats purr softly.", doc_id="cats"),
            Document("Dogs bark loudly.", doc_id="dogs"),
        ]
        llm = _QueueLLM(["How do cats purr?", "Which animals bark?"])
        retriever = HyQERetriever(
            documents,
            _TokenEmbedding(["cats", "purr", "animals", "bark"]),
            llm,
            n_questions_per_doc=1,
            include_original_content=False,
        )

        result = retriever.retrieve("Which animals bark?", top_k=1)

        self.assertEqual(result[0].doc_id, "dogs")
        self.assertEqual(result[0].metadata["hyqe_best_match"], "Which animals bark?")
        self.assertEqual(len(llm.prompts), 2)
        self.assert_strict_top_k(lambda value: retriever.retrieve("Which animals bark?", top_k=value))

    def test_rag_fusion_generates_variants_and_fuses_repeated_evidence(self) -> None:
        alpha = Document("alpha", doc_id="alpha", score=0.9)
        beta = Document("beta", doc_id="beta", score=0.8)
        gamma = Document("gamma", doc_id="gamma", score=0.7)
        base = _RoutingRetriever(
            {
                "original": [alpha, beta],
                "variant one": [beta, alpha],
                "variant two": [beta, gamma],
            }
        )
        llm = _QueueLLM(["variant one\nvariant two"])
        retriever = RAGFusionRetriever(
            base, llm, n_queries=2, rrf_k=60, per_query_top_k=2
        )

        result = retriever.retrieve("original", top_k=1)

        self.assertEqual(result[0].doc_id, "beta")
        self.assertGreater(result[0].score, 2 / 62)
        self.assertEqual(result[0].metadata["original_score"], 0.8)
        self.assertEqual(
            base.calls,
            [("original", 2), ("variant one", 2), ("variant two", 2)],
        )
        self.assert_strict_top_k(lambda value: retriever.retrieve("original", top_k=value))

    def test_self_query_applies_llm_structured_filters_before_vector_ranking(self) -> None:
        documents = [
            Document("Finance revenue report", metadata={"year": 2023, "team": "finance"}, doc_id="old"),
            Document("Finance revenue report", metadata={"year": 2025, "team": "finance"}, doc_id="current"),
            Document("Sales revenue report", metadata={"year": 2025, "team": "sales"}, doc_id="sales"),
        ]
        llm = _QueueLLM(
            [
                '{"cleaned_query": "revenue", "filters": '
                '{"year": {"$gte": 2024}, "team": "finance"}}'
            ]
        )
        retriever = SelfQueryRetriever(
            documents,
            _TokenEmbedding(["finance", "sales", "revenue"]),
            llm,
            metadata_schema={"year": "integer", "team": "string"},
        )

        result = retriever.retrieve("recent finance revenue", top_k=2)

        self.assertEqual([doc.doc_id for doc in result], ["current"])
        self.assertEqual(result[0].metadata["cleaned_query"], "revenue")
        self.assertEqual(result[0].metadata["applied_filters"]["year"], {"$gte": 2024})
        self.assertIn("- year : integer", llm.prompts[0])
        self.assert_strict_top_k(
            lambda value: retriever.retrieve("recent finance revenue", top_k=value)
        )

    def test_step_back_combines_specific_and_general_evidence_with_provenance(self) -> None:
        shared = Document("shared principle", doc_id="shared", score=0.4)
        specific = Document("specific exception", doc_id="specific", score=0.9)
        general = Document("general rule", doc_id="general", score=0.8)
        base = _RoutingRetriever(
            {
                "specific question": [specific, shared],
                "general question": [shared, general],
            }
        )
        llm = _QueueLLM(["general question"])
        retriever = StepBackRetriever(base, llm, n_original=2, n_stepback=2)

        result = retriever.retrieve("specific question", top_k=2)

        self.assertEqual([doc.doc_id for doc in result], ["shared", "specific"])
        self.assertEqual(result[0].metadata["retrieval_source"], "both")
        self.assertEqual(result[0].metadata["stepback_query"], "general question")
        self.assertEqual(
            base.calls,
            [("specific question", 2), ("general question", 2)],
        )
        self.assert_strict_top_k(
            lambda value: retriever.retrieve("specific question", top_k=value)
        )

    def test_query_decomposition_retrieves_each_subquestion_and_merges_best_hit(self) -> None:
        shared_low = Document("shared", doc_id="shared", score=0.2)
        shared_high = Document("shared", doc_id="shared", score=0.95)
        alpha = Document("alpha fact", doc_id="alpha", score=0.8)
        beta = Document("beta fact", doc_id="beta", score=0.7)
        base = _RoutingRetriever(
            {
                "complex question": [shared_low],
                "alpha question?": [alpha],
                "beta question?": [shared_high, beta],
            }
        )
        llm = _QueueLLM(["1. alpha question?\n2. beta question?"])
        retriever = QueryDecompositionRetriever(
            base, llm, max_subquestions=2, per_subquestion_top_k=2
        )

        result = retriever.retrieve("complex question", top_k=2)

        self.assertEqual([doc.doc_id for doc in result], ["shared", "alpha"])
        self.assertEqual(result[0].score, 0.95)
        self.assertEqual(
            result[0].metadata["matched_subquestions"],
            ["complex question", "beta question?"],
        )
        self.assertEqual(result[0].metadata["decomposed_from"], "complex question")
        self.assertEqual(len(base.calls), 3)
        self.assertTrue(all(call[1] == 2 for call in base.calls))
        self.assert_strict_top_k(
            lambda value: retriever.retrieve("complex question", top_k=value)
        )

    def test_chain_of_note_filters_irrelevant_context_and_returns_structured_note(self) -> None:
        irrelevant = Document("weather", doc_id="weather", score=0.9)
        relevant = Document("refund period is 14 days", doc_id="policy", score=0.7)
        base = _RoutingRetriever(default=[irrelevant, relevant])
        llm = _QueueLLM(
            [
                "PERTINENCE: non pertinent\nINFORMATION_CLE: aucune\nLIMITES: sujet différent",
                "PERTINENCE: directement pertinent\n"
                "INFORMATION_CLE: Le délai est de 14 jours.\nLIMITES: aucune",
            ]
        )
        retriever = ChainOfNoteRetriever(
            base, llm, drop_not_relevant=True, fetch_multiplier=2
        )

        result = retriever.retrieve("refund delay", top_k=1)

        self.assertEqual([doc.doc_id for doc in result], ["policy"])
        self.assertIn("Information clé : Le délai est de 14 jours.", result[0].content)
        self.assertEqual(result[0].metadata["con_pertinence"], "directement pertinent")
        self.assertEqual(result[0].metadata["original_content"], relevant.content)
        self.assertEqual(base.calls, [("refund delay", 2)])
        self.assert_strict_top_k(lambda value: retriever.retrieve("refund delay", top_k=value))

    def test_flare_alternates_draft_retrieval_and_grounded_generation(self) -> None:
        evidence = Document("The verified value is 42.", doc_id="fact-42", score=0.9)
        base = _RoutingRetriever(default=[evidence])
        llm = _QueueLLM(
            [
                "The answer probably contains the value 42.",
                "The verified value is 42. [source: fact-42]",
                "[DONE]",
            ]
        )
        pipeline = FLAREPipeline(
            base,
            llm,
            max_iterations=3,
            retrieval_top_k=2,
            min_draft_length=5,
        )

        result = pipeline.run("What is the verified value?")

        self.assertEqual(result["answer"], "The verified value is 42. [source: fact-42]")
        self.assertEqual(result["sources"][0]["doc_id"], "fact-42")
        self.assertEqual(len(result["iterations"]), 1)
        self.assertEqual(result["iterations"][0]["n_retrieved"], 1)
        self.assertEqual(len(base.calls), 1)
        self.assertIn("The answer probably contains the value 42.", base.calls[0][0])
        self.assertEqual(base.calls[0][1], 2)
        self.assertEqual(len(llm.prompts), 3)


if __name__ == "__main__":
    unittest.main()
