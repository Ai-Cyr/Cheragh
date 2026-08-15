import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from cheragh import Document, HashingEmbedding
from cheragh.base import BaseRetriever, EmbeddingModel, LLMClient
from cheragh.cache import (
    CachedEmbeddingModel,
    CachedLLMClient,
    CachedReranker,
    CachedRetriever,
    MemoryCache,
)
from cheragh.config.schema import validate_config
from cheragh.engine import _tokenizer_from_config
from cheragh.reranking import BaseReranker
from cheragh.llms import AzureOpenAIChatClient
from cheragh.vectorstores.faiss import FaissVectorStore
from cheragh.vectorstores.memory import MemoryVectorStore


class _DocumentRetriever(BaseRetriever):
    def __init__(self, documents):
        self.documents = list(documents)
        self.calls = 0

    def retrieve(self, query: str, top_k: int = 5):
        self.calls += 1
        return self.documents[:top_k]


class _OpaqueRetriever(BaseRetriever):
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.calls = 0

    def retrieve(self, query: str, top_k: int = 5):
        self.calls += 1
        return [Document(self.doc_id, doc_id=self.doc_id)][:top_k]


class _ModelLLM(LLMClient):
    def __init__(self, model: str, answer: str):
        self.model = model
        self.answer = answer
        self.calls = 0

    def generate(self, prompt: str, **kwargs):
        self.calls += 1
        return self.answer


class _ModelReranker(BaseReranker):
    def __init__(self, model: str, marker: str):
        self.model = model
        self.marker = marker
        self.calls = 0

    def rerank(self, query: str, documents, top_k: int = 5):
        self.calls += 1
        return [
            Document(
                document.content,
                metadata={**document.metadata, "marker": self.marker},
                doc_id=document.doc_id,
                score=document.score,
            )
            for document in documents[:top_k]
        ]


class _DirectionalEmbedding(EmbeddingModel):
    def __init__(self, reverse: bool):
        self.reverse = reverse
        self.client = object()

    def embed_documents(self, texts):
        import numpy as np

        rows = []
        for text in texts:
            first = text == "alpha"
            if self.reverse:
                first = not first
            rows.append([1.0, 0.0] if first else [0.0, 1.0])
        return np.asarray(rows, dtype=float)

    def embed_query(self, text):
        import numpy as np

        return np.asarray([1.0, 0.0], dtype=float)

    def get_fingerprint(self) -> str:
        # Simulates two provider clients that declare the same model while
        # pointing to opaque, behaviorally different deployments.
        return "provider::same-model"


class _FakeChatCompletions:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        message = SimpleNamespace(content=self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _azure_client(answer: str):
    completions = _FakeChatCompletions(answer)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


class ConfigConsistencyTests(unittest.TestCase):
    def test_retriever_and_vectorstore_are_each_validated(self):
        with self.assertRaises(ValidationError):
            validate_config(
                {
                    "retriever": {"type": "not-a-retriever"},
                    "vectorstore": {"type": "memory"},
                }
            )
        with self.assertRaises(ValidationError):
            validate_config(
                {
                    "retriever": {"type": "memory"},
                    "vectorstore": {"type": "not-a-store"},
                }
            )

    def test_explicit_retriever_vectorstore_conflict_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "conflict"):
            validate_config(
                {
                    "retriever": {"type": "hybrid"},
                    "vectorstore": {"type": "qdrant"},
                }
            )

        # ``vector`` remains a backward-compatible alias for the memory store.
        config = validate_config(
            {
                "retriever": {"type": "vector"},
                "vectorstore": {"type": "memory"},
            }
        )
        self.assertEqual(config.retriever.type, "vector")
        self.assertEqual(config.vectorstore.type, "memory")

        # A vectorstore can still be selected without explicitly overriding the
        # legacy retriever selector.
        self.assertEqual(validate_config({"vectorstore": {"type": "faiss"}}).vectorstore.type, "faiss")

    def test_component_names_and_provider_requirements_are_validated(self):
        invalid_configs = [
            {"embedding": {"provider": "unknown"}},
            {"generation": {"provider": "unknown"}},
            {"reranker": {"provider": "unknown"}},
            {"compression": {"type": "unknown"}},
            {"query": {"type": "unknown"}},
            {"embedding": {"provider": "azure_openai"}},
            {"generation": {"provider": "azure"}},
            {"generation": {"provider": "litellm"}},
        ]
        for raw in invalid_configs:
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                validate_config(raw)

        config = validate_config(
            {
                "embedding": {"provider": "azure_openai", "model": "embedding-deployment"},
                "generation": {"provider": "liteLLM", "model": "openai/gpt-4o-mini"},
                "reranker": {"provider": "cross_encoder"},
                "compression": {"type": "redundancy_filter"},
                "query": {"type": "step_back"},
            }
        )
        self.assertEqual(config.embedding.provider, "azure-openai")
        self.assertEqual(config.generation.provider, "litellm")
        self.assertEqual(config.reranker.provider, "cross-encoder")
        self.assertEqual(config.compression.type, "redundancy-filter")
        self.assertEqual(config.query.type, "step-back")

    def test_config_does_not_coerce_booleans_or_conflicting_aliases(self):
        invalid_configs = [
            {"retriever": {"top_k": True}},
            {"embedding": {"dimension": True}},
            {"indexing": {"lock_timeout_seconds": "10"}},
            {"cache": {"backend": "memory", "type": "redis"}},
            {"query": {"type": "multi-query", "transform": "step-back"}},
        ]
        for raw in invalid_configs:
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                validate_config(raw)

    def test_tokenizer_options_are_fully_validated_and_alias_is_preserved(self):
        config = validate_config(
            {
                "retriever": {
                    "tokenizer": {
                        "normalize_accents": False,
                        "ngram_range": [1, 3],
                        "min_token_length": 1,
                        "stopwords": ["a", "the"],
                    }
                }
            }
        )
        options = config.retriever.tokenizer
        self.assertNotIn("normalize_accents", options)
        self.assertFalse(options["strip_accents"])
        tokenizer = _tokenizer_from_config(options)
        self.assertEqual(tokenizer.ngram_range, (1, 3))
        self.assertEqual(tokenizer.stopwords, frozenset({"a", "the"}))

        invalid_options = [
            {"stopwords": None},
            {"stopwords": ["a"], "use_default_stopwords": False},
            {"stopwords": "a"},
            {"stopwords": ["a", 2]},
            {"ngram_range": [1]},
            {"ngram_range": [2, 1]},
            {"ngram_range": [True, 2]},
            {"min_token_length": 0},
            {"lowercase": "false"},
            {"normalize_accents": True, "strip_accents": False},
        ]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValidationError):
                validate_config({"retriever": {"tokenizer": options}})


class CacheFingerprintTests(unittest.TestCase):
    def test_retrieval_cache_is_isolated_by_corpus(self):
        cache = MemoryCache()
        first = _DocumentRetriever([Document("alpha", doc_id="a")])
        second = _DocumentRetriever([Document("beta", doc_id="b")])
        cached_first = CachedRetriever(first, cache)
        cached_second = CachedRetriever(second, cache)

        self.assertEqual(cached_first.retrieve("same query")[0].doc_id, "a")
        self.assertEqual(cached_second.retrieve("same query")[0].doc_id, "b")
        self.assertNotEqual(cached_first.fingerprint, cached_second.fingerprint)
        self.assertEqual((first.calls, second.calls), (1, 1))

    def test_equivalent_retrievers_share_a_stable_fingerprint(self):
        cache = MemoryCache()
        first_store = MemoryVectorStore(HashingEmbedding(8))
        second_store = MemoryVectorStore(HashingEmbedding(8))
        documents = [Document("alpha", metadata={"tenant": "acme"}, doc_id="a")]
        first_store.add_documents(documents)
        second_store.add_documents(documents)
        first = first_store.as_retriever()
        second = second_store.as_retriever()
        cached_first = CachedRetriever(first, cache)
        cached_second = CachedRetriever(second, cache)

        cached_first.retrieve("same query")
        cached_second.retrieve("same query")
        self.assertEqual(cached_first.fingerprint, cached_second.fingerprint)
        self.assertEqual(cache.stats().hits, 1)

    def test_opaque_retriever_fallback_prefers_isolation(self):
        cache = MemoryCache()
        first = CachedRetriever(_OpaqueRetriever("a"), cache)
        second = CachedRetriever(_OpaqueRetriever("b"), cache)

        self.assertEqual(first.retrieve("same query")[0].doc_id, "a")
        self.assertEqual(second.retrieve("same query")[0].doc_id, "b")
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_explicit_retriever_fingerprint_is_supported(self):
        cached = CachedRetriever(_OpaqueRetriever("a"), MemoryCache(), fingerprint="index-v7")
        self.assertEqual(cached.fingerprint, "index-v7")

    def test_llm_cache_key_includes_model(self):
        cache = MemoryCache()
        first = _ModelLLM("model-a", "answer-a")
        second = _ModelLLM("model-b", "answer-b")

        self.assertEqual(CachedLLMClient(first, cache).generate("prompt"), "answer-a")
        self.assertEqual(CachedLLMClient(second, cache).generate("prompt"), "answer-b")
        self.assertEqual((first.calls, second.calls), (1, 1))

    def test_opaque_provider_clients_do_not_share_cache_implicitly(self):
        cache = MemoryCache()
        first_client, first_calls = _azure_client("answer-a")
        second_client, second_calls = _azure_client("answer-b")
        first = AzureOpenAIChatClient(model="deployment", client=first_client)
        second = AzureOpenAIChatClient(model="deployment", client=second_client)

        self.assertEqual(CachedLLMClient(first, cache).generate("prompt"), "answer-a")
        self.assertEqual(CachedLLMClient(second, cache).generate("prompt"), "answer-b")
        self.assertEqual((first_calls.calls, second_calls.calls), (1, 1))

    def test_reranker_cache_key_includes_model(self):
        cache = MemoryCache()
        documents = [Document("alpha", doc_id="a")]
        first = _ModelReranker("model-a", "a")
        second = _ModelReranker("model-b", "b")

        first_result = CachedReranker(first, cache).rerank("query", documents)
        second_result = CachedReranker(second, cache).rerank("query", documents)
        self.assertEqual(first_result[0].metadata["marker"], "a")
        self.assertEqual(second_result[0].metadata["marker"], "b")
        self.assertEqual((first.calls, second.calls), (1, 1))

    def test_reranker_cache_key_includes_document_metadata(self):
        cache = MemoryCache()
        reranker = CachedReranker(_ModelReranker("model", "reranked"), cache)
        tenant_a = [Document("alpha", metadata={"tenant": "a"}, doc_id="same")]
        tenant_b = [Document("alpha", metadata={"tenant": "b"}, doc_id="same")]

        self.assertEqual(reranker.rerank("query", tenant_a)[0].metadata["tenant"], "a")
        self.assertEqual(reranker.rerank("query", tenant_b)[0].metadata["tenant"], "b")

    def test_reranker_bypasses_cache_for_incomplete_metadata_fingerprint(self):
        cache = MemoryCache()
        underlying = _ModelReranker("model", "reranked")
        reranker = CachedReranker(underlying, cache)
        first = [
            Document(
                "alpha",
                metadata={"updated_at": datetime(2024, 1, 1, tzinfo=timezone.utc)},
                doc_id="same",
            )
        ]
        second = [
            Document(
                "alpha",
                metadata={"updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc)},
                doc_id="same",
            )
        ]

        self.assertEqual(reranker.rerank("query", first)[0].metadata["updated_at"].year, 2024)
        self.assertEqual(reranker.rerank("query", second)[0].metadata["updated_at"].year, 2025)
        self.assertEqual(underlying.calls, 2)

    def test_cached_embedding_identity_propagates_to_retriever_fingerprint(self):
        cache = MemoryCache()
        documents = [Document("alpha", doc_id="a"), Document("beta", doc_id="b")]
        first_store = MemoryVectorStore(CachedEmbeddingModel(_DirectionalEmbedding(False), cache))
        second_store = MemoryVectorStore(CachedEmbeddingModel(_DirectionalEmbedding(True), cache))
        first_store.add_documents(documents)
        second_store.add_documents(documents)
        first = CachedRetriever(first_store.as_retriever(), cache)
        second = CachedRetriever(second_store.as_retriever(), cache)

        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.retrieve("query", top_k=1)[0].doc_id, "a")
        self.assertEqual(second.retrieve("query", top_k=1)[0].doc_id, "b")

    def test_cached_document_results_are_defensive_snapshots(self):
        cached = CachedRetriever(
            _DocumentRetriever([Document("alpha", metadata={"nested": {"value": 1}}, doc_id="a")]),
            MemoryCache(),
        )
        first = cached.retrieve("query")
        first[0].content = "mutated"
        first[0].metadata["nested"]["value"] = 2

        second = cached.retrieve("query")
        self.assertEqual(second[0].content, "alpha")
        self.assertEqual(second[0].metadata["nested"]["value"], 1)


class VectorStoreConsistencyTests(unittest.TestCase):
    def test_faiss_uses_the_common_metadata_filter_contract(self):
        store = FaissVectorStore(HashingEmbedding(8))
        store.documents = [
            Document("high", metadata={"quality": 0.95, "tags": ["legal"]}),
            Document("low", metadata={"quality": 0.5, "tags": ["other"]}),
        ]
        self.assertEqual(
            store._matching_indices({"quality": {"$gte": 0.9}, "tags": {"$contains": "legal"}}),
            [0],
        )

    def test_faiss_empty_filter_result_and_top_k_match_other_stores(self):
        store = FaissVectorStore(HashingEmbedding(8))
        store.documents = [Document("only", metadata={"quality": 0.2})]
        store.index = object()

        self.assertEqual(store.similarity_search("query", filters={"quality": {"$gte": 0.9}}), [])
        for invalid, exception in ((True, TypeError), (0, ValueError), (-1, ValueError)):
            with self.subTest(top_k=invalid), self.assertRaises(exception):
                store.similarity_search("query", top_k=invalid)

    def test_all_extra_satisfies_learned_retrieval_minimum(self):
        pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        all_section = pyproject.split("all = [", 1)[1].split("]", 1)[0]
        self.assertIn('"sentence-transformers>=5.0"', all_section)


if __name__ == "__main__":
    unittest.main()
