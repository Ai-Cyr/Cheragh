import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cheragh import Document, HashingEmbedding, RAGEngine, StaticLLMClient
from cheragh.cache import CachedLLMClient, CachedRetriever, MemoryCache
from cheragh.query import MultiQueryTransformer
from cheragh.engine import _llm_from_config


class APIFactoryConsistencyTests(unittest.TestCase):
    def test_cache_backend_provider_name_is_resolved_eagerly(self):
        engine = RAGEngine.from_documents(
            [Document("alpha", doc_id="a")],
            embedding_model=HashingEmbedding(32),
            llm_client=StaticLLMClient("answer [source: a]"),
            retriever_type="memory",
            cache_backend="memory",
        )

        self.assertIsInstance(engine.cache_backend, MemoryCache)
        self.assertIsInstance(engine.retriever, CachedRetriever)
        self.assertEqual(engine.ask("alpha").answer, "answer [source: a]")

    def test_cache_namespace_is_applied_to_every_layer(self):
        cache = MemoryCache()
        engine = RAGEngine.from_documents(
            [Document("alpha", doc_id="a")],
            embedding_model=HashingEmbedding(32),
            llm_client=StaticLLMClient("answer [source: a]"),
            retriever_type="memory",
            cache_backend=cache,
            cache_config={"enabled": True, "namespace": "tenant-a"},
        )

        self.assertEqual(engine.retriever.namespace, "tenant-a:retrieval")
        self.assertEqual(
            engine.retriever.retriever.store.embedding_model.namespace,
            "tenant-a:embeddings",
        )
        self.assertIsInstance(engine.llm_client, CachedLLMClient)
        self.assertEqual(engine.llm_client.namespace, "tenant-a:llm")

    def test_named_query_transformer_receives_the_configured_llm(self):
        llm = StaticLLMClient("variante une\nvariante deux")
        engine = RAGEngine.from_documents(
            [Document("alpha", doc_id="a")],
            embedding_model=HashingEmbedding(32),
            llm_client=llm,
            retriever_type="memory",
            query_transformer="multi-query",
        )

        self.assertIsInstance(engine.query_transformer, MultiQueryTransformer)
        self.assertIs(engine.query_transformer.llm_client, llm)
        self.assertEqual(
            engine.query_transformer.transform("question"),
            ["question", "variante une", "variante deux"],
        )

        direct = RAGEngine(
            engine.retriever,
            llm_client=llm,
            query_transformer="multi-query",
        )
        self.assertIs(direct.query_transformer.llm_client, llm)

    def test_invalid_cache_backend_fails_during_construction(self):
        with self.assertRaisesRegex(TypeError, "cache_backend"):
            RAGEngine.from_documents(
                [Document("alpha", doc_id="a")],
                embedding_model=HashingEmbedding(32),
                retriever_type="memory",
                cache_backend=object(),
            )

    def test_unknown_factory_keyword_fails_instead_of_being_ignored(self):
        with self.assertRaisesRegex(TypeError, "strict_groundng"):
            RAGEngine.from_documents(
                [Document("alpha", doc_id="a")],
                embedding_model=HashingEmbedding(32),
                strict_groundng=True,
            )

    def test_config_files_resolve_local_paths_from_their_own_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "project"
            corpus = config_dir / "private-corpus"
            corpus.mkdir(parents=True)
            (corpus / "knowledge.txt").write_text("configuration locale", encoding="utf-8")
            config_path = config_dir / "rag.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ingestion": {"path": "private-corpus"},
                        "embedding": {"provider": "hashing", "dimension": 32},
                        "retriever": {"type": "memory", "top_k": 1},
                        "vectorstore": {"type": "memory", "path": "index"},
                        "generation": {"provider": "extractive"},
                        "cache": {"enabled": True, "backend": "sqlite", "path": "cache.sqlite"},
                        "observability": {"enabled": True, "trace_export_path": "traces.jsonl"},
                    }
                ),
                encoding="utf-8",
            )

            engine = RAGEngine.from_config(config_path)
            response = engine.ask("configuration")

            self.assertEqual(response.retrieved_documents[0].metadata["filename"], "knowledge.txt")
            self.assertTrue((config_dir / "index" / "manifest.json").is_file())
            self.assertTrue((config_dir / "cache.sqlite").is_file())
            self.assertTrue((config_dir / "traces.jsonl").is_file())

    def test_openai_base_url_is_not_silently_ignored(self):
        with patch("cheragh.engine.OpenAILLMClient") as client:
            _llm_from_config(
                {
                    "provider": "openai",
                    "model": "compatible-model",
                    "api_key": "test-key",
                    "base_url": "https://provider.example/v1",
                }
            )

        client.assert_called_once_with(
            model="compatible-model",
            api_key="test-key",
            base_url="https://provider.example/v1",
        )


if __name__ == "__main__":
    unittest.main()
