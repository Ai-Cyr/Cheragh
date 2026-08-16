from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from cheragh import BaseRetriever, Document, RAGEngine, StaticLLMClient
from cheragh.compression import ContextCompressor
from cheragh.config import load_config, validate_config
from cheragh.cli.main import main as cli_main
from cheragh.engine import _embedding_from_config, _llm_from_config
from cheragh.llms import OllamaClient


class _Retriever(BaseRetriever):
    def __init__(self, documents):
        self.documents = documents
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5):
        self.calls.append((query, top_k))
        return self.documents


class _EchoCitationLLM:
    def generate(self, prompt: str, **kwargs):
        citations = re.findall(r"\[source:\s*([^\]]+)\]", prompt)
        source_id = next(item for item in citations if item != "doc_id")
        return f"Réponse [source: {source_id}]"

    def stream(self, prompt: str, **kwargs):
        yield self.generate(prompt, **kwargs)


class _InfiniteTransformer:
    def __init__(self):
        self.yielded = 0

    def transform(self, query: str):
        index = 0
        while True:
            self.yielded += 1
            yield f"{query} {index}"
            index += 1


class _InfiniteDocuments:
    def __init__(self):
        self.yielded = 0

    def __iter__(self):
        index = 0
        while True:
            self.yielded += 1
            yield Document(f"document {index}", score=float(index))
            index += 1


class _OverReturningCompressor(ContextCompressor):
    def __init__(self):
        self.yielded = 0

    def compress(self, query, documents):
        def generate():
            index = 0
            while True:
                self.yielded += 1
                yield Document(f"compressed {index}", doc_id=f"compressed-{index}")
                index += 1

        return generate()


class RuntimeBoundaryTests(unittest.TestCase):
    def test_constructor_fails_fast_on_invalid_collaborators_and_options(self):
        retriever = _Retriever([])
        with self.assertRaises(TypeError):
            RAGEngine(object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RAGEngine(retriever, llm_client=object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RAGEngine(retriever, answer_prompt=3)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            RAGEngine(retriever, answer_prompt="  ")
        with self.assertRaises(ValueError):
            RAGEngine(retriever, answer_prompt="{unknown}")
        with self.assertRaises(ValueError):
            RAGEngine(retriever, answer_prompt="{context.__class__}")
        with self.assertRaises(TypeError):
            RAGEngine(retriever, strict_grounding=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RAGEngine(retriever, min_score=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            RAGEngine(retriever, min_score=math.nan)

    def test_query_validation_is_shared_by_sync_and_stream(self):
        engine = RAGEngine(_Retriever([]), strict_grounding=True)
        for value, error in ((None, TypeError), (3, TypeError), ("", ValueError), (" \n", ValueError)):
            with self.subTest(value=value), self.assertRaises(error):
                engine.ask(value)  # type: ignore[arg-type]
            with self.subTest(stream_value=value), self.assertRaises(error):
                engine.stream_with_response(value)  # type: ignore[arg-type]

    def test_infinite_query_transform_is_deduplicated_and_strictly_capped(self):
        transformer = _InfiniteTransformer()
        retriever = _Retriever([])
        engine = RAGEngine(
            retriever,
            query_transformer=transformer,
            strict_grounding=True,
        )

        response = engine.ask("question", top_k=1)

        self.assertEqual(transformer.yielded, 33)
        self.assertEqual(len(retriever.calls), 32)
        self.assertIn("query_variants_capped:32", response.trace.warnings)

    def test_retriever_over_return_is_capped_snapshotted_and_citable(self):
        documents = _InfiniteDocuments()
        engine = RAGEngine(
            _Retriever(documents),
            llm_client=_EchoCitationLLM(),  # type: ignore[arg-type]
            require_citations=True,
        )

        response = engine.ask("question", top_k=2)

        self.assertEqual(documents.yielded, 2)
        self.assertEqual(len(response.retrieved_documents), 2)
        self.assertTrue(response.retrieved_documents[0].doc_id.startswith("rag-anonymous-"))
        self.assertEqual(response.citations, [response.retrieved_documents[0].doc_id])
        self.assertNotIn("unknown_citations", response.warnings)

        response.retrieved_documents[0].metadata["caller"] = True
        self.assertNotIn("caller", next(iter(documents)).metadata)

    def test_synthetic_document_id_is_stable(self):
        source = Document("same anonymous evidence", doc_id="   ")
        expected = "rag-anonymous-" + hashlib.sha256(source.content.encode()).hexdigest()[:20]
        engine = RAGEngine(_Retriever([source]), llm_client=_EchoCitationLLM())  # type: ignore[arg-type]

        first = engine.ask("q", top_k=1)
        second = engine.ask("q", top_k=1)

        self.assertEqual(first.retrieved_documents[0].doc_id, expected)
        self.assertEqual(second.retrieved_documents[0].doc_id, expected)
        self.assertEqual(source.doc_id, "   ")

    def test_malformed_retriever_documents_fail_closed(self):
        invalid = [
            "not-documents",
            [object()],
            [Document("", doc_id="empty")],
            [Document("content", metadata=None)],  # type: ignore[arg-type]
            [Document("content", doc_id=3)],  # type: ignore[arg-type]
            [Document("content", score=True)],
            [Document("content", score=math.nan)],
        ]
        for result in invalid:
            with self.subTest(result=result), self.assertRaises((TypeError, ValueError)):
                RAGEngine(_Retriever(result), strict_grounding=True).ask("q")

    def test_compressor_output_is_capped(self):
        compressor = _OverReturningCompressor()
        engine = RAGEngine(
            _Retriever([Document("source", doc_id="source")]),
            llm_client=StaticLLMClient("answer"),
            compressor=compressor,
        )

        response = engine.ask("q", top_k=3)

        self.assertEqual(compressor.yielded, 3)
        self.assertEqual(len(response.retrieved_documents), 3)

    def test_generation_type_errors_export_a_safe_failure_trace(self):
        class BadLLM:
            def generate(self, prompt: str, **kwargs):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            engine = RAGEngine(
                _Retriever([Document("evidence", doc_id="evidence")]),
                llm_client=BadLLM(),  # type: ignore[arg-type]
                trace_export_path=trace_path,
            )

            with self.assertRaisesRegex(TypeError, "must return a str"):
                engine.ask("q")

            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertIn("generation_error", trace["warnings"])
            self.assertNotIn("prompt", trace)

    def test_stream_rejects_non_text_chunks_and_exports_failure_trace(self):
        class BadStreamLLM:
            def generate(self, prompt: str, **kwargs):
                return "unused"

            def stream(self, prompt: str, **kwargs):
                yield "first"
                yield 2

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            engine = RAGEngine(
                _Retriever([Document("evidence", doc_id="evidence")]),
                llm_client=BadStreamLLM(),  # type: ignore[arg-type]
                trace_export_path=trace_path,
            )

            with self.assertRaisesRegex(TypeError, "yield only str"):
                list(engine.stream("q"))

            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertIn("stream_generation_error", trace["warnings"])

    def test_trace_export_failure_is_best_effort_and_redacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = RAGEngine(
                _Retriever([Document("evidence", doc_id="evidence")]),
                llm_client=StaticLLMClient("answer"),
                trace_export_path=tmp,
            )

            with self.assertLogs("cheragh.engine", level="ERROR") as logs:
                response = engine.ask("private question")

            self.assertEqual(response.answer, "answer")
            self.assertIn("trace_export_error", response.trace.warnings)
            rendered_logs = "\n".join(logs.output)
            self.assertNotIn("private question", rendered_logs)
            self.assertNotIn(tmp, rendered_logs)


class ProductionConfigTests(unittest.TestCase):
    def test_exact_environment_references_resolve_and_missing_secrets_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "generation": {
                            "provider": "openai",
                            "api_key": "${CHERAGH_TEST_API_KEY}",
                            "base_url": "prefix-${CHERAGH_TEST_API_KEY}",
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CHERAGH_TEST_API_KEY": "secret-value"}, clear=False):
                config = load_config(path)
            self.assertEqual(config["generation"]["api_key"], "secret-value")
            self.assertEqual(
                config["generation"]["base_url"],
                "prefix-${CHERAGH_TEST_API_KEY}",
            )

            path.write_text(
                json.dumps({"generation": {"api_key": "${CHERAGH_MISSING_SECRET}"}}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
                ValueError,
                "CHERAGH_MISSING_SECRET",
            ):
                load_config(path)

            with patch.dict(
                os.environ,
                {"CHERAGH_MISSING_SECRET": "   "},
                clear=True,
            ), self.assertRaisesRegex(ValueError, "not set or is empty"):
                load_config(path)

    def test_generation_timeout_and_retry_configuration_is_strict(self):
        config = validate_config({"generation": {"timeout_seconds": 12.5, "max_retries": 4}})
        self.assertEqual(config.generation.timeout_seconds, 12.5)
        self.assertEqual(config.generation.max_retries, 4)
        for payload in (
            {"timeout_seconds": True},
            {"timeout_seconds": 0.0},
            {"max_retries": True},
            {"max_retries": -1},
            {"max_retries": 11},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_config({"generation": payload})

        bounded_cache = validate_config({"cache": {"enabled": True, "max_entries": 250}})
        self.assertEqual(bounded_cache.cache.max_entries, 250)
        self.assertEqual(validate_config({}).cache.max_entries, 10_000)
        for invalid in (True, 0, -1):
            with self.subTest(max_entries=invalid), self.assertRaises(ValueError):
                validate_config({"cache": {"enabled": True, "max_entries": invalid}})

        embedding = validate_config(
            {"embedding": {"provider": "openai", "timeout_seconds": 8.0, "max_retries": 1}}
        ).embedding
        self.assertEqual(embedding.timeout_seconds, 8.0)
        self.assertEqual(embedding.max_retries, 1)
        for payload in (
            {"timeout_seconds": True},
            {"timeout_seconds": 0.0},
            {"max_retries": True},
            {"max_retries": 11},
            {"provider": "cohere", "max_retries": 1},
        ):
            with self.subTest(embedding=payload), self.assertRaises(ValueError):
                validate_config({"embedding": payload})

    def test_validate_config_cli_checks_environment_and_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "generation": {
                            "provider": "openai",
                            "api_key": "${CHERAGH_CLI_API_KEY}",
                        },
                        "cache": {
                            "enabled": True,
                            "secret_key": "${CHERAGH_CLI_HMAC_KEY}",
                            "hmac_key": "${CHERAGH_CLI_CREDENTIAL}",
                        },
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "CHERAGH_CLI_API_KEY": "provider-secret",
                    "CHERAGH_CLI_HMAC_KEY": "cache-secret",
                    "CHERAGH_CLI_CREDENTIAL": "opaque-credential",
                },
                clear=False,
            ), redirect_stdout(stdout):
                code = cli_main(["validate-config", str(path), "--json"])

            rendered = stdout.getvalue()
            self.assertEqual(code, 0)
            self.assertNotIn("provider-secret", rendered)
            self.assertNotIn("cache-secret", rendered)
            self.assertNotIn("opaque-credential", rendered)
            self.assertGreaterEqual(rendered.count('"***"'), 3)

            stderr = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), redirect_stderr(stderr):
                code = cli_main(["validate-config", str(path)])
            self.assertEqual(code, 1)
            self.assertIn("CHERAGH_CLI_API_KEY", stderr.getvalue())

    def test_serve_cli_forwards_production_limits(self):
        with patch("cheragh.server.main.serve") as serve:
            code = cli_main(
                [
                    "serve",
                    "--index",
                    "index",
                    "--port",
                    "9000",
                    "--require-auth",
                    "--max-top-k",
                    "20",
                    "--max-request-body-bytes",
                    "2048",
                    "--max-concurrent-operations",
                    "4",
                    "--max-server-connections",
                    "64",
                    "--request-timeout-seconds",
                    "12.5",
                    "--index-timeout-seconds",
                    "120",
                    "--stream-max-duration-seconds",
                    "45",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(serve.call_args.kwargs["port"], 9000)
        self.assertTrue(serve.call_args.kwargs["require_auth"])
        self.assertEqual(serve.call_args.kwargs["max_top_k"], 20)
        self.assertEqual(serve.call_args.kwargs["max_request_body_bytes"], 2048)
        self.assertEqual(serve.call_args.kwargs["max_concurrent_operations"], 4)
        self.assertEqual(serve.call_args.kwargs["max_server_connections"], 64)
        self.assertEqual(serve.call_args.kwargs["request_timeout_seconds"], 12.5)
        self.assertEqual(serve.call_args.kwargs["index_timeout_seconds"], 120.0)
        self.assertEqual(serve.call_args.kwargs["stream_max_duration_seconds"], 45.0)

    def test_openai_factory_forwards_bounded_network_options(self):
        captured = {}
        module = ModuleType("openai")

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        module.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"openai": module}):
            client = _llm_from_config(
                {
                    "provider": "openai",
                    "model": "model",
                    "api_key": "key",
                    "base_url": "https://example.invalid/v1",
                    "timeout_seconds": 12.0,
                    "max_retries": 3,
                }
            )

        self.assertEqual(client.model, "model")
        self.assertEqual(captured["timeout"], 12.0)
        self.assertEqual(captured["max_retries"], 3)
        self.assertEqual(captured["base_url"], "https://example.invalid/v1")

        with patch("cheragh.embeddings.OpenAIEmbedding") as embedding_client:
            _embedding_from_config(
                {
                    "provider": "openai",
                    "model": "embedding-model",
                    "api_key": "key",
                    "timeout_seconds": 7.0,
                    "max_retries": 1,
                }
            )
        embedding_client.assert_called_once_with(
            model="embedding-model",
            api_key="key",
            timeout=7.0,
            max_retries=1,
        )

    def test_ollama_timeout_is_not_leaked_into_request_payload(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"response":"ok"}'

        client = OllamaClient(timeout_seconds=9.0)
        with patch("cheragh.llms.request.urlopen", return_value=Response()) as urlopen:
            answer = client.generate("prompt", timeout=2.5, seed=7)

        request_object = urlopen.call_args.args[0]
        payload = json.loads(request_object.data.decode("utf-8"))
        self.assertEqual(answer, "ok")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 2.5)
        self.assertNotIn("timeout", payload)
        self.assertEqual(payload["seed"], 7)

        for timeout in (True, 0, math.nan):
            with self.subTest(timeout=timeout), self.assertRaises((TypeError, ValueError)):
                client.generate("prompt", timeout=timeout)


if __name__ == "__main__":
    unittest.main()
