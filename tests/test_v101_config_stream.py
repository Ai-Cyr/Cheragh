import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from cheragh import Document, RAGEngine, StaticLLMClient
from cheragh.cli.main import main
from cheragh.config import load_config


class FixedRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=5):
        return self.documents[:top_k]


class ChunkedLLM(StaticLLMClient):
    def stream(self, prompt, **kwargs):
        self.prompts.append(prompt)
        yield "first"
        yield " second"


class V101ConfigAndStreamTests(unittest.TestCase):
    def test_production_preset_tokenizer_is_valid(self):
        config = load_config("examples/presets/production_v100.yaml")
        self.assertTrue(config["retriever"]["tokenizer"]["strip_accents"])

    def test_invalid_vectorstore_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"vectorstore": {"type": "garbage"}}), encoding="utf-8")
            with self.assertRaises(Exception):
                load_config(path)

    def test_stream_enforces_min_score_and_strict_grounding(self):
        retriever = FixedRetriever([Document("irrelevant", doc_id="d1", score=0.1)])
        llm = StaticLLMClient("must not be generated")
        engine = RAGEngine(retriever, llm_client=llm, min_score=0.5, strict_grounding=True)
        self.assertEqual(
            "".join(engine.stream("question")),
            "Je ne sais pas : aucun extrait suffisamment pertinent n'a été trouvé.",
        )
        self.assertEqual(llm.prompts, [])

    def test_stream_with_response_exposes_citations_metadata_and_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "stream-traces.jsonl"
            retriever = FixedRetriever([Document("grounded context", doc_id="d1", score=0.9)])
            engine = RAGEngine(
                retriever,
                llm_client=StaticLLMClient("Grounded answer. [source: d1]"),
                require_citations=True,
                trace_export_path=trace_path,
            )

            stream = engine.stream_with_response("question", top_k=1)
            self.assertIsNone(stream.response)
            self.assertEqual("".join(stream), "Grounded answer. [source: d1]")

            response = stream.response
            self.assertIsNotNone(response)
            assert response is not None
            self.assertEqual(response.answer, "Grounded answer. [source: d1]")
            self.assertEqual(response.citations, ["d1"])
            self.assertEqual(response.metadata["top_k"], 1)
            self.assertEqual(response.sources[0].doc_id, "d1")
            self.assertTrue(response.citation_validation.ok)
            self.assertIsNotNone(response.trace)
            self.assertEqual([step.name for step in response.trace.steps], ["retrieval", "generation"])
            self.assertTrue(trace_path.exists())
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["query"], "question")
            self.assertEqual(payload["retrieval"][0]["documents"][0]["doc_id"], "d1")
            self.assertGreater(payload["token_usage"]["output_tokens_estimated"], 0)

    def test_closing_stream_exports_partial_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "cancelled-stream.jsonl"
            engine = RAGEngine(
                FixedRetriever([Document("context", doc_id="d1", score=0.9)]),
                llm_client=ChunkedLLM(),
                trace_export_path=trace_path,
            )

            stream = engine.stream_with_response("question")
            self.assertEqual(next(stream), "first")
            stream.close()

            self.assertIsNone(stream.response)
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertIn("stream_cancelled", payload["warnings"])
            self.assertEqual(payload["steps"][-1]["metadata"]["answer_chars"], 5)
            self.assertTrue(payload["steps"][-1]["metadata"]["cancelled"])

    def test_cli_config_does_not_override_top_k_unless_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "a.txt").write_text("alpha", encoding="utf-8")
            config = Path(tmp) / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "ingestion": {"path": str(docs)},
                        "retriever": {"type": "memory", "top_k": 1},
                        "generation": {"provider": "extractive"},
                    }
                ),
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                code = main(["ask", "alpha", "--config", str(config), "--json"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["metadata"]["top_k"], 1)


if __name__ == "__main__":
    unittest.main()
