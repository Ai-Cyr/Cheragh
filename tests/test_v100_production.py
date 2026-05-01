import json
import tempfile
import unittest
from pathlib import Path

from cheragh import (
    Chunk,
    Document,
    HashingEmbedding,
    RAGEngine,
    RAGResponse,
    RetrieverProtocol,
    Source,
    StaticLLMClient,
    index_path,
    inspect_index,
    load_manifest,
)


class V100ProductionBaselineTests(unittest.TestCase):
    def test_public_schema_and_protocol_exports(self):
        doc = Document("hello", metadata={"parent_doc_id": "parent", "source_char_start": 1, "source_char_end": 5}, doc_id="c1")
        chunk = Chunk.from_document(doc)
        self.assertEqual(chunk.parent_doc_id, "parent")
        response = RAGResponse(query="q", answer="a", sources=[Source("c1", 1.0, "hello")], retrieved_documents=[doc], prompt="p")
        self.assertEqual(response.to_dict()["sources"][0]["doc_id"], "c1")
        self.assertTrue(isinstance(type("R", (), {"retrieve": lambda self, query, top_k=5: []})(), RetrieverProtocol))

    def test_trace_export_jsonl_includes_steps_and_token_estimates(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "traces.jsonl"
            docs = [Document("SQLite read-only protège les données.", doc_id="sqlite")]
            engine = RAGEngine.from_documents(
                docs,
                embedding_model=HashingEmbedding(dimension=32),
                llm_client=StaticLLMClient("SQLite est protégé. [source: sqlite]"),
                trace_export_path=trace_path,
                trace_pricing={"input_per_1k": 0.001, "output_per_1k": 0.002},
            )
            response = engine.ask("Comment SQLite est-il protégé ?")
            self.assertIsNotNone(response.trace)
            payload = json.loads(trace_path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["query"], "Comment SQLite est-il protégé ?")
            self.assertGreater(payload["duration_ms"], 0)
            self.assertIn("total_tokens_estimated", payload["token_usage"])
            self.assertIn("total_cost_estimated", payload["cost"])

    def test_incremental_index_dry_run_and_deleted_file_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "docs"
            out = Path(tmp) / "index"
            root.mkdir()
            a = root / "a.txt"
            b = root / "b.txt"
            a.write_text("alpha security", encoding="utf-8")
            b.write_text("beta cache", encoding="utf-8")

            first = index_path(root, out, embedding_model=HashingEmbedding(dimension=32), chunk_size=20, chunk_overlap=2)
            self.assertEqual(first["changed_files"], 2)
            manifest = load_manifest(out)
            self.assertEqual(manifest.schema_version, 3)
            self.assertEqual(len(manifest.files), 2)

            a.write_text("alpha security updated", encoding="utf-8")
            b.unlink()
            dry = index_path(root, out, embedding_model=HashingEmbedding(dimension=32), chunk_size=20, chunk_overlap=2, dry_run=True)
            self.assertTrue(dry["dry_run"])
            self.assertEqual(dry["plan"]["changed_count"], 1)
            self.assertEqual(dry["plan"]["deleted_count"], 1)

            second = index_path(root, out, embedding_model=HashingEmbedding(dimension=32), chunk_size=20, chunk_overlap=2)
            self.assertEqual(second["deleted_files"], 1)
            info = inspect_index(out)
            self.assertEqual(info["files"], 1)
            self.assertGreaterEqual(info["documents"], 1)


if __name__ == "__main__":
    unittest.main()
