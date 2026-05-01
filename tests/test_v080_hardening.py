import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from cheragh import Document, SQLRAGEngine
from cheragh.cache import SQLiteCache


class V080HardeningTests(unittest.TestCase):
    def test_sqlite_cache_defaults_to_json_and_roundtrips_safe_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.sqlite"
            cache = SQLiteCache(path)
            vector = [1.0, 2.0, 3.0]
            docs = [Document("Alpha", metadata={"tenant": "a"}, doc_id="d1", score=0.9)]
            cache.set("vector", vector)
            cache.set("docs", docs)
            cache.close()

            reopened = SQLiteCache(path)
            self.assertEqual([round(float(x), 4) for x in reopened.get("vector")], [1.0, 2.0, 3.0])
            self.assertEqual(reopened.get("docs")[0].doc_id, "d1")
            self.assertEqual(reopened.get("docs")[0].metadata["tenant"], "a")
            reopened.close()

    def test_persistent_pickle_cache_must_be_signed_or_explicitly_unsafe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.sqlite"
            with self.assertRaises(ValueError):
                SQLiteCache(path, serializer="pickle", allow_pickle=True)
            signed = SQLiteCache(path, serializer="signed-pickle", secret_key="secret", allow_pickle=True)
            signed.set("k", {"v": 1})
            self.assertEqual(signed.get("k"), {"v": 1})
            signed.close()

    def test_signed_cache_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.sqlite"
            cache = SQLiteCache(path, serializer="signed-pickle", secret_key="secret", allow_pickle=True)
            cache.set("k", {"v": 1})
            cache.close()
            conn = sqlite3.connect(path)
            payload = conn.execute("SELECT payload FROM cache_entries").fetchone()[0]
            tampered = bytearray(payload)
            tampered[-1] = tampered[-1] ^ 1
            conn.execute("UPDATE cache_entries SET payload=?", (bytes(tampered),))
            conn.commit()
            conn.close()

            reopened = SQLiteCache(path, serializer="signed-pickle", secret_key="secret", allow_pickle=True)
            self.assertIsNone(reopened.get("k"))
            self.assertGreaterEqual(reopened.stats().errors, 1)

    def test_sql_rag_is_query_only_after_materialization(self):
        engine = SQLRAGEngine.from_records("sales", [{"client": "Alpha", "revenue": 100}])
        self.assertTrue(engine.read_only)
        self.assertIn("100", engine.ask("total revenue").answer)
        with self.assertRaises(RuntimeError):
            engine.add_table("other", [{"x": 1}])
        with self.assertRaises(ValueError):
            engine.execute_sql("PRAGMA table_info(sales)")
        with self.assertRaises(sqlite3.DatabaseError):
            engine.connection.execute("INSERT INTO sales(client, revenue) VALUES ('Beta', 2)")

    def test_sqlite_file_database_opens_real_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "data.sqlite"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE sales(client TEXT, revenue INTEGER)")
            conn.execute("INSERT INTO sales VALUES ('Alpha', 100)")
            conn.commit()
            conn.close()

            engine = SQLRAGEngine(database=db_path, table_allowlist=["sales"], read_only=True)
            self.assertEqual(engine.execute_sql("SELECT SUM(revenue) AS total FROM sales").rows[0]["total"], 100)
            with self.assertRaises(sqlite3.OperationalError):
                engine.connection.execute("INSERT INTO sales VALUES ('Beta', 1)")

    def test_pydantic_config_validation(self):
        from cheragh.config.schema import validate_config

        good = validate_config(
            {
                "ingestion": {"path": "./docs", "chunk_size": 100, "chunk_overlap": 10},
                "cache": {"enabled": True, "backend": "sqlite", "path": "cache.sqlite", "serializer": "json"},
            }
        )
        self.assertEqual(good.ingestion.chunk_size, 100)
        with self.assertRaises(ValidationError):
            validate_config({"ingestion": {"path": "./docs", "chunk_size": 10, "chunk_overlap": 10}})
        with self.assertRaises(ValidationError):
            validate_config(
                {"cache": {"enabled": True, "backend": "sqlite", "serializer": "pickle", "allow_pickle": True}}
            )
        with self.assertRaises(ValidationError):
            validate_config({"unknown": True})

    def test_load_config_returns_normalized_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rag.json"
            path.write_text(json.dumps({"ingestion": {"path": "./docs"}}), encoding="utf-8")
            from cheragh.config.loader import load_config

            data = load_config(path)
            self.assertIn("ingestion", data)
            self.assertEqual(data["embedding"]["provider"], "hashing")

    def test_docker_files_are_pinned(self):
        root = Path(__file__).resolve().parents[1]
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
        compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
        constraints = (root / "docker" / "constraints.txt").read_text(encoding="utf-8")
        self.assertIn("python:3.12.7-slim-bookworm", dockerfile)
        self.assertNotIn(":latest", compose)
        self.assertIn("pip==24.3.1", dockerfile)
        self.assertIn("fastapi==", constraints)


if __name__ == "__main__":
    unittest.main()
