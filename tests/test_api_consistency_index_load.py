import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cheragh import Document, EmbeddingModel, HashingEmbedding, MemoryVectorStore
from cheragh.cli.main import main


class _CustomEmbedding(EmbeddingModel):
    dimension = 7

    def embed_documents(self, texts: list[str]):
        return np.vstack([self.embed_query(text) for text in texts]) if texts else np.zeros((0, self.dimension))

    def embed_query(self, text: str):
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[len(text) % self.dimension] = 1.0
        return vector

    def get_fingerprint(self) -> str:
        return "tests.CustomEmbedding::v1"


class IndexLoadContractTests(unittest.TestCase):
    def _save_hashing_store(self, path: Path, *, dimension: int = 23, ngram_range=(1, 3)) -> None:
        store = MemoryVectorStore(HashingEmbedding(dimension=dimension, ngram_range=ngram_range))
        store.add_documents([Document("manifest driven retrieval", doc_id="doc-1")])
        store.save(path)

    def test_load_without_model_reconstructs_non_default_hashing_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index"
            self._save_hashing_store(index)

            loaded = MemoryVectorStore.load(index)

            self.assertIsInstance(loaded.embedding_model, HashingEmbedding)
            self.assertEqual(loaded.embedding_model.dimension, 23)
            self.assertEqual(loaded.embedding_model.ngram_range, (1, 3))
            self.assertEqual(loaded.similarity_search("retrieval", top_k=1)[0].doc_id, "doc-1")

    def test_load_rejects_wrong_embedding_fingerprint_before_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index"
            self._save_hashing_store(index, dimension=23)

            with self.assertRaisesRegex(ValueError, "Embedding model mismatch"):
                MemoryVectorStore.load(index, HashingEmbedding(dimension=384))

    def test_load_rejects_manifest_and_stored_dimension_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index"
            self._save_hashing_store(index, dimension=23)
            vectors = np.load(index / "embeddings.npy", allow_pickle=False)
            with (index / "embeddings.npy").open("wb") as file:
                np.save(file, vectors[:, :11], allow_pickle=False)

            with self.assertRaisesRegex(ValueError, "manifest embedding dimension 23 != stored dimension 11"):
                MemoryVectorStore.load(index)

    def test_automatic_loading_refuses_custom_provider_but_explicit_model_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index"
            custom = _CustomEmbedding()
            store = MemoryVectorStore(custom)
            store.add_documents([Document("custom vector", doc_id="custom")])
            store.save(index)

            with self.assertRaisesRegex(ValueError, "only supports.*provider='hashing'"):
                MemoryVectorStore.load(index)
            loaded = MemoryVectorStore.load(index, custom)
            self.assertEqual(loaded.documents[0].doc_id, "custom")

    def test_legacy_hashing_manifest_can_still_be_loaded_automatically(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index"
            self._save_hashing_store(index, dimension=19, ngram_range=(2, 2))
            manifest_path = index / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 1
            manifest.pop("dimension")
            manifest.pop("embedding")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            loaded = MemoryVectorStore.load(index)

            self.assertEqual(loaded.embedding_model.dimension, 19)
            self.assertEqual(loaded.embedding_model.ngram_range, (2, 2))

    def test_manifest_is_validated_before_data_files_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index"
            index.mkdir()
            (index / "manifest.json").write_text('{"schema_version": 2}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "count must be an integer"):
                MemoryVectorStore.load(index)

    def test_cli_ask_derives_hashing_dimension_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index"
            self._save_hashing_store(index, dimension=29)
            output = io.StringIO()

            with redirect_stdout(output):
                status = main(["ask", "retrieval", "--index", str(index), "--json"])

            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.getvalue())["query"], "retrieval")

    def test_cli_rejects_conflicting_sources_and_index_only_overrides(self):
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit):
            main(["ask", "question", "--config", "rag.yaml", "--index", "index"])

        errors = io.StringIO()
        with redirect_stderr(errors):
            status = main(["ask", "question", "--config", "rag.yaml", "--dimension", "32"])
        self.assertEqual(status, 2)
        self.assertIn("can only be used with --index", errors.getvalue())

    def test_server_passes_no_hardcoded_hashing_dimension(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest("fastapi is not installed")
        from cheragh.server.app import create_app

        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index"
            self._save_hashing_store(index, dimension=31)

            with patch.object(MemoryVectorStore, "load", wraps=MemoryVectorStore.load) as load:
                app = create_app(index_path=str(index))

            self.assertIsNotNone(app)
            schema = app.openapi()
            self.assertIn("/ask", schema["paths"])
            self.assertIn("AskRequest", schema["components"]["schemas"])
            load.assert_called_once_with(str(index), None)


if __name__ == "__main__":
    unittest.main()
