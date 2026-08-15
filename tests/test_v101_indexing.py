import tempfile
import unittest
from pathlib import Path

from cheragh import HashingEmbedding, index_path, load_manifest
from cheragh.ingestion import load_documents


class CountingEmbedding(HashingEmbedding):
    def __init__(self, dimension=32):
        super().__init__(dimension=dimension)
        self.embedded_document_count = 0

    def embed_documents(self, texts):
        self.embedded_document_count += len(texts)
        return super().embed_documents(texts)


class V101IndexingSafetyTests(unittest.TestCase):
    def test_output_inside_source_is_never_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.txt").write_text("source knowledge", encoding="utf-8")
            output = root / ".cheragh_index"

            index_path(root, output, embedding_model=HashingEmbedding(32))
            second = index_path(root, output, embedding_model=HashingEmbedding(32))

            self.assertEqual(second["changed_files"], 0)
            self.assertEqual(list(load_manifest(output).files), [str((root / "source.txt").resolve())])

    def test_incremental_update_reuses_unchanged_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "docs"
            root.mkdir()
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "b.txt").write_text("beta", encoding="utf-8")
            output = Path(tmp) / "index"

            first_encoder = CountingEmbedding()
            index_path(root, output, embedding_model=first_encoder)
            self.assertEqual(first_encoder.embedded_document_count, 2)

            second_encoder = CountingEmbedding()
            index_path(root, output, embedding_model=second_encoder)
            self.assertEqual(second_encoder.embedded_document_count, 0)

    def test_custom_excludes_do_not_disable_safe_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "keep.txt").write_text("keep", encoding="utf-8")
            (root / "skip.log").write_text("skip", encoding="utf-8")
            cache = root / ".cheragh"
            cache.mkdir()
            (cache / "generated.json").write_text('{"generated": true}', encoding="utf-8")

            docs = load_documents(root, exclude_patterns=["*.log"])
            self.assertEqual([doc.metadata["filename"] for doc in docs], ["keep.txt"])


if __name__ == "__main__":
    unittest.main()
