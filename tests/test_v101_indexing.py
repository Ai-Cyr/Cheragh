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

    def test_chunking_option_changes_rebuild_unchanged_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "docs"
            root.mkdir()
            (root / "guide.txt").write_text(
                " ".join(f"section-{index:03d}" for index in range(80)),
                encoding="utf-8",
            )
            output = Path(tmp) / "index"

            index_path(
                root,
                output,
                embedding_model=CountingEmbedding(),
                chunk_size=80,
                chunk_overlap=8,
            )

            resized_encoder = CountingEmbedding()
            resized = index_path(
                root,
                output,
                embedding_model=resized_encoder,
                chunk_size=140,
                chunk_overlap=8,
            )
            self.assertTrue(resized["indexing_options_changed"])
            self.assertEqual(resized["changed_files"], 1)
            self.assertEqual(resized["unchanged_files"], 0)
            self.assertEqual(resized_encoder.embedded_document_count, resized["indexed_documents"])
            self.assertGreater(resized_encoder.embedded_document_count, 0)

            overlap_encoder = CountingEmbedding()
            overlap_changed = index_path(
                root,
                output,
                embedding_model=overlap_encoder,
                chunk_size=140,
                chunk_overlap=24,
            )
            self.assertTrue(overlap_changed["indexing_options_changed"])
            self.assertEqual(overlap_encoder.embedded_document_count, overlap_changed["indexed_documents"])
            manifest_options = load_manifest(output).metadata["indexing_options"]
            self.assertEqual(manifest_options["chunk_size"], 140)
            self.assertEqual(manifest_options["chunk_overlap"], 24)

    def test_corpus_selection_option_changes_invalidate_incremental_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "docs"
            root.mkdir()
            (root / "guide.txt").write_text("stable corpus content", encoding="utf-8")
            output = Path(tmp) / "index"
            baseline = {
                "recursive": True,
                "include_pdf": False,
                "include_docx": False,
                "exclude_patterns": ["*.tmp"],
                "max_file_size_mb": 1,
            }
            index_path(root, output, embedding_model=HashingEmbedding(32), **baseline)
            original_options = load_manifest(output).metadata["indexing_options"]

            changes = {
                "recursive": False,
                "include_pdf": True,
                "include_docx": True,
                "exclude_patterns": ["*.tmp", "*.log"],
                "max_file_size_mb": 2,
            }
            for option, value in changes.items():
                with self.subTest(option=option):
                    candidate = dict(baseline)
                    candidate[option] = value
                    result = index_path(
                        root,
                        output,
                        embedding_model=HashingEmbedding(32),
                        dry_run=True,
                        **candidate,
                    )
                    self.assertTrue(result["indexing_options_changed"])
                    self.assertEqual(result["plan"]["changed_count"], 1)
                    self.assertEqual(result["plan"]["unchanged_count"], 0)

            # Pattern order and duplicates are semantically irrelevant.
            equivalent = dict(baseline)
            equivalent["exclude_patterns"] = ["*.tmp", "*.tmp"]
            unchanged = index_path(
                root,
                output,
                embedding_model=HashingEmbedding(32),
                dry_run=True,
                **equivalent,
            )
            self.assertFalse(unchanged["indexing_options_changed"])
            self.assertEqual(unchanged["plan"]["unchanged_count"], 1)
            self.assertEqual(load_manifest(output).metadata["indexing_options"], original_options)

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
