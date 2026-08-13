import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cheragh import Document, HashingEmbedding, MemoryVectorStore, ParentDocumentRetriever, index_path


class V101PersistenceHardeningTests(unittest.TestCase):
    def test_parent_document_generated_id_is_sha256_stable(self):
        content = "A parent document with a process-independent identity."
        expected_id = f"parent::{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
        first_document = Document(content)
        second_document = Document(content)

        first = ParentDocumentRetriever(
            [first_document],
            HashingEmbedding(16),
            child_chunk_size=20,
            child_chunk_overlap=0,
        )
        second = ParentDocumentRetriever(
            [second_document],
            HashingEmbedding(16),
            child_chunk_size=20,
            child_chunk_overlap=0,
        )

        self.assertEqual(first_document.doc_id, expected_id)
        self.assertEqual(second_document.doc_id, expected_id)
        self.assertEqual(first.child_documents[0].metadata["parent_id"], expected_id)
        self.assertEqual(second.child_documents[0].doc_id, f"{expected_id}::child::0")

    def test_memory_store_save_stages_then_atomically_replaces_all_files(self):
        store = MemoryVectorStore(HashingEmbedding(16))
        store.add_documents([Document("alpha", doc_id="alpha")])

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch("cheragh.vectorstores.memory.os.replace", wraps=os.replace) as replace:
                store.save(output)

            destinations = [Path(call.args[1]).name for call in replace.call_args_list]
            self.assertEqual(destinations, ["documents.jsonl", "embeddings.npy", "manifest.json"])
            self.assertTrue(all(Path(call.args[0]).parent == output for call in replace.call_args_list))
            self.assertEqual(list(output.glob(".*.tmp")), [])
            loaded = MemoryVectorStore.load(output, HashingEmbedding(16))
            self.assertEqual([document.doc_id for document in loaded.documents], ["alpha"])

    def test_memory_store_staging_failure_preserves_previous_snapshot(self):
        store = MemoryVectorStore(HashingEmbedding(16))
        store.add_documents([Document("alpha", doc_id="alpha")])

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            store.save(output)
            persisted_names = ("documents.jsonl", "embeddings.npy", "manifest.json")
            before = {name: (output / name).read_bytes() for name in persisted_names}

            store.add_documents([Document("beta", doc_id="beta")])
            with patch("numpy.save", side_effect=OSError("simulated write failure")):
                with self.assertRaisesRegex(OSError, "simulated write failure"):
                    store.save(output)

            after = {name: (output / name).read_bytes() for name in persisted_names}
            self.assertEqual(after, before)
            self.assertEqual(list(output.glob(".*.tmp")), [])
            loaded = MemoryVectorStore.load(output, HashingEmbedding(16))
            self.assertEqual([document.doc_id for document in loaded.documents], ["alpha"])

    def test_atomic_store_format_remains_incremental_index_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "docs"
            source.mkdir()
            (source / "alpha.txt").write_text("alpha", encoding="utf-8")
            output = Path(tmp) / "index"

            index_path(source, output, embedding_model=HashingEmbedding(16))
            second = index_path(source, output, embedding_model=HashingEmbedding(16))

            self.assertEqual(second["changed_files"], 0)
            self.assertEqual(second["unchanged_files"], 1)
            loaded = MemoryVectorStore.load(output, HashingEmbedding(16))
            self.assertEqual(len(loaded.documents), 1)


if __name__ == "__main__":
    unittest.main()
