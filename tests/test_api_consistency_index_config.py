import json
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from cheragh.base import HashingEmbedding
from cheragh.cli.main import main
from cheragh.indexing import index_from_config


class IndexFromConfigContractTests(unittest.TestCase):
    def _write_config(self, root: Path, data: dict) -> Path:
        config = root / "configs" / "rag.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps(data), encoding="utf-8")
        return config

    def test_forwards_validated_ingestion_and_indexing_options(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._write_config(
                root,
                {
                    "ingestion": {
                        "path": "../corpus",
                        "chunk_size": 64,
                        "chunk_overlap": 8,
                        "recursive": False,
                        "exclude_patterns": ["*.tmp"],
                        "max_file_size_mb": 3.5,
                    },
                    "embedding": {"provider": "hashing", "dimension": 17},
                    "vectorstore": {"type": "memory", "path": "configured-index"},
                    "indexing": {
                        "incremental": False,
                        "dry_run": False,
                        "force": True,
                        "use_lock": False,
                        "lock_timeout_seconds": 2.5,
                    },
                },
            )

            with patch("cheragh.indexing.index_path", return_value={"ok": True}) as build:
                result = index_from_config(config, output="explicit-index")

            self.assertEqual(result, {"ok": True})
            args, kwargs = build.call_args
            self.assertEqual(args[0], (config.parent / "../corpus").resolve())
            self.assertEqual(args[1], (config.parent / "explicit-index").resolve())
            self.assertIsInstance(kwargs["embedding_model"], HashingEmbedding)
            self.assertEqual(kwargs["embedding_model"].dimension, 17)
            self.assertEqual(kwargs["chunk_size"], 64)
            self.assertEqual(kwargs["chunk_overlap"], 8)
            self.assertIs(kwargs["recursive"], False)
            self.assertIs(kwargs["incremental"], False)
            self.assertIs(kwargs["dry_run"], False)
            self.assertIs(kwargs["force"], True)
            self.assertEqual(kwargs["exclude_patterns"], ["*.tmp"])
            self.assertEqual(kwargs["max_file_size_mb"], 3.5)
            self.assertIs(kwargs["use_lock"], False)
            self.assertEqual(kwargs["lock_timeout_seconds"], 2.5)

    def test_output_precedence_and_default_are_config_relative(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {
                "ingestion": {"path": "corpus", "max_file_size_mb": None},
                "vectorstore": {"type": "memory", "path": "saved-index"},
            }
            config = self._write_config(root, base)

            with patch("cheragh.indexing.index_path", return_value={}) as build:
                index_from_config(config, embedding_model=HashingEmbedding(8))
            self.assertEqual(build.call_args.args[1], (config.parent / "saved-index").resolve())
            self.assertIsNone(build.call_args.kwargs["max_file_size_mb"])

            default_config = self._write_config(
                root / "default",
                {"ingestion": {"path": "corpus"}, "vectorstore": {"type": "memory"}},
            )
            with patch("cheragh.indexing.index_path", return_value={}) as build:
                index_from_config(default_config, embedding_model=HashingEmbedding(8))
            self.assertEqual(build.call_args.args[1], (default_config.parent / ".cheragh_index").resolve())

    def test_overrides_are_whitelisted_revalidated_and_applied(self):
        with TemporaryDirectory() as tmp:
            config = self._write_config(
                Path(tmp),
                {"ingestion": {"path": "corpus"}, "vectorstore": {"type": "memory"}},
            )
            with patch("cheragh.indexing.index_path", return_value={}) as build:
                index_from_config(
                    config,
                    embedding_model=HashingEmbedding(8),
                    chunk_size=32,
                    chunk_overlap=4,
                    dry_run=True,
                )
            self.assertEqual(build.call_args.kwargs["chunk_size"], 32)
            self.assertEqual(build.call_args.kwargs["chunk_overlap"], 4)
            self.assertIs(build.call_args.kwargs["dry_run"], True)

            override_path = Path("../other-corpus")
            with patch("cheragh.indexing.index_path", return_value={}) as build:
                index_from_config(
                    config,
                    embedding_model=HashingEmbedding(8),
                    path=override_path,
                )
            self.assertEqual(build.call_args.args[0], (config.parent / override_path).resolve())

            with self.assertRaisesRegex(TypeError, "Unsupported index_from_config override"):
                index_from_config(config, typo=True)
            with self.assertRaisesRegex(ValueError, "cannot be None"):
                index_from_config(config, dry_run=None)
            with self.assertRaisesRegex(ValueError, "cannot be None"):
                index_from_config(config, max_file_size_mb=None)
            with self.assertRaisesRegex(Exception, "chunk_overlap"):
                index_from_config(config, chunk_size=4, chunk_overlap=4)

    def test_rejects_non_memory_vectorstore_before_indexing(self):
        with TemporaryDirectory() as tmp:
            config = self._write_config(
                Path(tmp),
                {"ingestion": {"path": "corpus"}, "vectorstore": {"type": "faiss"}},
            )
            with self.assertRaisesRegex(ValueError, "only writes the local MemoryVectorStore format"):
                index_from_config(config)

    def test_dry_run_does_not_create_output_directory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._write_config(
                root,
                {
                    "ingestion": {"path": "../corpus", "chunk_size": 32, "chunk_overlap": 4},
                    "embedding": {"provider": "hashing", "dimension": 8},
                    "vectorstore": {"type": "memory", "path": "nested/index"},
                    "indexing": {"dry_run": True, "use_lock": True},
                },
            )
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "note.txt").write_text("Cheragh dry run", encoding="utf-8")
            output = config.parent / "nested" / "index"

            result = index_from_config(config)

            self.assertIs(result["dry_run"], True)
            self.assertEqual(Path(result["output"]), output.resolve())
            self.assertFalse(output.exists())

    def test_config_chunking_change_rebuilds_an_unchanged_source(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "guide.txt").write_text(
                " ".join(f"topic-{index:03d}" for index in range(60)),
                encoding="utf-8",
            )
            data = {
                "ingestion": {
                    "path": "../corpus",
                    "chunk_size": 60,
                    "chunk_overlap": 6,
                },
                "embedding": {"provider": "hashing", "dimension": 8},
                "vectorstore": {"type": "memory", "path": "index"},
                "indexing": {"incremental": True},
            }
            config = self._write_config(root, data)
            index_from_config(config)

            data["ingestion"]["chunk_size"] = 120
            data["ingestion"]["chunk_overlap"] = 12
            config.write_text(json.dumps(data), encoding="utf-8")
            rebuilt = index_from_config(config)

            self.assertTrue(rebuilt["indexing_options_changed"])
            self.assertEqual(rebuilt["changed_files"], 1)
            self.assertEqual(rebuilt["unchanged_files"], 0)

    def test_available_from_package_root(self):
        from cheragh import index_from_config as exported

        self.assertIs(exported, index_from_config)


class IndexConfigCLIContractTests(unittest.TestCase):
    def test_config_mode_passes_no_implicit_overrides(self):
        output = io.StringIO()
        with patch("cheragh.cli.main.index_from_config", return_value={"ok": True}) as configured_index:
            with redirect_stdout(output):
                status = main(["index", "--config", "rag.yaml"])

        self.assertEqual(status, 0)
        configured_index.assert_called_once_with("rag.yaml")
        self.assertEqual(json.loads(output.getvalue()), {"ok": True})

    def test_config_mode_passes_only_explicit_overrides_including_false_booleans(self):
        output = io.StringIO()
        argv = [
            "index",
            "--config",
            "rag.yaml",
            "--output",
            "custom-index",
            "--dimension",
            "29",
            "--chunk-size",
            "64",
            "--chunk-overlap",
            "8",
            "--no-incremental",
            "--no-dry-run",
            "--force",
            "--exclude",
            "*.tmp",
            "--max-file-size-mb",
            "3.5",
            "--no-lock",
            "--lock-timeout",
            "2.5",
        ]
        with patch("cheragh.cli.main.index_from_config", return_value={}) as configured_index:
            with redirect_stdout(output):
                status = main(argv)

        self.assertEqual(status, 0)
        args, kwargs = configured_index.call_args
        self.assertEqual(args, ("rag.yaml",))
        self.assertEqual(kwargs["output"], "custom-index")
        self.assertIsInstance(kwargs["embedding_model"], HashingEmbedding)
        self.assertEqual(kwargs["embedding_model"].dimension, 29)
        self.assertEqual(
            {key: value for key, value in kwargs.items() if key != "embedding_model"},
            {
                "output": "custom-index",
                "chunk_size": 64,
                "chunk_overlap": 8,
                "incremental": False,
                "dry_run": False,
                "force": True,
                "exclude_patterns": ["*.tmp"],
                "max_file_size_mb": 3.5,
                "use_lock": False,
                "lock_timeout_seconds": 2.5,
            },
        )

    def test_positive_boolean_flags_are_explicit_config_overrides(self):
        output = io.StringIO()
        with patch("cheragh.cli.main.index_from_config", return_value={}) as configured_index:
            with redirect_stdout(output):
                status = main(
                    [
                        "index",
                        "--config",
                        "rag.yaml",
                        "--incremental",
                        "--dry-run",
                        "--no-force",
                        "--use-lock",
                    ]
                )

        self.assertEqual(status, 0)
        self.assertEqual(
            configured_index.call_args.kwargs,
            {"incremental": True, "dry_run": True, "force": False, "use_lock": True},
        )

    def test_direct_mode_preserves_historical_defaults(self):
        output = io.StringIO()
        with patch("cheragh.cli.main.build_index", return_value={}) as direct_index:
            with redirect_stdout(output):
                status = main(["index", "corpus"])

        self.assertEqual(status, 0)
        args, kwargs = direct_index.call_args
        self.assertEqual(args, ("corpus", ".cheragh_index"))
        self.assertIsInstance(kwargs["embedding_model"], HashingEmbedding)
        self.assertEqual(kwargs["embedding_model"].dimension, 384)
        self.assertEqual(
            {key: value for key, value in kwargs.items() if key != "embedding_model"},
            {
                "chunk_size": 800,
                "chunk_overlap": 120,
                "incremental": True,
                "dry_run": False,
                "force": False,
                "exclude_patterns": None,
                "max_file_size_mb": 50,
                "use_lock": True,
                "lock_timeout_seconds": 10.0,
            },
        )

    def test_index_requires_exactly_one_source(self):
        for argv in (["index"], ["index", "corpus", "--config", "rag.yaml"]):
            with self.subTest(argv=argv):
                errors = io.StringIO()
                with redirect_stderr(errors):
                    status = main(list(argv))
                self.assertEqual(status, 2)
                self.assertIn("exactly one source", errors.getvalue())

    def test_config_errors_are_readable_usage_errors(self):
        errors = io.StringIO()
        with patch("cheragh.cli.main.index_from_config", side_effect=ValueError("bad chunk_overlap")):
            with redirect_stderr(errors):
                status = main(["index", "--config", "rag.yaml"])

        self.assertEqual(status, 2)
        self.assertIn("Invalid index configuration: bad chunk_overlap", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
