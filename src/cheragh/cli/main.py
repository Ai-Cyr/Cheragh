"""Command line interface for cheragh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..base import HashingEmbedding, OpenAILLMClient, ExtractiveLLMClient
from ..engine import RAGEngine
from ..evaluation import evaluate_retrieval
from ..indexing import index_path as build_index, inspect_index
from ..vectorstores import MemoryVectorStore

DEFAULT_CONFIG = """# cheragh configuration
ingestion:
  path: ./docs
  chunk_size: 800
  chunk_overlap: 120

embedding:
  provider: hashing
  dimension: 384

retriever:
  type: memory
  top_k: 5

compression:
  enabled: true
  type: default

query:
  enabled: false
  type: multi-query

generation:
  provider: extractive

strict_grounding: true
require_citations: false
trace_enabled: true

observability:
  enabled: true
  trace_export_path: .cheragh/traces.jsonl
  trace_include_prompt: false

indexing:
  incremental: true
  use_lock: true
  lock_timeout_seconds: 10
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cheragh", description="Index, query, serve and evaluate RAG corpora.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create a starter rag.yaml")
    init.add_argument("--output", "-o", default="rag.yaml")

    index = sub.add_parser("index", help="Index a file or directory into a local vector store")
    index.add_argument("path", help="File or directory to index")
    index.add_argument("--output", "-o", default=".cheragh_index", help="Output index directory")
    index.add_argument("--chunk-size", type=int, default=800)
    index.add_argument("--chunk-overlap", type=int, default=120)
    index.add_argument("--dimension", type=int, default=384, help="HashingEmbedding dimension")
    index.add_argument("--incremental", action="store_true", default=True, help="Re-index only changed files and remove deleted ones")
    index.add_argument("--no-incremental", action="store_false", dest="incremental", help="Rebuild the entire index")
    index.add_argument("--dry-run", action="store_true", help="Show the incremental plan without writing the index")
    index.add_argument("--force", action="store_true", help="Treat all current files as changed")
    index.add_argument("--exclude", action="append", default=None, help="Additional glob exclusion pattern; can be repeated")
    index.add_argument("--max-file-size-mb", type=float, default=50)
    index.add_argument("--no-lock", action="store_false", dest="use_lock", help="Disable index writer lock")

    ask = sub.add_parser("ask", help="Ask a question against a config or local vector index")
    ask.add_argument("question")
    ask.add_argument("--config", default=None, help="Load a RAGEngine from YAML/JSON config")
    ask.add_argument("--index", default=".cheragh_index", help="Index directory")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--dimension", type=int, default=384)
    ask.add_argument("--openai-model", default=None, help="Use OpenAI for generation when provided")
    ask.add_argument("--json", action="store_true", help="Return JSON")
    ask.add_argument("--include-prompt", action="store_true", help="Include full prompt in trace JSON")
    ask.add_argument("--trace-output", default=None, help="Append request traces to this JSONL file")

    evaluate = sub.add_parser("eval", help="Evaluate retrieval from a JSONL dataset")
    evaluate.add_argument("dataset", help="JSONL with query and expected_doc_ids")
    evaluate.add_argument("--index", default=".cheragh_index")
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--dimension", type=int, default=384)

    inspect = sub.add_parser("inspect-index", help="Inspect a local vector index")
    inspect.add_argument("--index", default=".cheragh_index")

    doctor = sub.add_parser("doctor", help="Check local installation and optional dependencies")
    doctor.add_argument("--json", action="store_true", help="Print checks as JSON")

    validate = sub.add_parser("validate-config", help="Validate a YAML/JSON config with the v1.0 Pydantic schema")
    validate.add_argument("config", help="Path to rag.yaml or rag.json")
    validate.add_argument("--json", action="store_true", help="Print normalized config as JSON")

    serve = sub.add_parser("serve", help="Serve a RAG API with FastAPI")
    serve.add_argument("--config", default=None)
    serve.add_argument("--index", default=None)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--enable-indexing",
        action="store_true",
        help="Enable the disabled-by-default POST /index endpoint",
    )
    serve.add_argument("--index-root", default=None, help="Restrict POST /index paths to this root")
    serve.add_argument("--api-key", default=None, help="Require this X-API-Key on API endpoints")

    args = parser.parse_args(argv)
    if args.command == "init":
        return _cmd_init(args)
    if args.command == "index":
        return _cmd_index(args)
    if args.command == "ask":
        return _cmd_ask(args)
    if args.command == "eval":
        return _cmd_eval(args)
    if args.command == "inspect-index":
        return _cmd_inspect(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "validate-config":
        return _cmd_validate_config(args)
    if args.command == "serve":
        return _cmd_serve(args)
    return 2


def _cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.output)
    if path.exists():
        print(f"Refusing to overwrite existing file: {path}", file=sys.stderr, flush=True)
        return 1
    path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    print(f"Created {path}", flush=True)
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    result = build_index(
        args.path,
        args.output,
        embedding_model=HashingEmbedding(dimension=args.dimension),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        incremental=args.incremental,
        dry_run=args.dry_run,
        force=args.force,
        exclude_patterns=args.exclude,
        max_file_size_mb=args.max_file_size_mb,
        use_lock=args.use_lock,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    if args.config:
        engine = RAGEngine.from_config(args.config)
        if args.trace_output:
            engine.trace_export_path = Path(args.trace_output)
    else:
        embedder = HashingEmbedding(dimension=args.dimension)
        store = MemoryVectorStore.load(args.index, embedder)
        llm = OpenAILLMClient(model=args.openai_model) if args.openai_model else ExtractiveLLMClient()
        engine = RAGEngine(store.as_retriever(), llm_client=llm, top_k=args.top_k, trace_export_path=args.trace_output)
    response = engine.ask(args.question, top_k=args.top_k)
    if args.json:
        data = response.to_dict(include_prompt=args.include_prompt)
        print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
    else:
        print(response.answer, flush=True)
        if response.sources:
            print("\nSources:", flush=True)
            for source in response.sources:
                score = f" score={source.score:.4f}" if source.score is not None else ""
                print(f"- {source.doc_id}{score}", flush=True)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    embedder = HashingEmbedding(dimension=args.dimension)
    store = MemoryVectorStore.load(args.index, embedder)
    examples = []
    with Path(args.dataset).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    result = evaluate_retrieval(examples, store.as_retriever(), top_k=args.top_k)
    print(json.dumps({"metrics": result.metrics, "rows": result.rows}, ensure_ascii=False, indent=2), flush=True)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_index(args.index), ensure_ascii=False, indent=2), flush=True)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    import importlib.util
    from .. import __version__

    optional = ["numpy", "pydantic", "yaml", "fastapi", "qdrant_client", "chromadb", "redis", "sentence_transformers"]
    checks = {name: importlib.util.find_spec(name) is not None for name in optional}
    payload = {"version": __version__, "optional_dependencies": checks}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"cheragh {__version__}", flush=True)
        for name, ok in checks.items():
            print(f"- {name}: {'ok' if ok else 'missing'}", flush=True)
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    from pydantic import ValidationError
    from ..config import load_and_validate_config

    try:
        config = load_and_validate_config(args.config)
    except ValidationError as exc:
        print(exc, file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        print(f"Invalid config: {exc}", file=sys.stderr, flush=True)
        return 1
    if args.json:
        print(json.dumps(config.to_legacy_dict(), ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"Config OK: {args.config}", flush=True)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    if not args.config and not args.index:
        print("serve requires --config or --index", file=sys.stderr, flush=True)
        return 1
    from ..server.main import serve

    serve(
        config=args.config,
        index=args.index,
        host=args.host,
        port=args.port,
        enable_indexing=args.enable_indexing,
        allowed_index_root=args.index_root,
        api_key=args.api_key,
    )
    return 0


def cli_entrypoint() -> None:  # pragma: no cover
    code = main(sys.argv[1:])
    raise SystemExit(code)


if __name__ == "__main__":  # pragma: no cover
    cli_entrypoint()
