"""Command line interface for cheragh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..base import HashingEmbedding, OpenAILLMClient, ExtractiveLLMClient
from ..engine import RAGEngine
from ..evaluation import evaluate_retrieval
from ..indexing import index_from_config, index_path as build_index, inspect_index
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


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cheragh", description="Index, query, serve and evaluate RAG corpora.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create a starter rag.yaml")
    init.add_argument("--output", "-o", default="rag.yaml")

    index = sub.add_parser("index", help="Index a file or directory into a local vector store")
    index.add_argument("path", nargs="?", help="File or directory to index")
    index.add_argument("--config", default=None, help="Load ingestion, embedding, and indexing options from YAML/JSON")
    index.add_argument("--output", "-o", default=None, help="Output index directory")
    index.add_argument("--chunk-size", type=_positive_int, default=None)
    index.add_argument("--chunk-overlap", type=int, default=None)
    index.add_argument("--dimension", type=_positive_int, default=None, help="Override the HashingEmbedding dimension")
    incremental = index.add_mutually_exclusive_group()
    incremental.add_argument(
        "--incremental",
        action="store_true",
        dest="incremental",
        help="Re-index only changed files and remove deleted ones",
    )
    incremental.add_argument(
        "--no-incremental",
        action="store_false",
        dest="incremental",
        help="Rebuild the entire index",
    )
    dry_run = index.add_mutually_exclusive_group()
    dry_run.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Show the incremental plan without writing the index",
    )
    dry_run.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Write the index even if config enables dry-run",
    )
    force = index.add_mutually_exclusive_group()
    force.add_argument("--force", action="store_true", dest="force", help="Treat all current files as changed")
    force.add_argument(
        "--no-force",
        action="store_false",
        dest="force",
        help="Do not force unchanged files to be re-indexed",
    )
    index.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Additional glob exclusion pattern; can be repeated",
    )
    index.add_argument("--max-file-size-mb", type=float, default=None)
    locking = index.add_mutually_exclusive_group()
    locking.add_argument("--use-lock", action="store_true", dest="use_lock", help="Enable the index writer lock")
    locking.add_argument(
        "--no-lock",
        "--no-use-lock",
        action="store_false",
        dest="use_lock",
        help="Disable the index writer lock",
    )
    index.add_argument(
        "--lock-timeout",
        "--lock-timeout-seconds",
        dest="lock_timeout_seconds",
        type=_non_negative_float,
        default=None,
        help="Seconds to wait for the index writer lock",
    )
    index.set_defaults(incremental=None, dry_run=None, force=None, use_lock=None)

    ask = sub.add_parser("ask", help="Ask a question against a config or local vector index")
    ask.add_argument("question")
    ask_source = ask.add_mutually_exclusive_group()
    ask_source.add_argument("--config", default=None, help="Load a RAGEngine from YAML/JSON config")
    ask_source.add_argument("--index", default=None, help="Index directory (default: .cheragh_index)")
    ask.add_argument("--top-k", type=_positive_int, default=None, help="Override the engine/config top_k")
    ask.add_argument(
        "--dimension",
        type=_positive_int,
        default=None,
        help="Validate a HashingEmbedding dimension (default: derive it safely from the index manifest)",
    )
    ask.add_argument("--openai-model", default=None, help="Use OpenAI for generation when provided")
    ask.add_argument("--json", action="store_true", help="Return JSON")
    ask.add_argument("--include-prompt", action="store_true", help="Include full prompt in trace JSON")
    ask.add_argument("--trace-output", default=None, help="Append request traces to this JSONL file")

    evaluate = sub.add_parser("eval", help="Evaluate retrieval from a JSONL dataset")
    evaluate.add_argument("dataset", help="JSONL with query and expected_doc_ids")
    evaluate.add_argument("--index", default=".cheragh_index")
    evaluate.add_argument("--top-k", type=_positive_int, default=5)
    evaluate.add_argument(
        "--dimension",
        type=_positive_int,
        default=None,
        help="Validate a HashingEmbedding dimension (default: derive it safely from the index manifest)",
    )

    inspect = sub.add_parser("inspect-index", help="Inspect a local vector index")
    inspect.add_argument("--index", default=".cheragh_index")

    doctor = sub.add_parser("doctor", help="Check local installation and optional dependencies")
    doctor.add_argument("--json", action="store_true", help="Print checks as JSON")

    validate = sub.add_parser("validate-config", help="Validate a YAML/JSON config with the v1.0 Pydantic schema")
    validate.add_argument("config", help="Path to rag.yaml or rag.json")
    validate.add_argument("--json", action="store_true", help="Print normalized config as JSON")

    techniques = sub.add_parser("techniques", help="Inspect the machine-readable RAG technique catalogue")
    technique_sub = techniques.add_subparsers(dest="techniques_command", required=True)
    technique_list = technique_sub.add_parser("list", help="List techniques and maturity levels")
    technique_list.add_argument("--status", choices=["stable", "beta", "experimental", "planned"])
    technique_list.add_argument(
        "--family",
        choices=["indexing", "retrieval", "query", "augmentation", "orchestration", "structured", "multimodal", "evaluation", "governance"],
    )
    technique_list.add_argument("--available", action="store_true", help="Show only implemented techniques")
    technique_list.add_argument("--json", action="store_true")
    technique_show = technique_sub.add_parser("show", help="Show one technique")
    technique_show.add_argument("technique_id")
    technique_show.add_argument("--json", action="store_true")

    serve = sub.add_parser("serve", help="Serve a RAG API with FastAPI")
    serve_source = serve.add_mutually_exclusive_group(required=True)
    serve_source.add_argument("--config", default=None)
    serve_source.add_argument("--index", default=None)
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
    if args.command == "techniques":
        return _cmd_techniques(args)
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
    if (args.path is None) == (args.config is None):
        print("index requires exactly one source: PATH or --config CONFIG", file=sys.stderr, flush=True)
        return 2

    if args.config:
        call_kwargs = {}
        if args.output is not None:
            call_kwargs["output"] = args.output
        if args.dimension is not None:
            call_kwargs["embedding_model"] = HashingEmbedding(dimension=args.dimension)
        config_overrides = {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "incremental": args.incremental,
            "dry_run": args.dry_run,
            "force": args.force,
            "exclude_patterns": args.exclude,
            "max_file_size_mb": args.max_file_size_mb,
            "use_lock": args.use_lock,
            "lock_timeout_seconds": args.lock_timeout_seconds,
        }
        call_kwargs.update({key: value for key, value in config_overrides.items() if value is not None})
        try:
            result = index_from_config(args.config, **call_kwargs)
        except Exception as exc:
            print(f"Invalid index configuration: {exc}", file=sys.stderr, flush=True)
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0

    result = build_index(
        args.path,
        args.output if args.output is not None else ".cheragh_index",
        embedding_model=HashingEmbedding(dimension=args.dimension if args.dimension is not None else 384),
        chunk_size=args.chunk_size if args.chunk_size is not None else 800,
        chunk_overlap=args.chunk_overlap if args.chunk_overlap is not None else 120,
        incremental=args.incremental if args.incremental is not None else True,
        dry_run=args.dry_run if args.dry_run is not None else False,
        force=args.force if args.force is not None else False,
        exclude_patterns=args.exclude,
        max_file_size_mb=args.max_file_size_mb if args.max_file_size_mb is not None else 50,
        use_lock=args.use_lock if args.use_lock is not None else True,
        lock_timeout_seconds=args.lock_timeout_seconds if args.lock_timeout_seconds is not None else 10.0,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    if args.config:
        if args.dimension is not None or args.openai_model is not None:
            print("--dimension and --openai-model can only be used with --index", file=sys.stderr, flush=True)
            return 2
        engine = RAGEngine.from_config(args.config)
        if args.trace_output:
            engine.trace_export_path = Path(args.trace_output)
    else:
        embedder = HashingEmbedding(dimension=args.dimension) if args.dimension is not None else None
        store = MemoryVectorStore.load(args.index or ".cheragh_index", embedder)
        llm = OpenAILLMClient(model=args.openai_model) if args.openai_model else ExtractiveLLMClient()
        effective_top_k = args.top_k if args.top_k is not None else 5
        engine = RAGEngine(store.as_retriever(), llm_client=llm, top_k=effective_top_k, trace_export_path=args.trace_output)
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
    embedder = HashingEmbedding(dimension=args.dimension) if args.dimension is not None else None
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


def _cmd_techniques(args: argparse.Namespace) -> int:
    from ..catalog import get_technique, list_techniques

    if args.techniques_command == "show":
        try:
            spec = get_technique(args.technique_id)
        except KeyError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            return 1
        payload = spec.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        else:
            print(f"{spec.id}: {spec.name}", flush=True)
            print(f"status={spec.status.value} family={spec.family.value} available={spec.available}", flush=True)
            print(spec.summary, flush=True)
            if spec.implementation:
                print(f"implementation={spec.implementation}", flush=True)
            for limitation in spec.limitations:
                print(f"limitation: {limitation}", flush=True)
        return 0

    specs = list_techniques(
        status=args.status,
        family=args.family,
        available=True if args.available else None,
    )
    if args.json:
        print(json.dumps([spec.to_dict() for spec in specs], ensure_ascii=False, indent=2), flush=True)
    else:
        for spec in specs:
            marker = "yes" if spec.available else "no"
            print(f"{spec.id:24} {spec.status.value:12} {spec.family.value:14} available={marker}", flush=True)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
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
