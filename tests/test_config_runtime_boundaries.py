"""Startup configuration rejects malformed values before building resources."""

import json
import math

import pytest
from pydantic import ValidationError

from cheragh.cli.main import main
from cheragh.config import load_and_validate_config, load_config, load_raw_config, validate_config


@pytest.mark.parametrize("contents", ["false", "0", "[]", "null", "''", "", "- item"])
@pytest.mark.parametrize("loader", [load_raw_config, load_config, load_and_validate_config])
def test_yaml_non_mapping_roots_do_not_enable_default_configuration(tmp_path, contents, loader):
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="Config root must be a mapping"):
        loader(path)


@pytest.mark.parametrize("loader", [load_config, load_and_validate_config])
def test_yaml_recursive_alias_is_rejected_without_recursion_error(tmp_path, loader):
    path = tmp_path / "config.yaml"
    path.write_text("retriever:\n  filters: &cycle\n    nested: *cycle\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cyclic reference"):
        loader(path)


def test_yaml_alias_expansion_is_bounded_before_schema_validation(tmp_path):
    path = tmp_path / "config.yaml"
    lines = ["a0: &a0 [leaf, leaf]"]
    lines.extend(f"a{i}: &a{i} [*a{i - 1}, *a{i - 1}]" for i in range(1, 20))
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="expanded values"):
        load_config(path, validate=False)


@pytest.mark.parametrize("extension", ["yaml", "json"])
def test_config_nesting_is_bounded_before_schema_validation(tmp_path, extension):
    data = {}
    for _ in range(70):
        data = {"nested": data}
    path = tmp_path / f"config.{extension}"
    # JSON is also valid YAML and gives the two parsers the same structure.
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="nesting levels"):
        load_config(path, validate=False)


def test_shared_yaml_aliases_and_environment_references_still_work(tmp_path, monkeypatch):
    monkeypatch.setenv("CHERAGH_TEST_TENANT", "tenant-alpha")
    path = tmp_path / "config.yaml"
    path.write_text(
        "retriever:\n  filters:\n    first: &scope {tenant: '${CHERAGH_TEST_TENANT}'}\n"
        "    second: *scope\n",
        encoding="utf-8",
    )
    config = load_config(path)
    filters = config["retriever"]["filters"]
    assert filters == {"first": {"tenant": "tenant-alpha"}, "second": {"tenant": "tenant-alpha"}}
    filters["first"]["tenant"] = "changed"
    assert filters["second"]["tenant"] == "tenant-alpha"
    assert load_raw_config(path)["retriever"]["filters"]["first"]["tenant"] == "${CHERAGH_TEST_TENANT}"


def test_yaml_syntax_error_does_not_print_literal_credentials(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("generation:\n  api_key: [literal-secret-value\n", encoding="utf-8")
    assert main(["validate-config", str(path)]) == 1
    diagnostic = capsys.readouterr()
    assert "Invalid YAML config at line" in diagnostic.err
    assert "literal-secret-value" not in diagnostic.out + diagnostic.err


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("section,field", [
    ("ingestion", "max_file_size_mb"),
    ("retriever", "bm25_k1"),
    ("indexing", "lock_timeout_seconds"),
    ("cache", "ttl"),
    ("cache", "default_ttl"),
    (None, "min_score"),
    (None, "cache_ttl"),
])
def test_non_finite_runtime_limits_are_rejected(section, field, value):
    raw = {section: {field: value}} if section else {field: value}
    with pytest.raises(ValidationError):
        validate_config(raw)


@pytest.mark.parametrize("cache", [None, False, 1, "secret-string", [], [["enabled", True]]])
def test_invalid_cache_sections_retain_typed_redacted_errors(cache):
    with pytest.raises(ValidationError) as failure:
        validate_config({"cache": cache, "cache_backend": "memory"})
    assert failure.value.errors(include_input=False)[0]["loc"] == ("cache",)
    assert "secret-string" not in str(failure.value)


def test_legacy_cache_options_still_migrate_without_mutating_input():
    raw = {"cache_backend": "sqlite", "cache_path": "cache.sqlite", "cache_ttl": 30}
    config = validate_config(raw)
    assert config.cache.enabled
    assert config.cache.backend == "sqlite"
    assert config.cache.path == "cache.sqlite"
    assert config.cache.ttl == 30
    assert "cache" not in raw
