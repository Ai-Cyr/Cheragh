"""Diagnostics must not turn invalid provider configuration into a secret log."""

import json

import pytest
from pydantic import ValidationError

from cheragh.cli.main import _redact_config_secrets, main
from cheragh.config.schema import validate_config


def test_model_validation_error_omits_secret_input_but_preserves_reason():
    with pytest.raises(ValidationError) as failure:
        validate_config({"embedding": {"api_key": "s3cr3t", "provider": "azure"}})
    diagnostic = str(failure.value)
    assert "s3cr3t" not in diagnostic
    assert "input_value" not in diagnostic
    assert "embedding.model is required" in diagnostic


def test_index_config_error_does_not_log_resolved_environment_secret(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHERAGH_TEST_KEY", "s3cr3t")
    config = tmp_path / "rag.json"
    config.write_text(json.dumps({"embedding": {"api_key": "${CHERAGH_TEST_KEY}", "provider": "azure"}}))
    assert main(["index", "--config", str(config)]) == 2
    diagnostic = capsys.readouterr()
    assert "s3cr3t" not in diagnostic.out + diagnostic.err
    assert "embedding.model is required" in diagnostic.err


@pytest.mark.parametrize("url", [
    "https://operator:s3cr3t@example.invalid:6333",
    "https://s3cr3t@example.invalid:6333",
    "https://example.invalid:6333?api_key=s3cr3t",
    "https://example.invalid:6333?%61ccess_token=s3cr3t",
    "https://example.invalid:6333?X-API-Key=s3cr3t",
    "https://example.invalid:6333?apiKey=s3cr3t",
    "https://example.invalid:6333?X-Amz-Signature=s3cr3t",
    "https://example.invalid:6333?X-Amz-Credential=s3cr3t",
    "https://example.invalid:6333?sig=s3cr3t",
])
def test_validated_config_redacts_credential_bearing_urls(tmp_path, capsys, url):
    config = tmp_path / "rag.json"
    config.write_text(json.dumps({"vectorstore": {"type": "qdrant", "url": url}}))
    assert main(["validate-config", str(config), "--json"]) == 0
    output = capsys.readouterr().out
    assert "s3cr3t" not in output
    assert json.loads(output)["vectorstore"]["url"] == "***"
    # Redaction is only a display operation; do not change the runtime URL.
    assert validate_config(json.loads(config.read_text())).vectorstore.url == url


def test_redaction_preserves_benign_urls_and_does_not_mutate_input():
    value = {"url": "https://example.invalid:6333/path?timeout=10", "nested": [{"api_key": "s3cr3t"}]}
    redacted = _redact_config_secrets(value)
    assert redacted["url"] == value["url"]
    assert redacted["nested"][0]["api_key"] == "***"
    assert value["nested"][0]["api_key"] == "s3cr3t"
