from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from cheragh.cache import build_cache_backend
from cheragh.cli.main import main as cli_main
from cheragh.config import validate_config
from cheragh.tracing import RAGTrace


class ConfigErrorRedactionTests(unittest.TestCase):
    def test_validate_config_never_prints_invalid_secret_input(self):
        secret = "cli-validation-super-secret"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"generation": {"api_key": {"secret": secret}}}),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = cli_main(["validate-config", str(path)])

        rendered = stderr.getvalue()
        errors = json.loads(rendered)
        self.assertEqual(code, 1)
        self.assertIsInstance(errors, list)
        self.assertEqual(errors[0]["loc"], ["generation", "api_key"])
        self.assertNotIn(secret, rendered)
        self.assertNotIn("input_value", rendered)
        self.assertNotIn("input_url", rendered)


class CacheBooleanParsingTests(unittest.TestCase):
    def test_boolean_strings_use_a_strict_whitelist(self):
        self.assertIsNone(build_cache_backend({"enabled": "false"}))
        self.assertIsNotNone(
            build_cache_backend(
                {
                    "enabled": "yes",
                    "backend": "memory",
                    "allow_pickle": "off",
                    "allow_unsigned_pickle": "0",
                }
            )
        )

    def test_boolean_typos_and_non_boolean_values_are_rejected(self):
        invalid_configs = [
            {"enabled": "flase"},
            {"enabled": 1},
            {"allow_pickle": "flase"},
            {"allow_pickle": None},
            {"allow_unsigned_pickle": "flase"},
            {"allow_unsigned_pickle": []},
        ]
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaisesRegex(
                ValueError, "must be a boolean"
            ):
                build_cache_backend(config)

    def test_unsigned_pickle_typo_cannot_enable_unsafe_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "cache.allow_unsigned_pickle must be a boolean"
        ):
            build_cache_backend(
                {
                    "backend": "sqlite",
                    "path": str(Path(tmp) / "cache.sqlite"),
                    "serializer": "pickle",
                    "allow_pickle": True,
                    "allow_unsigned_pickle": "flase",
                }
            )


class PricingValidationTests(unittest.TestCase):
    def test_pricing_config_is_typed_and_normalized(self):
        config = validate_config(
            {
                "observability": {
                    "pricing": {
                        "input_per_1k": 1,
                        "output_per_1k": 0.25,
                        "currency": " EUR ",
                    }
                }
            }
        )
        pricing = config.observability.pricing
        self.assertIsNotNone(pricing)
        assert pricing is not None
        self.assertEqual(pricing.input_per_1k, 1.0)
        self.assertEqual(pricing.output_per_1k, 0.25)
        self.assertEqual(pricing.currency, "EUR")

    def test_pricing_config_rejects_unsafe_values_and_unknown_keys(self):
        invalid_pricing = [
            {"input_per_1k": True},
            {"input_per_1k": "0.1"},
            {"input_per_1k": -0.1},
            {"input_per_1k": math.nan},
            {"output_per_1k": math.inf},
            {"currency": ""},
            {"currency": "   "},
            {"currency": 123},
            {"per_request": 1.0},
            None,
        ]
        for pricing in invalid_pricing:
            with self.subTest(pricing=pricing), self.assertRaises(ValidationError):
                validate_config({"observability": {"pricing": pricing}})

    def test_record_generation_validates_direct_api_pricing_before_mutation(self):
        invalid_pricing = [
            {"input_per_1k": True},
            {"input_per_1k": "0.1"},
            {"input_per_1k": -0.1},
            {"input_per_1k": math.nan},
            {"output_per_1k": math.inf},
            {"currency": "   "},
            {"unknown": 1.0},
            [],
        ]
        for pricing in invalid_pricing:
            trace = RAGTrace()
            with self.subTest(pricing=pricing), self.assertRaises((TypeError, ValueError)):
                trace.record_generation(
                    prompt="prompt",
                    answer="answer",
                    pricing=pricing,  # type: ignore[arg-type]
                )
            self.assertEqual(trace.token_usage, {})
            self.assertEqual(trace.cost, {})

    def test_record_generation_normalizes_valid_direct_api_pricing(self):
        trace = RAGTrace()
        trace.record_generation(
            prompt="abcd",
            answer="abcdefgh",
            pricing={
                "input_per_1k": 1,
                "output_per_1k": 2.0,
                "currency": " EUR ",
            },
        )
        self.assertEqual(trace.cost["currency"], "EUR")
        self.assertEqual(trace.cost["input_cost_estimated"], 0.001)
        self.assertEqual(trace.cost["output_cost_estimated"], 0.004)


if __name__ == "__main__":
    unittest.main()
