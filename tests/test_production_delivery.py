from __future__ import annotations

import re
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


class ProductionPackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        cls.pyproject = tomllib.loads(cls.pyproject_text)

    def test_build_backend_inputs_are_exactly_pinned(self) -> None:
        requirements = self.pyproject["build-system"]["requires"]
        self.assertGreaterEqual(len(requirements), 2)
        self.assertTrue(all("==" in requirement for requirement in requirements))
        self.assertEqual("setuptools.build_meta", self.pyproject["build-system"]["build-backend"])

    def test_runtime_dependencies_have_major_version_guards(self) -> None:
        project = self.pyproject["project"]
        for requirement in project["dependencies"]:
            self.assertIn("<", requirement, requirement)
        for extra, requirements in project["optional-dependencies"].items():
            if extra == "dev":
                continue
            for requirement in requirements:
                self.assertIn("<", requirement, f"{extra}: {requirement}")

    def test_distribution_metadata_is_complete(self) -> None:
        project = self.pyproject["project"]
        self.assertEqual(["LICENSE"], project["license-files"])
        self.assertIn("Typing :: Typed", project["classifiers"])
        self.assertEqual(
            "https://github.com/Ai-Cyr/Cheragh",
            project["urls"]["Repository"],
        )
        self.assertEqual(["py.typed"], self.pyproject["tool"]["setuptools"]["package-data"]["cheragh"])


class ProductionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((ROOT / ".github" / "workflows").glob("*.yml"))
        )

    def test_actions_are_immutable_and_checkout_drops_credentials(self) -> None:
        use_lines = re.findall(r"^\s*uses:\s*(\S+)(?:\s+#.*)?$", self.workflow, flags=re.MULTILINE)
        self.assertTrue(use_lines)
        for use in use_lines:
            self.assertRegex(use, r"^[^@]+@[0-9a-f]{40}$")
        checkout_count = sum(use.startswith("actions/checkout@") for use in use_lines)
        self.assertEqual(checkout_count, self.workflow.count("persist-credentials: false"))

    def test_workflow_has_least_privilege_and_bounded_execution(self) -> None:
        self.assertIn("permissions:\n  contents: read", self.workflow)
        self.assertNotIn("pull_request_target:", self.workflow)
        self.assertIn("concurrency:", self.workflow)
        self.assertIn("cancel-in-progress: true", self.workflow)
        job_count = self.workflow.count("runs-on: ubuntu-24.04")
        self.assertGreaterEqual(job_count, 4)
        self.assertEqual(job_count, self.workflow.count("timeout-minutes:"))
        self.assertNotIn("runs-on: ubuntu-latest", self.workflow)

    def test_artifacts_and_dependencies_are_checked(self) -> None:
        self.assertIn("python -m twine check dist/*", self.workflow)
        self.assertIn("Smoke-test wheel", self.workflow)
        self.assertIn("Smoke-test source distribution", self.workflow)
        self.assertIn("python -m pip check", self.workflow)
        self.assertIn('"pip-audit==2.10.1"', self.workflow)
        self.assertIn("python -m pip_audit --strict", self.workflow)
        self.assertIn("actions/dependency-review-action@", self.workflow)
        self.assertIn("fail-on-severity: high", self.workflow)
        self.assertIn("docker build --pull --tag cheragh:ci .", self.workflow)
        self.assertIn("docker compose config --quiet", self.workflow)


class ProductionContainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        cls.dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        cls.compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    def test_runtime_is_a_minimal_multi_stage_wheel_install(self) -> None:
        self.assertIn("AS builder", self.dockerfile)
        self.assertIn("AS runtime", self.dockerfile)
        self.assertIn("python -m build --wheel --no-isolation", self.dockerfile)
        runtime = self.dockerfile.split("AS runtime", maxsplit=1)[1]
        self.assertNotIn("COPY src", runtime)
        self.assertIn("COPY --from=builder /wheels", runtime)
        self.assertIn("/opt/venv/bin/python -m pip check", runtime)

    def test_builder_installs_the_exact_pep517_requirements(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        for requirement in pyproject["build-system"]["requires"]:
            self.assertIn(f'"{requirement}"', self.dockerfile)

    def test_runtime_is_non_root_and_has_process_guards(self) -> None:
        self.assertIn("adduser --system --uid 10001", self.dockerfile)
        self.assertIn("USER 10001:10001", self.dockerfile)
        self.assertIn("HEALTHCHECK", self.dockerfile)
        self.assertIn("http://127.0.0.1:8000/ready", self.dockerfile)
        self.assertIn("STOPSIGNAL SIGTERM", self.dockerfile)
        self.assertIn("CHERAGH_ENABLE_INDEXING=false", self.dockerfile)
        self.assertIn("CHERAGH_REQUIRE_AUTH=true", self.dockerfile)
        self.assertNotIn("CHERAGH_API_KEY=", self.dockerfile)
        self.assertIn("ARG CHERAGH_EXTRAS=fastapi,config", self.dockerfile)

    def test_build_context_excludes_secrets_and_development_data(self) -> None:
        ignored = set(self.dockerignore.splitlines())
        for pattern in {".git", ".env", ".env.*", "*.pem", "*.key", "data", "tests"}:
            self.assertIn(pattern, ignored)

    def test_compose_is_secure_by_default_and_does_not_fake_qdrant_usage(self) -> None:
        self.assertIn('"127.0.0.1:${CHERAGH_PORT:-8000}:8000"', self.compose)
        self.assertIn("${CHERAGH_API_KEY:?CHERAGH_API_KEY must be set}", self.compose)
        self.assertIn("read_only: true", self.compose)
        self.assertIn("./data:/data:ro", self.compose)
        self.assertIn("no-new-privileges:true", self.compose)
        self.assertIn("pids_limit: 256", self.compose)
        self.assertIn("mem_limit: 2g", self.compose)
        self.assertIn('cpus: "2.0"', self.compose)
        self.assertNotIn("\n  qdrant:", self.compose)


class DependencyMaintenanceTests(unittest.TestCase):
    def test_dependabot_covers_python_actions_and_container_pins(self) -> None:
        config = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
        for ecosystem in {"pip", "github-actions", "docker"}:
            self.assertIn(f"package-ecosystem: {ecosystem}", config)
        self.assertEqual(3, config.count("interval: weekly"))
        self.assertNotIn("interval: daily", config)


class ProductionDocumentationTests(unittest.TestCase):
    def test_operational_topics_are_documented(self) -> None:
        guide = (ROOT / "docs" / "production.md").read_text(encoding="utf-8")
        expected_sections = {
            "## Construire et vérifier le paquet",
            "## Frontière HTTP, TLS et authentification",
            "## Secrets et données sensibles",
            "## Dimensionnement et montée en charge",
            "## Sauvegardes et restauration",
            "## Gate qualité avant déploiement",
            "## Checklist de mise en production",
        }
        for section in expected_sections:
            self.assertIn(section, guide)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
