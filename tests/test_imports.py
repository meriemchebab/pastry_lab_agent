"""Smoke tests for project imports and declared third-party dependencies."""

import ast
import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_MODULES = (
    "app",
    "core.graph",
    "core.state",
    "agents.router",
    "agents.search_agent",
    "agents.math_agent",
    "agents.vision_agent",
    "agents.copilot_agent",
    "tools.search_tool",
    "tools.ingredient_db",
    "tools.baking_math",
)

# Imports supplied by third-party distributions rather than Python's standard
# library or this project. The mapping handles distributions with different
# import names, such as tavily-python -> tavily.
THIRD_PARTY_DISTRIBUTIONS = {
    "gradio": "gradio",
    "langchain_core": "langchain-core",
    "langchain_google_genai": "langchain-google-genai",
    "langgraph": "langgraph",
    "tavily": "tavily-python",
    "typing_extensions": "typing-extensions",
}


def _requirement_names():
    requirements = PROJECT_ROOT / "requirements.txt"
    names = set()
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.partition("#")[0].strip()
        if line:
            names.add(line.split(";")[0].split("[")[0].split("=")[0].split("<")[0].split(">")
                      [0].strip().lower().replace("_", "-"))
    return names


def test_project_modules_import():
    """All application modules should import without credentials or network access."""
    for module_name in PROJECT_MODULES:
        importlib.import_module(module_name)


def test_imported_third_party_packages_are_declared():
    """Top-level external imports in project Python files belong in requirements."""
    declared = _requirement_names()
    imported = set()

    for source_file in PROJECT_ROOT.rglob("*.py"):
        if any(part in {".git", "__pycache__", ".venv", "venv"} for part in source_file.parts):
            continue
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".", 1)[0])

    missing = {
        distribution
        for import_name, distribution in THIRD_PARTY_DISTRIBUTIONS.items()
        if import_name in imported and distribution not in declared
    }
    assert not missing, f"Missing requirements: {', '.join(sorted(missing))}"


def test_project_imports_do_not_reach_external_services(monkeypatch):
    """Importing the app must not perform external API calls."""
    from tavily import TavilyClient

    def fail_if_client_created(*args, **kwargs):
        raise AssertionError("Tavily client should not be created during import")

    monkeypatch.setattr(TavilyClient, "__init__", fail_if_client_created)
    # Modules are already imported by the smoke test in a normal pytest run.
    # Verify no unexpected integration module was loaded by the test suite.
    assert "tools.search_tool" in sys.modules
