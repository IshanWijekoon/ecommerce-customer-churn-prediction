"""Repo hygiene checks that do not need the modelling stack."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECRET_MARKERS = (
    "OPENROUTER_API_KEY=" + "sk-",
    "sk-or-" + "v1-",
)


def test_readme_and_license_exist() -> None:
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / "requirements-dev.txt").is_file()


def test_tracked_files_do_not_embed_openrouter_keys() -> None:
    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".ruff_cache",
        ".pytest_cache",
        ".cursor",
    }
    text_suffixes = {".py", ".md", ".yml", ".yaml", ".txt", ".toml", ".json", ".ipynb", ".example"}
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() not in text_suffixes and path.name != ".env.example":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for marker in SECRET_MARKERS:
            if marker in text:
                hits.append(f"{path.relative_to(ROOT)} contains {marker}")
    assert hits == [], "Possible secrets committed:\n" + "\n".join(hits)
