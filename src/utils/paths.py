"""Project path resolution relative to the repository root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` until ``environment.yml`` or ``.git`` is found.

    Args:
        start: Directory to begin the search. Defaults to this file's parents.

    Returns:
        Absolute path to the project root.

    Raises:
        FileNotFoundError: If no project root marker is found.
    """
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "environment.yml").exists() or (candidate / ".git").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate project root (looked for environment.yml or .git)."
    )


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical directories used across notebooks, scripts, and the dashboard."""

    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def external(self) -> Path:
        return self.data / "external"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def notebooks(self) -> Path:
        return self.root / "notebooks"

    @property
    def dashboard(self) -> Path:
        return self.root / "dashboard"

    @property
    def src(self) -> Path:
        return self.root / "src"

    @property
    def raw_dataset(self) -> Path:
        """Normalized raw Excel path (original: ``E Commerce Dataset.xlsx``)."""
        return self.raw / "E_Commerce_Dataset.xlsx"

    def ensure_directories(self) -> None:
        """Create data, model, and report directories if they do not exist."""
        for path in (
            self.raw,
            self.interim,
            self.processed,
            self.external,
            self.models,
            self.reports,
            self.figures,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_paths(root: Path | None = None) -> ProjectPaths:
    """Return a :class:`ProjectPaths` instance for ``root`` or the discovered root."""
    return ProjectPaths(root=root.resolve() if root else find_project_root())
