"""Project-wide configuration defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.paths import ProjectPaths, get_paths


@dataclass(frozen=True)
class ProjectConfig:
    """Immutable runtime configuration shared by notebooks and modules.

    Attributes:
        paths: Resolved project directories.
        random_seed: Global RNG seed for reproducible splits and models.
        target_column: Churn label column name in the E Comm sheet.
        id_column: Customer identifier column.
        excel_data_sheet: Sheet containing observations.
        excel_dict_sheet: Sheet containing the feature dictionary.
        test_size: Hold-out fraction for stratified train/test split.
        n_jobs: Parallelism hint for scikit-learn style estimators.
        extra: Optional overrides for later phases (HHO, TabNet, etc.).
    """

    paths: ProjectPaths
    random_seed: int = 42
    target_column: str = "Churn"
    id_column: str = "CustomerID"
    excel_data_sheet: str = "E Comm"
    excel_dict_sheet: str = "Data Dict"
    test_size: float = 0.2
    n_jobs: int = -1
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(
    root: Path | None = None,
    *,
    random_seed: int | None = None,
    **overrides: Any,
) -> ProjectConfig:
    """Build a :class:`ProjectConfig` with optional field overrides.

    Args:
        root: Optional project root override.
        random_seed: Optional seed override (also accepted via ``overrides``).
        **overrides: Additional ``ProjectConfig`` field values.

    Returns:
        Configured :class:`ProjectConfig` instance.
    """
    paths = get_paths(root)
    values: dict[str, Any] = {"paths": paths}
    if random_seed is not None:
        values["random_seed"] = random_seed
    values.update(overrides)
    return ProjectConfig(**values)
