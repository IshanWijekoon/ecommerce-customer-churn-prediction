"""Notebook JSON must parse; CI does not execute cells (no Kaggle data / GPU)."""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

EXPECTED = [
    "1_data_understanding.ipynb",
    "2_data_exploration.ipynb",
    "3_data_preprocessing.ipynb",
    "4_baseline_models.ipynb",
    "5_ga_feature_selection.ipynb",
    "6_pso_feature_selection.ipynb",
    "7_tabnet_model.ipynb",
    "8_prediction_and_insights.ipynb",
]


def test_all_analysis_notebooks_present_and_valid() -> None:
    found = {path.name for path in NOTEBOOKS.glob("*.ipynb")}
    missing = [name for name in EXPECTED if name not in found]
    assert missing == [], f"Missing notebooks: {missing}"

    for name in EXPECTED:
        nb = nbformat.read(NOTEBOOKS / name, as_version=4)
        assert len(nb.cells) > 0, f"{name} has no cells"
        assert any(cell.cell_type == "code" for cell in nb.cells), f"{name} has no code cells"
