"""Load the e-commerce churn Excel workbook and its sheets."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from src.utils.config import ProjectConfig, load_config
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths, get_paths

logger = get_logger(__name__)

# Original download filename → normalized path under data/raw/
ORIGINAL_DATASET_FILENAME = "E Commerce Dataset.xlsx"
NORMALIZED_DATASET_FILENAME = "E_Commerce_Dataset.xlsx"

DEFAULT_DATA_SHEET = "E Comm"
DEFAULT_DICT_SHEET = "Data Dict"


def resolve_raw_dataset_path(
    path: Path | str | None = None,
    paths: ProjectPaths | None = None,
) -> Path:
    """Resolve the raw Excel path, defaulting to ``data/raw/E_Commerce_Dataset.xlsx``.

    Args:
        path: Explicit file path override.
        paths: Optional :class:`ProjectPaths` (discovered if omitted).

    Returns:
        Absolute path to the workbook.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if path is not None:
        candidate = Path(path).expanduser().resolve()
    else:
        project_paths = paths or get_paths()
        candidate = project_paths.raw_dataset.resolve()

    if not candidate.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {candidate}. "
            f"Expected the file moved from '{ORIGINAL_DATASET_FILENAME}' "
            f"to 'data/raw/{NORMALIZED_DATASET_FILENAME}'."
        )
    return candidate


def load_raw_workbook(
    path: Path | str | None = None,
    *,
    sheet_name: str | list[str] | None = None,
    paths: ProjectPaths | None = None,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """Read the raw Excel workbook.

    Args:
        path: Optional path to the Excel file.
        sheet_name: Sheet selector passed to ``pandas.read_excel``.
            ``None`` loads all sheets as a dict.
        paths: Optional project paths for the default location.

    Returns:
        A DataFrame (single sheet) or mapping of sheet name → DataFrame.
    """
    dataset_path = resolve_raw_dataset_path(path, paths=paths)
    logger.info("Loading workbook from %s", dataset_path)
    return pd.read_excel(dataset_path, sheet_name=sheet_name, engine="openpyxl")


def load_ecommerce_data(
    path: Path | str | None = None,
    *,
    sheet_name: str | None = None,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Load the primary observation sheet (``E Comm`` by default).

    Args:
        path: Optional Excel path override.
        sheet_name: Sheet name override; defaults to config / ``E Comm``.
        config: Optional project config for sheet name and paths.

    Returns:
        DataFrame of customer records.
    """
    cfg = config or load_config()
    sheet = sheet_name or cfg.excel_data_sheet or DEFAULT_DATA_SHEET
    frame = load_raw_workbook(path, sheet_name=sheet, paths=cfg.paths)
    assert isinstance(frame, pd.DataFrame)
    logger.info(
        "Loaded '%s': shape=%s, columns=%d",
        sheet,
        frame.shape,
        frame.shape[1],
    )
    return frame


def load_data_dict(
    path: Path | str | None = None,
    *,
    sheet_name: str | None = None,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Load the feature dictionary sheet (``Data Dict`` by default).

    Args:
        path: Optional Excel path override.
        sheet_name: Sheet name override; defaults to config / ``Data Dict``.
        config: Optional project config for sheet name and paths.

    Returns:
        DataFrame describing feature names and definitions.
    """
    cfg = config or load_config()
    sheet = sheet_name or cfg.excel_dict_sheet or DEFAULT_DICT_SHEET
    frame = load_raw_workbook(path, sheet_name=sheet, paths=cfg.paths)
    assert isinstance(frame, pd.DataFrame)
    logger.info("Loaded '%s': shape=%s", sheet, frame.shape)
    return frame


def load_all_sheets(
    path: Path | str | None = None,
    *,
    config: ProjectConfig | None = None,
) -> Mapping[str, pd.DataFrame]:
    """Load every sheet in the raw workbook as a name → DataFrame mapping."""
    cfg = config or load_config()
    sheets = load_raw_workbook(path, sheet_name=None, paths=cfg.paths)
    assert isinstance(sheets, dict)
    logger.info("Loaded %d sheets: %s", len(sheets), list(sheets.keys()))
    return sheets
