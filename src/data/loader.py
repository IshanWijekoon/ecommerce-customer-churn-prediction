"""Load the e-commerce churn Excel workbook and its sheets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.data.validation import ValidationResult, validate_ecommerce_frame
from src.utils.config import ProjectConfig, load_config
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths, get_paths

logger = get_logger(__name__)

# Original download filename → normalized path under data/raw/
ORIGINAL_DATASET_FILENAME = "E Commerce Dataset.xlsx"
NORMALIZED_DATASET_FILENAME = "E_Commerce_Dataset.xlsx"

DEFAULT_DATA_SHEET = "E Comm"
DEFAULT_DICT_SHEET = "Data Dict"

DEFAULT_INTERIM_STEM = "ecomm_validated"


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


def load_and_validate(
    path: Path | str | None = None,
    *,
    config: ProjectConfig | None = None,
    raise_on_invalid: bool = True,
) -> tuple[pd.DataFrame, ValidationResult]:
    """Load the ``E Comm`` sheet and run schema / quality validation.

    Args:
        path: Optional Excel path override.
        config: Optional project config.
        raise_on_invalid: If True, raise when validation fails.

    Returns:
        Tuple of ``(dataframe, ValidationResult)``.
    """
    cfg = config or load_config()
    df = load_ecommerce_data(path, config=cfg)
    result = validate_ecommerce_frame(
        df,
        id_column=cfg.id_column,
        target_column=cfg.target_column,
    )
    if raise_on_invalid:
        result.raise_if_invalid()
    return df, result


def _write_table(df: pd.DataFrame, path: Path) -> Path:
    """Persist a DataFrame as parquet (preferred) or CSV fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return path

    parquet_path = path if path.suffix.lower() == ".parquet" else path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ValueError) as exc:
        csv_path = parquet_path.with_suffix(".csv")
        logger.warning(
            "Parquet write failed (%s); falling back to CSV at %s",
            exc,
            csv_path,
        )
        df.to_csv(csv_path, index=False)
        return csv_path


def save_interim_snapshot(
    df: pd.DataFrame,
    *,
    validation: ValidationResult | None = None,
    config: ProjectConfig | None = None,
    stem: str = DEFAULT_INTERIM_STEM,
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write a validated interim table + JSON metadata under ``data/interim/``.

    Args:
        df: Validated observation frame (no train/test split yet).
        validation: Optional validation result embedded in the sidecar.
        config: Project config for paths.
        stem: Filename stem (default ``ecomm_validated``).
        extra_metadata: Optional additional JSON-serializable fields.

    Returns:
        ``(table_path, metadata_path)``.
    """
    cfg = config or load_config()
    cfg.paths.ensure_directories()
    interim = cfg.paths.interim

    table_path = _write_table(df, interim / f"{stem}.parquet")
    meta: dict[str, Any] = {
        "stem": stem,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "table_file": table_path.name,
        "source_raw": str(cfg.paths.raw_dataset.name),
        "id_column": cfg.id_column,
        "target_column": cfg.target_column,
        "random_seed": cfg.random_seed,
    }
    if validation is not None:
        meta["validation"] = validation.to_dict()
    if extra_metadata:
        meta["extra"] = dict(extra_metadata)

    meta_path = interim / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved interim snapshot: %s (+ %s)", table_path, meta_path.name)
    return table_path, meta_path


def load_interim_snapshot(
    *,
    config: ProjectConfig | None = None,
    stem: str = DEFAULT_INTERIM_STEM,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a previously saved interim snapshot and its metadata.

    Raises:
        FileNotFoundError: If neither parquet nor CSV snapshot exists.
    """
    cfg = config or load_config()
    interim = cfg.paths.interim
    parquet_path = interim / f"{stem}.parquet"
    csv_path = interim / f"{stem}.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        table_path = parquet_path
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        table_path = csv_path
    else:
        raise FileNotFoundError(
            f"No interim snapshot found for stem '{stem}' under {interim}"
        )

    meta_path = interim / f"{stem}_meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("table_file", table_path.name)
    logger.info("Loaded interim snapshot from %s", table_path)
    return df, meta
