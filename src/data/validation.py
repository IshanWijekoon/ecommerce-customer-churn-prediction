"""Schema and quality checks for the e-commerce churn dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Columns expected in the E Comm sheet (UCI / Kaggle e-commerce churn dataset).
EXPECTED_COLUMNS: tuple[str, ...] = (
    "CustomerID",
    "Churn",
    "Tenure",
    "PreferredLoginDevice",
    "CityTier",
    "WarehouseToHome",
    "PreferredPaymentMode",
    "Gender",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "PreferedOrderCat",
    "SatisfactionScore",
    "MaritalStatus",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
)


@dataclass
class ValidationResult:
    """Outcome of schema / quality validation.

    Attributes:
        is_valid: True when no error-level issues were found.
        errors: Blocking problems (missing columns, duplicate IDs when required).
        warnings: Non-blocking observations (missing values, unexpected columns).
        n_rows: Row count of the validated frame.
        n_columns: Column count of the validated frame.
        missing_by_column: Fraction of nulls per column (0–1).
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_columns: int = 0
    missing_by_column: dict[str, float] = field(default_factory=dict)

    def raise_if_invalid(self) -> None:
        """Raise ``ValueError`` if validation failed."""
        if not self.is_valid:
            joined = "; ".join(self.errors)
            raise ValueError(f"Dataset validation failed: {joined}")


def validate_ecommerce_frame(
    df: pd.DataFrame,
    *,
    expected_columns: Sequence[str] = EXPECTED_COLUMNS,
    id_column: str = "CustomerID",
    target_column: str = "Churn",
    require_unique_ids: bool = True,
    allowed_target_values: Sequence[int] | None = (0, 1),
) -> ValidationResult:
    """Validate dtypes presence, ID uniqueness, target domain, and missingness.

    Args:
        df: Observation frame (typically the ``E Comm`` sheet).
        expected_columns: Columns that must be present.
        id_column: Primary key column.
        target_column: Churn label column.
        require_unique_ids: If True, duplicate IDs are errors.
        allowed_target_values: Permitted target values, or ``None`` to skip.

    Returns:
        :class:`ValidationResult` summarizing checks.
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in expected_columns if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    unexpected = [c for c in df.columns if c not in expected_columns]
    if unexpected:
        warnings.append(f"Unexpected columns present: {unexpected}")

    if id_column in df.columns and require_unique_ids:
        n_dupes = int(df[id_column].duplicated().sum())
        if n_dupes > 0:
            errors.append(f"{id_column} has {n_dupes} duplicate value(s)")

    if target_column in df.columns and allowed_target_values is not None:
        unique_targets = set(df[target_column].dropna().unique().tolist())
        allowed = set(allowed_target_values)
        invalid = unique_targets - allowed
        if invalid:
            errors.append(
                f"{target_column} has unexpected values {sorted(invalid)}; "
                f"allowed={sorted(allowed)}"
            )

    missing_by_column = {
        col: float(df[col].isna().mean()) for col in df.columns
    }
    high_missing = {
        col: rate for col, rate in missing_by_column.items() if rate > 0
    }
    if high_missing:
        top = sorted(high_missing.items(), key=lambda x: x[1], reverse=True)[:10]
        summary = ", ".join(f"{c}={r:.1%}" for c, r in top)
        warnings.append(f"Columns with missing values (top): {summary}")

    if df.empty:
        errors.append("DataFrame is empty")

    result = ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        missing_by_column=missing_by_column,
    )

    for warning in result.warnings:
        logger.warning(warning)
    for error in result.errors:
        logger.error(error)
    if result.is_valid:
        logger.info(
            "Validation passed: rows=%d, columns=%d",
            result.n_rows,
            result.n_columns,
        )
    return result


def summarize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact dtype / cardinality / null summary for profiling."""
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "n_unique": int(series.nunique(dropna=True)),
                "n_missing": int(series.isna().sum()),
                "pct_missing": float(series.isna().mean()),
            }
        )
    return pd.DataFrame(rows)
