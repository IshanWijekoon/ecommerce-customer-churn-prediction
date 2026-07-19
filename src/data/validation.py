"""Schema and quality checks for the e-commerce churn dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

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

# Enriched business definitions (source sheet text is terse / has typos).
FEATURE_BUSINESS_DEFINITIONS: dict[str, str] = {
    "CustomerID": (
        "Unique customer primary key. Used for joins and duplicate checks; "
        "dropped before model training."
    ),
    "Churn": (
        "Binary churn flag (1 = churned, 0 = retained). Positive class for "
        "classification; expected prevalence ~17%."
    ),
    "Tenure": (
        "Months the customer has been with the platform. Proxy for loyalty "
        "and switching cost; low tenure often signals higher churn risk."
    ),
    "PreferredLoginDevice": (
        "Device used most often to access the storefront (Computer, Mobile Phone, "
        "Phone). Note: 'Phone' and 'Mobile Phone' may be overlapping labels."
    ),
    "CityTier": (
        "Ordinal city development tier (1-3). Captures urbanisation / market "
        "context that can affect fulfilment expectations and churn."
    ),
    "WarehouseToHome": (
        "Distance from fulfilment warehouse to the customer's home. Longer "
        "distances may imply slower delivery and weaker retention."
    ),
    "PreferredPaymentMode": (
        "Preferred payment method. Note: 'CC'/'Credit Card' and "
        "'COD'/'Cash on Delivery' appear as near-duplicates in the raw data."
    ),
    "Gender": "Self-reported gender (Female / Male).",
    "HourSpendOnApp": (
        "Hours spent on the app/website. Engagement proxy; very low usage "
        "may precede churn."
    ),
    "NumberOfDeviceRegistered": (
        "Count of devices registered to the account. Multi-device usage can "
        "indicate deeper product adoption."
    ),
    "PreferedOrderCat": (
        "Preferred order category in the last month (source spelling). "
        "'Mobile' and 'Mobile Phone' may need harmonisation in preprocessing."
    ),
    "SatisfactionScore": (
        "Ordinal service satisfaction (1-5). Low scores plus complaints are "
        "strong retention-risk signals."
    ),
    "MaritalStatus": "Marital status (Single / Married / Divorced).",
    "NumberOfAddress": (
        "Number of addresses saved on the account. May relate to household "
        "complexity or account maturity."
    ),
    "Complain": (
        "Whether a complaint was raised in the last month (1 = yes). High-cost "
        "churn driver when paired with low satisfaction."
    ),
    "OrderAmountHikeFromlastYear": (
        "Year-over-year percentage increase in order amount. Rising spend "
        "suggests healthy engagement; flat/missing may need careful imputation."
    ),
    "CouponUsed": (
        "Coupons redeemed in the last month. Heavy coupon reliance can indicate "
        "price sensitivity."
    ),
    "OrderCount": (
        "Orders placed in the last month. Frequency signal for RFM-style "
        "behavioural analysis."
    ),
    "DaySinceLastOrder": (
        "Days since the most recent order (recency). High values are a classic "
        "churn leading indicator."
    ),
    "CashbackAmount": (
        "Average cashback received last month. Incentive exposure that may "
        "affect retention elasticity."
    ),
}

# Likely overlapping category labels observed in the raw workbook.
KNOWN_CATEGORY_OVERLAPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "PreferredLoginDevice": (("Phone", "Mobile Phone"),),
    "PreferredPaymentMode": (
        ("CC", "Credit Card"),
        ("COD", "Cash on Delivery"),
    ),
    "PreferedOrderCat": (("Mobile", "Mobile Phone"),),
}


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON / interim metadata sidecars."""
        return asdict(self)


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


def parse_data_dictionary(raw_dict: pd.DataFrame) -> pd.DataFrame:
    """Normalize the messy ``Data Dict`` Excel sheet into clean columns.

    The source sheet uses unnamed headers and a title row; this returns
    ``variable``, ``source_description``, and ``sheet`` columns.
    """
    work = raw_dict.copy()
    # Drop fully empty columns (leading Unnamed index-like columns).
    work = work.dropna(axis=1, how="all")
    if work.shape[1] < 3:
        raise ValueError(
            f"Data Dict sheet has unexpected shape {raw_dict.shape}; "
            "expected at least sheet / variable / description columns."
        )

    cols = list(work.columns)
    # Typical layout after dropping empties: Data | Variable | Discerption
    sheet_col, var_col, desc_col = cols[0], cols[1], cols[2]
    rows = work[[sheet_col, var_col, desc_col]].copy()
    rows.columns = ["sheet", "variable", "source_description"]

    # Drop header/title rows where variable is literally "Variable" or null.
    rows = rows.dropna(subset=["variable"])
    rows = rows[rows["variable"].astype(str).str.strip().str.lower() != "variable"]
    rows["variable"] = rows["variable"].astype(str).str.strip()
    rows["source_description"] = (
        rows["source_description"].astype(str).str.strip().replace({"nan": ""})
    )
    rows["sheet"] = rows["sheet"].astype(str).str.strip()
    return rows.reset_index(drop=True)


def build_feature_dictionary(raw_dict: pd.DataFrame) -> pd.DataFrame:
    """Merge source Data Dict text with enriched business definitions.

    Args:
        raw_dict: Raw ``Data Dict`` sheet as loaded from Excel.

    Returns:
        Feature dictionary with source + business definitions and role tags.
    """
    parsed = parse_data_dictionary(raw_dict)
    parsed["business_definition"] = parsed["variable"].map(
        lambda v: FEATURE_BUSINESS_DEFINITIONS.get(v, "")
    )

    def _role(name: str) -> str:
        if name == "CustomerID":
            return "id"
        if name == "Churn":
            return "target"
        return "feature"

    def _inferred_type(name: str) -> str:
        nominal = {
            "PreferredLoginDevice",
            "PreferredPaymentMode",
            "Gender",
            "PreferedOrderCat",
            "MaritalStatus",
        }
        ordinal = {"CityTier", "SatisfactionScore"}
        binary = {"Churn", "Complain"}
        if name == "CustomerID":
            return "id"
        if name in binary:
            return "binary"
        if name in ordinal:
            return "ordinal"
        if name in nominal:
            return "nominal"
        return "numeric"

    parsed["role"] = parsed["variable"].map(_role)
    parsed["inferred_type"] = parsed["variable"].map(_inferred_type)
    return parsed


def target_class_balance(
    df: pd.DataFrame,
    *,
    target_column: str = "Churn",
) -> pd.DataFrame:
    """Return absolute and relative class counts for the churn target."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found")

    counts = df[target_column].value_counts(dropna=False).sort_index()
    props = df[target_column].value_counts(normalize=True, dropna=False).sort_index()
    out = pd.DataFrame(
        {
            "class": counts.index.astype(object),
            "count": counts.to_numpy(),
            "proportion": props.to_numpy(),
        }
    )
    out["label"] = out["class"].map({0: "Retained", 1: "Churned"}).fillna("Unknown")
    return out.reset_index(drop=True)


def list_data_quality_issues(
    df: pd.DataFrame,
    *,
    id_column: str = "CustomerID",
    target_column: str = "Churn",
) -> list[dict[str, str]]:
    """Compile a structured list of known / observed data-quality issues.

    Returns:
        List of ``{severity, category, detail}`` dicts for notebook reporting.
    """
    issues: list[dict[str, str]] = []

    missing = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if int(df[col].isna().sum()) > 0
    }
    if missing:
        detail = ", ".join(
            f"{c}={n} ({df[c].isna().mean():.1%})"
            for c, n in sorted(missing.items(), key=lambda x: -x[1])
        )
        issues.append(
            {
                "severity": "medium",
                "category": "missing_values",
                "detail": (
                    f"{len(missing)} numeric feature(s) contain nulls: {detail}. "
                    "Imputation policy deferred to preprocessing (Phase 3)."
                ),
            }
        )

    n_dupes = (
        int(df[id_column].duplicated().sum()) if id_column in df.columns else 0
    )
    if n_dupes > 0:
        issues.append(
            {
                "severity": "high",
                "category": "duplicate_ids",
                "detail": f"{id_column} has {n_dupes} duplicate value(s).",
            }
        )
    else:
        issues.append(
            {
                "severity": "info",
                "category": "duplicate_ids",
                "detail": f"{id_column} values are unique (no duplicates).",
            }
        )

    for col, groups in KNOWN_CATEGORY_OVERLAPS.items():
        if col not in df.columns:
            continue
        present = set(df[col].dropna().astype(str).unique())
        for group in groups:
            overlap = [v for v in group if v in present]
            if len(overlap) >= 2:
                issues.append(
                    {
                        "severity": "medium",
                        "category": "category_overlap",
                        "detail": (
                            f"{col} contains overlapping labels {overlap}; "
                            "harmonise in preprocessing to avoid split signal."
                        ),
                    }
                )

    if "PreferedOrderCat" in df.columns:
        issues.append(
            {
                "severity": "low",
                "category": "naming",
                "detail": (
                    "Column 'PreferedOrderCat' retains source spelling "
                    "(missing 'r' in Preferred); keep as-is for schema "
                    "compatibility, document in dictionary."
                ),
            }
        )

    if target_column in df.columns:
        rate = float(df[target_column].mean())
        issues.append(
            {
                "severity": "info",
                "category": "class_imbalance",
                "detail": (
                    f"Churn prevalence is {rate:.1%}. Prefer Recall / F1 / "
                    "PR-AUC over Accuracy; apply class weights or "
                    "scale_pos_weight in baselines."
                ),
            }
        )

    return issues


def profile_frame(
    df: pd.DataFrame,
    *,
    id_column: str = "CustomerID",
    target_column: str = "Churn",
) -> dict[str, Any]:
    """Aggregate shape, dtype summary, target balance, and DQ issues."""
    balance = target_class_balance(df, target_column=target_column)
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtype_summary": summarize_dtypes(df),
        "target_balance": balance,
        "churn_rate": float(df[target_column].mean())
        if target_column in df.columns
        else None,
        "n_duplicate_ids": int(df[id_column].duplicated().sum())
        if id_column in df.columns
        else None,
        "data_quality_issues": list_data_quality_issues(
            df, id_column=id_column, target_column=target_column
        ),
    }
