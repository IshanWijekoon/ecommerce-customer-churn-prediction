"""
Apply the notebook-3 transform to raw customer rows for live scoring.

Uses ``models/preprocess.joblib`` (scaler, fill values, dummy columns, feature
order) so form input matches the matrices the saved models were trained on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESS_PATH = ROOT / "models" / "preprocess.joblib"

# Same alias maps as notebook 3 / scripts/export_scoring_artifacts.py
LOGIN_ALIASES = {"Phone": "Mobile Phone"}
PAYMENT_ALIASES = {"CC": "Credit Card", "COD": "Cash on Delivery"}
ORDER_CAT_ALIASES = {"Mobile": "Mobile Phone"}


def load_preprocess_bundle(path: Path | str | None = None) -> dict[str, Any]:
    """Load the fitted preprocess bundle saved by notebook 3."""
    bundle_path = Path(path) if path is not None else DEFAULT_PREPROCESS_PATH
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing preprocess bundle at {bundle_path}. "
            "Run notebook 3 or scripts/export_scoring_artifacts.py first."
        )
    return joblib.load(bundle_path)


def harmonise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate category spellings (Phone→Mobile Phone, CC→Credit Card, …)."""
    out = df.copy()
    if "PreferredLoginDevice" in out.columns:
        out["PreferredLoginDevice"] = out["PreferredLoginDevice"].replace(LOGIN_ALIASES)
    if "PreferredPaymentMode" in out.columns:
        out["PreferredPaymentMode"] = out["PreferredPaymentMode"].replace(PAYMENT_ALIASES)
    if "PreferedOrderCat" in out.columns:
        out["PreferedOrderCat"] = out["PreferedOrderCat"].replace(ORDER_CAT_ALIASES)
    return out


def engineer_pre_split(df: pd.DataFrame) -> pd.DataFrame:
    """Create TenureBin and UnhappyComplain (same cut points as notebook 3)."""
    out = df.copy()
    out["TenureBin"] = pd.cut(
        out["Tenure"],
        bins=[-0.1, 3, 9, 15, 1e9],
        labels=[0, 1, 2, 3],
    ).astype("float")
    out["UnhappyComplain"] = (
        (out["Complain"].fillna(0) == 1) & (out["SatisfactionScore"].fillna(3) <= 2)
    ).astype(int)
    return out


def transform_raw_customers(
    raw: pd.DataFrame | dict[str, Any],
    bundle: dict[str, Any] | None = None,
    *,
    preprocess_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Turn one or more raw Kaggle-style customer rows into the 33-feature matrix.

    Accepts a DataFrame or a single-customer dict. Drops ``CustomerID`` / ``Churn``
    if present. Returns a DataFrame with columns in ``bundle['feature_names']`` order.
    """
    if bundle is None:
        bundle = load_preprocess_bundle(preprocess_path)

    if isinstance(raw, dict):
        frame = pd.DataFrame([raw])
    else:
        frame = raw.copy()

    drop_cols = [c for c in ("CustomerID", "Churn") if c in frame.columns]
    if drop_cols:
        frame = frame.drop(columns=drop_cols)

    frame = engineer_pre_split(harmonise_labels(frame))

    numeric_columns: list[str] = list(bundle["numeric_columns"])
    category_columns: list[str] = list(bundle["category_columns"])
    continuous_columns: list[str] = list(bundle["continuous_columns"])
    ordinal_and_flags: list[str] = list(bundle["ordinal_and_flags"])
    dummy_columns: list[str] = list(bundle["dummy_columns"])
    feature_names: list[str] = list(bundle["feature_names"])
    numeric_fill: dict[str, Any] = dict(bundle["numeric_fill_values"])
    category_fill: dict[str, Any] = dict(bundle["category_fill_values"])
    recency_threshold = float(bundle["recency_threshold"])
    scaler = bundle["scaler"]

    for col in numeric_columns:
        if col not in frame.columns:
            frame[col] = pd.NA
        frame[col] = frame[col].fillna(numeric_fill[col])

    for col in category_columns:
        if col not in frame.columns:
            frame[col] = pd.NA
        frame[col] = frame[col].fillna(category_fill[col])

    frame["IsDormant"] = (frame["DaySinceLastOrder"] > recency_threshold).astype(int)

    # Recompute UnhappyComplain after fill so missing Complain/Satisfaction match train
    frame["UnhappyComplain"] = (
        (frame["Complain"] == 1) & (frame["SatisfactionScore"] <= 2)
    ).astype(int)

    num_part = frame[continuous_columns + ordinal_and_flags].copy()
    cat_part = pd.get_dummies(frame[category_columns], dtype=int)
    cat_part = cat_part.reindex(columns=dummy_columns, fill_value=0)

    scaled = pd.DataFrame(
        scaler.transform(num_part[continuous_columns]),
        columns=continuous_columns,
        index=frame.index,
    )
    final = pd.concat(
        [scaled, num_part[ordinal_and_flags], cat_part],
        axis=1,
    )
    return final.reindex(columns=feature_names)


def raw_feature_snapshot(
    raw: pd.DataFrame | dict[str, Any],
    *,
    include_engineered: bool = True,
    bundle: dict[str, Any] | None = None,
    preprocess_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Compact raw-attribute dict for the retention agent (unscaled values).

    When ``include_engineered`` is True, adds TenureBin, UnhappyComplain, and
    IsDormant (IsDormant uses the train recency threshold from the bundle).
    """
    if isinstance(raw, dict):
        row = dict(raw)
    else:
        if len(raw) != 1:
            raise ValueError("raw_feature_snapshot expects a single customer row")
        row = raw.iloc[0].to_dict()

    row.pop("CustomerID", None)
    snapshot = {k: _jsonable(v) for k, v in row.items() if k != "Churn"}

    if not include_engineered:
        return snapshot

    cleaned = engineer_pre_split(harmonise_labels(pd.DataFrame([row])))
    for key in ("TenureBin", "UnhappyComplain"):
        if key in cleaned.columns:
            snapshot[key] = _jsonable(cleaned.iloc[0][key])

    if bundle is None:
        try:
            bundle = load_preprocess_bundle(preprocess_path)
        except FileNotFoundError:
            bundle = None
    if bundle is not None and "DaySinceLastOrder" in cleaned.columns:
        day = cleaned.iloc[0]["DaySinceLastOrder"]
        if pd.notna(day):
            snapshot["IsDormant"] = int(float(day) > float(bundle["recency_threshold"]))
        elif "DaySinceLastOrder" in bundle.get("numeric_fill_values", {}):
            filled = float(bundle["numeric_fill_values"]["DaySinceLastOrder"])
            snapshot["IsDormant"] = int(filled > float(bundle["recency_threshold"]))
    return snapshot


def _jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value
