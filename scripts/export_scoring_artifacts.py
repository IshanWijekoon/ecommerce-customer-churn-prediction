"""
Rebuild scoring artifacts without re-running notebook 3 end-to-end.

Produces:
  - models/preprocess.joblib  (scaler, fill stats, dummy columns, feature order)
  - data/clean/raw_test_profiles.csv  (raw step1 rows for held-out CustomerIDs)

Also checks that the clone-and-run scoring bundle files exist (models +
leaderboard / retention / importance / winner JSON). Re-run notebooks 4–8
if any of those are missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
MODELS = ROOT / "models"
RANDOM_SEED = 42

NUMERIC_COLUMNS = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "NumberOfAddress",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
    "CityTier",
    "SatisfactionScore",
    "Complain",
    "TenureBin",
]
CATEGORY_COLUMNS = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]
ORDINAL_AND_FLAGS = [
    "CityTier",
    "SatisfactionScore",
    "Complain",
    "TenureBin",
    "IsDormant",
    "UnhappyComplain",
]
CONTINUOUS_COLUMNS = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "NumberOfAddress",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

# Files the Streamlit demo expects (besides preprocess.joblib / raw_test_profiles)
SCORING_BUNDLE = [
    MODELS / "lr_baseline.joblib",
    MODELS / "rf_baseline.joblib",
    MODELS / "tabnet_meta.joblib",
    MODELS / "tabnet_winning" / "model.zip",
    CLEAN / "id_test.csv",
    CLEAN / "nia_feature_selection_winner.json",
    CLEAN / "permutation_importance.csv",
    CLEAN / "model_leaderboard.csv",
    CLEAN / "retention_plays.csv",
]


def harmonise_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["PreferredLoginDevice"] = out["PreferredLoginDevice"].replace(
        {"Phone": "Mobile Phone"}
    )
    out["PreferredPaymentMode"] = out["PreferredPaymentMode"].replace(
        {"CC": "Credit Card", "COD": "Cash on Delivery"}
    )
    out["PreferedOrderCat"] = out["PreferedOrderCat"].replace(
        {"Mobile": "Mobile Phone"}
    )
    return out


def engineer_pre_split(df: pd.DataFrame) -> pd.DataFrame:
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


def fit_preprocess_bundle(customers: pd.DataFrame) -> dict:
    """Mirror notebook 3 train-only fit path; return the joblib bundle dict."""
    df = engineer_pre_split(harmonise_labels(customers))
    target = "Churn"
    id_col = "CustomerID"

    y = df[target].astype(int)
    x = df.drop(columns=[target])
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    id_test = x_test[[id_col]].copy().reset_index(drop=True)
    x_train = x_train.drop(columns=[id_col]).copy()
    x_test = x_test.drop(columns=[id_col]).copy()

    numeric_fill_values = {
        col: x_train[col].median() for col in NUMERIC_COLUMNS
    }
    category_fill_values = {
        col: x_train[col].mode().iloc[0] for col in CATEGORY_COLUMNS
    }

    for col, fill_value in numeric_fill_values.items():
        x_train[col] = x_train[col].fillna(fill_value)
        x_test[col] = x_test[col].fillna(fill_value)
    for col, fill_value in category_fill_values.items():
        x_train[col] = x_train[col].fillna(fill_value)
        x_test[col] = x_test[col].fillna(fill_value)

    recency_threshold = float(x_train["DaySinceLastOrder"].median())
    x_train["IsDormant"] = (x_train["DaySinceLastOrder"] > recency_threshold).astype(
        int
    )
    x_test["IsDormant"] = (x_test["DaySinceLastOrder"] > recency_threshold).astype(int)

    x_train_num = x_train[CONTINUOUS_COLUMNS + ORDINAL_AND_FLAGS].copy()
    x_train_cat = pd.get_dummies(x_train[CATEGORY_COLUMNS], dtype=int)
    dummy_columns = list(x_train_cat.columns)

    scaler = RobustScaler()
    scaler.fit(x_train_num[CONTINUOUS_COLUMNS])

    x_train_continuous = pd.DataFrame(
        scaler.transform(x_train_num[CONTINUOUS_COLUMNS]),
        columns=CONTINUOUS_COLUMNS,
        index=x_train_num.index,
    )
    x_train_final = pd.concat(
        [x_train_continuous, x_train_num[ORDINAL_AND_FLAGS], x_train_cat],
        axis=1,
    )
    feature_names = list(x_train_final.columns)

    # Keep id_test aligned with existing notebook output when present
    existing_ids = CLEAN / "id_test.csv"
    if existing_ids.exists():
        saved_ids = pd.read_csv(existing_ids)
        if not saved_ids["CustomerID"].equals(id_test["CustomerID"].reset_index(drop=True)):
            print(
                "Warning: rebuilt split IDs differ from data/clean/id_test.csv. "
                "Using saved id_test.csv for raw profiles (models stay valid)."
            )
            id_test = saved_ids

    meta = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(len(feature_names)),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "recency_threshold": recency_threshold,
        "random_seed": RANDOM_SEED,
    }

    bundle = {
        "scaler": scaler,
        "numeric_fill_values": numeric_fill_values,
        "category_fill_values": category_fill_values,
        "recency_threshold": recency_threshold,
        "dummy_columns": dummy_columns,
        "feature_names": feature_names,
        "continuous_columns": CONTINUOUS_COLUMNS,
        "ordinal_and_flags": ORDINAL_AND_FLAGS,
        "numeric_columns": NUMERIC_COLUMNS,
        "category_columns": CATEGORY_COLUMNS,
        "random_seed": RANDOM_SEED,
    }
    return bundle, meta, id_test


def export_raw_test_profiles(id_test: pd.DataFrame, customers: pd.DataFrame) -> Path:
    """Join held-out IDs to original (unscaled) step1 rows for the UI."""
    profiles = id_test.merge(customers, on="CustomerID", how="left")
    out_path = CLEAN / "raw_test_profiles.csv"
    profiles.to_csv(out_path, index=False)
    return out_path


def check_scoring_bundle() -> list[Path]:
    missing = [path for path in SCORING_BUNDLE if not path.exists()]
    return missing


def main() -> int:
    step1 = CLEAN / "step1_customers.csv"
    if not step1.exists():
        print(f"Missing {step1}. Run notebook 1 first.", file=sys.stderr)
        return 1

    customers = pd.read_csv(step1)
    MODELS.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)

    bundle, meta, id_test = fit_preprocess_bundle(customers)
    preprocess_path = MODELS / "preprocess.joblib"
    joblib.dump(bundle, preprocess_path)

    meta_path = CLEAN / "preprocess_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Prefer IDs already written by notebook 3 so profiles match trained models
    id_path = CLEAN / "id_test.csv"
    if id_path.exists():
        id_test = pd.read_csv(id_path)
    else:
        id_test.to_csv(id_path, index=False)

    profiles_path = export_raw_test_profiles(id_test, customers)

    print(f"Wrote {preprocess_path}")
    print(f"  features={len(bundle['feature_names'])} "
          f"dummies={len(bundle['dummy_columns'])} "
          f"recency_threshold={bundle['recency_threshold']}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {profiles_path} ({len(id_test)} test customers)")

    missing = check_scoring_bundle()
    if missing:
        print("\nScoring bundle incomplete (re-run notebooks 4–8 if needed):")
        for path in missing:
            print(f"  missing: {path.relative_to(ROOT)}")
        return 1

    print("\nScoring bundle ready:")
    for path in SCORING_BUNDLE + [preprocess_path, profiles_path]:
        print(f"  ok: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
