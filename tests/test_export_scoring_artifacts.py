"""Tests for scripts/export_scoring_artifacts.py (no Kaggle file, no PyTorch)."""

from __future__ import annotations

import pandas as pd

import export_scoring_artifacts as exp


def test_harmonise_labels_maps_aliases() -> None:
    raw = pd.DataFrame(
        {
            "PreferredLoginDevice": ["Phone", "Mobile Phone"],
            "PreferredPaymentMode": ["CC", "COD"],
            "PreferedOrderCat": ["Mobile", "Fashion"],
        }
    )
    out = exp.harmonise_labels(raw)
    assert out["PreferredLoginDevice"].tolist() == ["Mobile Phone", "Mobile Phone"]
    assert out["PreferredPaymentMode"].tolist() == ["Credit Card", "Cash on Delivery"]
    assert out["PreferedOrderCat"].tolist() == ["Mobile Phone", "Fashion"]


def test_engineer_pre_split_tenure_bins_and_unhappy_flag() -> None:
    raw = pd.DataFrame(
        {
            "Tenure": [0.0, 3.0, 4.0, 9.0, 10.0, 15.0, 20.0, None],
            "Complain": [1, 1, 1, 0, 1, 0, None, 1],
            "SatisfactionScore": [1, 2, 3, 1, 5, 2, 1, None],
        }
    )
    out = exp.engineer_pre_split(raw)
    assert out["TenureBin"].tolist()[:7] == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    assert pd.isna(out.loc[7, "TenureBin"])
    # Complain==1 and satisfaction <= 2 (missing satisfaction treated as 3)
    assert out["UnhappyComplain"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]


def test_fit_preprocess_bundle_has_expected_keys(
    tmp_path, monkeypatch, synthetic_customers: pd.DataFrame
) -> None:
    monkeypatch.setattr(exp, "CLEAN", tmp_path)
    bundle, meta, id_test = exp.fit_preprocess_bundle(synthetic_customers)

    for key in (
        "scaler",
        "numeric_fill_values",
        "category_fill_values",
        "recency_threshold",
        "dummy_columns",
        "feature_names",
        "continuous_columns",
        "ordinal_and_flags",
        "numeric_columns",
        "category_columns",
        "random_seed",
    ):
        assert key in bundle

    assert meta["random_seed"] == 42
    assert meta["n_train"] + meta["n_test"] == len(synthetic_customers)
    assert len(id_test) == meta["n_test"]
    assert "CustomerID" in id_test.columns
    assert bundle["scaler"].n_features_in_ == len(exp.CONTINUOUS_COLUMNS)
    for flag in ("TenureBin", "IsDormant", "UnhappyComplain", "Complain"):
        assert flag in bundle["feature_names"]
    assert len(bundle["dummy_columns"]) >= 1
    assert set(bundle["feature_names"]) == set(
        bundle["continuous_columns"]
        + bundle["ordinal_and_flags"]
        + bundle["dummy_columns"]
    )


def test_export_raw_test_profiles(tmp_path, monkeypatch, synthetic_customers: pd.DataFrame) -> None:
    monkeypatch.setattr(exp, "CLEAN", tmp_path)
    ids = pd.DataFrame({"CustomerID": synthetic_customers["CustomerID"].head(5)})
    out = exp.export_raw_test_profiles(ids, synthetic_customers)
    profiles = pd.read_csv(out)
    assert len(profiles) == 5
    assert "Churn" in profiles.columns
    assert "Tenure" in profiles.columns
    assert profiles["CustomerID"].tolist() == ids["CustomerID"].tolist()


def test_check_scoring_bundle_reports_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "MODELS", tmp_path / "models")
    monkeypatch.setattr(exp, "CLEAN", tmp_path / "clean")
    monkeypatch.setattr(
        exp,
        "SCORING_BUNDLE",
        [tmp_path / "models" / "lr_baseline.joblib", tmp_path / "clean" / "id_test.csv"],
    )
    missing = exp.check_scoring_bundle()
    assert len(missing) == 2
