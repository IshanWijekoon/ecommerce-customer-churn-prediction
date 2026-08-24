"""Unit tests for src/ (no PyTorch / TabNet / API calls)."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.agent import generate_retention_note, get_api_key
from src.explain import (
    compare_to_reference,
    explain_customer,
    format_explanation_text,
    load_retention_plays,
)
from src.preprocess import (
    engineer_pre_split,
    harmonise_labels,
    raw_feature_snapshot,
    transform_raw_customers,
)
from src.scoring import ScoreResult, results_as_table, risk_band


@pytest.fixture
def tiny_bundle(synthetic_customers: pd.DataFrame) -> dict:
    """Minimal preprocess bundle fitted like notebook 3 (no disk I/O)."""
    import export_scoring_artifacts as exp

    bundle, _meta, _ids = exp.fit_preprocess_bundle(synthetic_customers)
    return bundle


def test_harmonise_and_tenure_bins() -> None:
    raw = pd.DataFrame(
        {
            "PreferredLoginDevice": ["Phone"],
            "PreferredPaymentMode": ["CC"],
            "PreferedOrderCat": ["Mobile"],
            "Tenure": [4.0],
            "Complain": [1],
            "SatisfactionScore": [2],
        }
    )
    cleaned = engineer_pre_split(harmonise_labels(raw))
    assert cleaned["PreferredLoginDevice"].iloc[0] == "Mobile Phone"
    assert cleaned["PreferredPaymentMode"].iloc[0] == "Credit Card"
    assert cleaned["PreferedOrderCat"].iloc[0] == "Mobile Phone"
    assert cleaned["TenureBin"].iloc[0] == 1.0
    assert cleaned["UnhappyComplain"].iloc[0] == 1


def test_transform_raw_customers_shape(tiny_bundle: dict, synthetic_customers: pd.DataFrame) -> None:
    row = synthetic_customers.iloc[[0]].drop(columns=["Churn"])
    matrix = transform_raw_customers(row, tiny_bundle)
    assert list(matrix.columns) == tiny_bundle["feature_names"]
    assert len(matrix) == 1
    assert matrix.isna().sum().sum() == 0
    assert isinstance(tiny_bundle["scaler"], RobustScaler)


def test_transform_accepts_dict(tiny_bundle: dict, synthetic_customers: pd.DataFrame) -> None:
    payload = synthetic_customers.iloc[3].drop(labels=["Churn"]).to_dict()
    matrix = transform_raw_customers(payload, tiny_bundle)
    assert matrix.shape == (1, len(tiny_bundle["feature_names"]))


def test_risk_band_cutoffs() -> None:
    assert risk_band(0.70) == "High"
    assert risk_band(0.699) == "Medium"
    assert risk_band(0.40) == "Medium"
    assert risk_band(0.399) == "Low"


def test_explain_and_agent_fallback(tiny_bundle: dict, synthetic_customers: pd.DataFrame) -> None:
    raw = synthetic_customers.iloc[[2]]
    snap = raw_feature_snapshot(raw, bundle=tiny_bundle)
    score = ScoreResult(
        model="tabnet",
        probability=0.82,
        predicted_churn=1,
        label="churn",
        risk_band="High",
    )
    explanation = explain_customer(
        score=score,
        raw_snapshot=snap,
        preprocess_bundle=tiny_bundle,
    )
    assert explanation["source"] == "rule_based"
    assert explanation["predicted_churn"] == 1
    assert len(explanation["why"]) >= 1
    assert len(explanation["guide"]) >= 1
    text = format_explanation_text(explanation)
    assert "P(churn)" in text

    note = generate_retention_note(
        score=score,
        raw_snapshot=snap,
        preprocess_bundle=tiny_bundle,
        use_llm=False,
    )
    assert note["source"] == "rule_based"
    assert note["error"] is None
    assert "churn" in note["text"].lower() or "P(churn)" in note["text"]


def test_agent_missing_key_uses_fallback(
    tiny_bundle: dict, synthetic_customers: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert get_api_key() is None
    snap = raw_feature_snapshot(synthetic_customers.iloc[[1]], bundle=tiny_bundle)
    score = {
        "model": "random_forest",
        "probability": 0.12,
        "predicted_churn": 0,
        "label": "stay",
        "risk_band": "Low",
    }
    note = generate_retention_note(
        score=score,
        raw_snapshot=snap,
        preprocess_bundle=tiny_bundle,
        use_llm=True,
    )
    assert note["source"] == "rule_based"
    assert note["error"] is not None


def test_compare_complain_flag() -> None:
    risky = compare_to_reference("Complain", 1, 0)
    safe = compare_to_reference("Complain", 0, 0)
    assert risky["direction"] == "riskier"
    assert safe["direction"] == "safer"


def test_retention_plays_builtin_or_csv() -> None:
    plays = load_retention_plays()
    assert len(plays) >= 5
    assert "play" in plays[0]


def test_results_as_table() -> None:
    scores = {
        "tabnet": ScoreResult("tabnet", 0.9, 1, "churn", "High"),
        "random_forest": ScoreResult("random_forest", 0.2, 0, "stay", "Low"),
    }
    table = results_as_table(scores)
    assert set(table["model"]) == {"tabnet", "random_forest"}
