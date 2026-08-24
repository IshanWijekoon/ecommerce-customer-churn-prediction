"""
Load saved churn models and score a preprocessed feature matrix.

Models:
  - Logistic Regression + Random Forest on all 33 features
  - TabNet on the GA winning 17-feature subset
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CLEAN = ROOT / "data" / "clean"

DEFAULT_WINNER_PATH = CLEAN / "nia_feature_selection_winner.json"
DEFAULT_LR_PATH = MODELS / "lr_baseline.joblib"
DEFAULT_RF_PATH = MODELS / "rf_baseline.joblib"
DEFAULT_TABNET_DIR = MODELS / "tabnet_winning"
DEFAULT_TABNET_META = MODELS / "tabnet_meta.joblib"

# Notebook 8 risk bands
HIGH_RISK = 0.70
MEDIUM_RISK = 0.40
DECISION_THRESHOLD = 0.5

# Keep soft probs (esp. peaked TabNet) from collapsing to bare 0/1 in the UI.
PROBABILITY_DECIMALS = 8

MODEL_NAMES = ("logistic_regression", "random_forest", "tabnet")


@dataclass(frozen=True)
class ScoreResult:
    """One model's churn score for a single customer."""

    model: str
    probability: float
    predicted_churn: int
    label: str
    risk_band: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def risk_band(probability: float) -> str:
    """Map P(churn) to High / Medium / Low (notebook 8 cutoffs)."""
    p = float(probability)
    if p >= HIGH_RISK:
        return "High"
    if p >= MEDIUM_RISK:
        return "Medium"
    return "Low"


def predict_from_proba(probability: float, threshold: float = DECISION_THRESHOLD) -> int:
    return int(float(probability) >= threshold)


def load_winning_features(path: Path | str | None = None) -> list[str]:
    """Feature subset used by TabNet (GA winner)."""
    winner_path = Path(path) if path is not None else DEFAULT_WINNER_PATH
    if winner_path.exists():
        with winner_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        return list(payload["winning_features"])

    meta_path = DEFAULT_TABNET_META
    if meta_path.exists():
        meta = joblib.load(meta_path)
        return list(meta["features"])

    raise FileNotFoundError(
        f"Missing winning features at {winner_path} (and no tabnet_meta.joblib)."
    )


def load_sklearn_models(
    *,
    lr_path: Path | str | None = None,
    rf_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load Logistic Regression and Random Forest baselines."""
    lr_file = Path(lr_path) if lr_path is not None else DEFAULT_LR_PATH
    rf_file = Path(rf_path) if rf_path is not None else DEFAULT_RF_PATH
    missing = [p for p in (lr_file, rf_file) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing baseline model(s): "
            + ", ".join(str(p) for p in missing)
            + ". Re-run notebook 4."
        )
    return {
        "logistic_regression": joblib.load(lr_file),
        "random_forest": joblib.load(rf_file),
    }


def load_tabnet(
    *,
    model_dir: Path | str | None = None,
    meta_path: Path | str | None = None,
) -> tuple[Any, list[str]]:
    """Load TabNetClassifier + its winning feature list."""
    from pytorch_tabnet.tab_model import TabNetClassifier

    directory = Path(model_dir) if model_dir is not None else DEFAULT_TABNET_DIR
    # notebook 7: save_model(.../model) writes model.zip; load_model needs the .zip path
    zip_path = directory / "model.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing TabNet model at {zip_path}. Re-run notebook 7."
        )

    model = TabNetClassifier()
    model.load_model(str(zip_path))

    features: list[str]
    meta_file = Path(meta_path) if meta_path is not None else DEFAULT_TABNET_META
    if meta_file.exists():
        meta = joblib.load(meta_file)
        features = list(meta["features"])
    else:
        features = load_winning_features()
    return model, features


@dataclass
class ModelBundle:
    """Cached models for repeated Streamlit scoring."""

    logistic_regression: Any
    random_forest: Any
    tabnet: Any
    tabnet_features: list[str]

    @classmethod
    def load(cls, root: Path | str | None = None) -> ModelBundle:
        base = Path(root) if root is not None else ROOT
        models_dir = base / "models"
        clean_dir = base / "data" / "clean"
        sklearn = load_sklearn_models(
            lr_path=models_dir / "lr_baseline.joblib",
            rf_path=models_dir / "rf_baseline.joblib",
        )
        tabnet, feats = load_tabnet(
            model_dir=models_dir / "tabnet_winning",
            meta_path=models_dir / "tabnet_meta.joblib",
        )
        # Prefer JSON on disk when present (same list as meta)
        winner_json = clean_dir / "nia_feature_selection_winner.json"
        if winner_json.exists():
            feats = load_winning_features(winner_json)
        return cls(
            logistic_regression=sklearn["logistic_regression"],
            random_forest=sklearn["random_forest"],
            tabnet=tabnet,
            tabnet_features=feats,
        )


def _proba_sklearn(model: Any, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def _proba_tabnet(model: Any, X: pd.DataFrame, features: list[str]) -> np.ndarray:
    missing = [f for f in features if f not in X.columns]
    if missing:
        raise KeyError(f"TabNet features missing from matrix: {missing}")
    values = X[features].to_numpy(dtype=np.float32)
    return np.asarray(model.predict_proba(values)[:, 1], dtype=float)


def score_matrix(
    X: pd.DataFrame,
    bundle: ModelBundle | None = None,
    *,
    models: tuple[str, ...] = MODEL_NAMES,
) -> dict[str, list[ScoreResult]]:
    """
    Score every row in ``X`` with the requested models.

    Returns ``{model_name: [ScoreResult, ...]}`` aligned with ``X`` row order.
    """
    if bundle is None:
        bundle = ModelBundle.load()

    out: dict[str, list[ScoreResult]] = {}
    n = len(X)

    if "logistic_regression" in models:
        probs = _proba_sklearn(bundle.logistic_regression, X)
        out["logistic_regression"] = [
            _make_result("logistic_regression", float(probs[i])) for i in range(n)
        ]

    if "random_forest" in models:
        probs = _proba_sklearn(bundle.random_forest, X)
        out["random_forest"] = [
            _make_result("random_forest", float(probs[i])) for i in range(n)
        ]

    if "tabnet" in models:
        probs = _proba_tabnet(bundle.tabnet, X, bundle.tabnet_features)
        out["tabnet"] = [_make_result("tabnet", float(probs[i])) for i in range(n)]

    return out


def score_customer(
    X_row: pd.DataFrame | pd.Series,
    bundle: ModelBundle | None = None,
    *,
    models: tuple[str, ...] = MODEL_NAMES,
) -> dict[str, ScoreResult]:
    """Score a single preprocessed customer; returns one ScoreResult per model."""
    if isinstance(X_row, pd.Series):
        frame = X_row.to_frame().T
    else:
        frame = X_row
    if len(frame) != 1:
        raise ValueError("score_customer expects exactly one row")

    multi = score_matrix(frame, bundle=bundle, models=models)
    return {name: results[0] for name, results in multi.items()}


def results_as_table(scores: dict[str, ScoreResult]) -> pd.DataFrame:
    """Side-by-side comparison table for the Streamlit UI."""
    rows = [scores[name].to_dict() for name in scores]
    return pd.DataFrame(rows)


def _make_result(model: str, probability: float) -> ScoreResult:
    raw = float(probability)
    pred = predict_from_proba(raw)
    return ScoreResult(
        model=model,
        probability=round(raw, PROBABILITY_DECIMALS),
        predicted_churn=pred,
        label="churn" if pred == 1 else "stay",
        risk_band=risk_band(raw),
    )
