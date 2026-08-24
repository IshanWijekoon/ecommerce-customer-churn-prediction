"""Cached loaders for dashboard artifacts and models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from _bootstrap import ROOT

CLEAN = ROOT / "data" / "clean"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "tabnet": "TabNet",
}

GALLERY = [
    {
        "file": "1_churn_share.png",
        "caption": "Churn is rare (~16.8%) — accuracy alone is misleading.",
    },
    {
        "file": "2_complain_satisfaction.png",
        "caption": "Complaint + low satisfaction is the sharpest signal.",
    },
    {
        "file": "4_metric_comparison.png",
        "caption": "Random Forest beats Logistic Regression on F1.",
    },
    {
        "file": "8_nia_summary.png",
        "caption": "GA beats PSO — 17 features at stronger CV F1.",
    },
    {
        "file": "7_tabnet_confusion.png",
        "caption": "TabNet catches almost every churner (Recall ~0.97).",
    },
    {
        "file": "8_permutation_importance.png",
        "caption": "Tenure and Complain are the top drivers.",
    },
    {
        "file": "8_risk_bands.png",
        "caption": "High-risk band is genuinely high-risk (~91% churned).",
    },
    {
        "file": "8_model_leaderboard.png",
        "caption": "TabNet on GA features wins the leaderboard.",
    },
]

# Canonical form choices (preprocess still accepts Kaggle aliases)
LOGIN_OPTIONS = ["Computer", "Mobile Phone", "Phone"]
PAYMENT_OPTIONS = [
    "Debit Card",
    "Credit Card",
    "CC",
    "E wallet",
    "UPI",
    "Cash on Delivery",
    "COD",
]
GENDER_OPTIONS = ["Male", "Female"]
ORDER_CAT_OPTIONS = [
    "Laptop & Accessory",
    "Mobile Phone",
    "Mobile",
    "Fashion",
    "Grocery",
    "Others",
]
MARITAL_OPTIONS = ["Single", "Married", "Divorced"]


@st.cache_resource(show_spinner="Loading models…")
def get_model_bundle():
    from src.scoring import ModelBundle

    return ModelBundle.load(ROOT)


@st.cache_resource(show_spinner="Loading preprocess bundle…")
def get_preprocess_bundle() -> dict[str, Any]:
    from src.preprocess import load_preprocess_bundle

    return load_preprocess_bundle(MODELS / "preprocess.joblib")


@st.cache_data(show_spinner=False)
def load_leaderboard() -> pd.DataFrame:
    path = CLEAN / "model_leaderboard.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_preprocess_meta() -> dict[str, Any]:
    import json

    path = CLEAN / "preprocess_meta.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_winner_meta() -> dict[str, Any]:
    import json

    path = CLEAN / "nia_feature_selection_winner.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_raw_test_profiles() -> pd.DataFrame:
    path = CLEAN / "raw_test_profiles.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_retention_plays() -> pd.DataFrame:
    path = CLEAN / "retention_plays.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def overview_kpis() -> dict[str, str]:
    """Headline numbers for the Overview page."""
    meta = load_preprocess_meta()
    winner = load_winner_meta()
    board = load_leaderboard()

    churn = meta.get("train_churn_rate")
    churn_txt = f"{100 * float(churn):.1f}%" if churn is not None else "~16.8%"

    n_feat = winner.get("ga_n_selected") or winner.get("winning_features")
    if isinstance(n_feat, list):
        n_feat = len(n_feat)
    feat_txt = f"{int(n_feat)} features" if n_feat else "17 features"

    tabnet_f1 = None
    if not board.empty and "model" in board.columns and "f1" in board.columns:
        hit = board[board["model"].astype(str).str.contains("TabNet_subset", case=False)]
        if not hit.empty:
            tabnet_f1 = float(hit.iloc[0]["f1"])
    f1_txt = f"{tabnet_f1:.2f}" if tabnet_f1 is not None else "~0.92"

    winner_name = str(winner.get("winner", "GA"))
    return {
        "churn_rate": churn_txt,
        "tabnet_f1": f1_txt,
        "ga_features": feat_txt,
        "nia_winner": winner_name,
    }


def form_defaults(bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Sensible defaults from train fill stats when available."""
    defaults: dict[str, Any] = {
        "Tenure": 9.0,
        "PreferredLoginDevice": "Mobile Phone",
        "CityTier": 2,
        "WarehouseToHome": 14.0,
        "PreferredPaymentMode": "Debit Card",
        "Gender": "Male",
        "HourSpendOnApp": 3.0,
        "NumberOfDeviceRegistered": 4,
        "PreferedOrderCat": "Laptop & Accessory",
        "SatisfactionScore": 3,
        "MaritalStatus": "Married",
        "NumberOfAddress": 3,
        "Complain": 0,
        "OrderAmountHikeFromlastYear": 15.0,
        "CouponUsed": 1.0,
        "OrderCount": 2.0,
        "DaySinceLastOrder": 3.0,
        "CashbackAmount": 160.0,
    }
    if bundle is None:
        return defaults
    numeric = dict(bundle.get("numeric_fill_values", {}))
    category = dict(bundle.get("category_fill_values", {}))
    for key, value in numeric.items():
        if key in defaults and value is not None:
            defaults[key] = value
    for key, value in category.items():
        if key in defaults and value is not None:
            defaults[key] = value
    return defaults


def figure_path(name: str) -> Path | None:
    path = FIGURES / name
    return path if path.exists() else None
