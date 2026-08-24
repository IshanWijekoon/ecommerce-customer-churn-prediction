"""
Rule-based local explanations for a scored customer.

Used as the retention-agent fallback when ``OPENROUTER_API_KEY`` is missing,
and as structured driver facts fed into the LLM prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.scoring import ScoreResult, risk_band

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
DEFAULT_IMPORTANCE_PATH = CLEAN / "permutation_importance.csv"
DEFAULT_PLAYS_PATH = CLEAN / "retention_plays.csv"

# Drivers the plan calls out for "why churn / why stay" copy
FOCUS_DRIVERS = (
    "Tenure",
    "Complain",
    "SatisfactionScore",
    "DaySinceLastOrder",
    "UnhappyComplain",
    "IsDormant",
)

# Higher = more churn risk for these continuous / ordinal fields (EDA / notebook 8)
HIGHER_IS_RISKIER = {
    "Complain",
    "UnhappyComplain",
    "IsDormant",
    "DaySinceLastOrder",
    "WarehouseToHome",
    "NumberOfAddress",
    "CityTier",
    "MaritalStatus_Single",
}
LOWER_IS_RISKIER = {
    "Tenure",
    "SatisfactionScore",
    "CashbackAmount",
    "CouponUsed",
    "OrderCount",
}


def load_top_drivers(
    path: Path | str | None = None,
    *,
    n: int = 8,
) -> list[dict[str, Any]]:
    """Top features by permutation importance (notebook 8)."""
    importance_path = Path(path) if path is not None else DEFAULT_IMPORTANCE_PATH
    if not importance_path.exists():
        return [{"feature": name, "importance": None} for name in FOCUS_DRIVERS[:n]]
    table = pd.read_csv(importance_path)
    table = table.sort_values("importance", ascending=False).head(n)
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        rows.append(
            {
                "feature": str(row["feature"]),
                "importance": float(row["importance"]),
            }
        )
    return rows


def load_retention_plays(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Five retention plays from notebook 8."""
    plays_path = Path(path) if path is not None else DEFAULT_PLAYS_PATH
    if not plays_path.exists():
        return _builtin_plays()
    table = pd.read_csv(plays_path)
    return table.to_dict(orient="records")


def _builtin_plays() -> list[dict[str, Any]]:
    return [
        {
            "play": "Complaint recovery SWAT",
            "lever": "Complain / UnhappyComplain",
            "owner": "Customer Experience",
            "expected_effect": "Fix issues before sending discounts",
        },
        {
            "play": "New-customer onboarding",
            "lever": "Tenure / TenureBin",
            "owner": "Growth",
            "expected_effect": "Protect customers in first months",
        },
        {
            "play": "Dormancy win-back",
            "lever": "DaySinceLastOrder / IsDormant",
            "owner": "CRM",
            "expected_effect": "Reactivate silent buyers",
        },
        {
            "play": "Satisfaction rescue",
            "lever": "SatisfactionScore",
            "owner": "CX / Product",
            "expected_effect": "Follow up on low scores quickly",
        },
        {
            "play": "Targeted loyalty rewards",
            "lever": "CouponUsed / category features",
            "owner": "Loyalty",
            "expected_effect": "Give relevant offers, not blanket discounts",
        },
    ]


def _as_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_to_reference(
    feature: str,
    customer_value: Any,
    reference_value: Any,
) -> dict[str, Any]:
    """
    Compare one feature to the train median / reference.

    Returns direction: ``riskier``, ``safer``, ``similar``, or ``flag``.
    """
    cust = _as_float(customer_value)
    ref = _as_float(reference_value)

    # Binary-style flags
    if feature in {"Complain", "UnhappyComplain", "IsDormant"}:
        on = int(cust or 0) == 1
        return {
            "feature": feature,
            "customer_value": int(cust or 0) if cust is not None else customer_value,
            "reference_value": ref,
            "direction": "riskier" if on else "safer",
            "note": _flag_note(feature, on),
        }

    if cust is None or ref is None:
        return {
            "feature": feature,
            "customer_value": customer_value,
            "reference_value": reference_value,
            "direction": "similar",
            "note": f"{feature}: value unavailable for comparison",
        }

    # Relative gap vs train median
    if abs(cust - ref) < 1e-9:
        direction = "similar"
    elif feature in HIGHER_IS_RISKIER:
        direction = "riskier" if cust > ref else "safer"
    elif feature in LOWER_IS_RISKIER:
        direction = "riskier" if cust < ref else "safer"
    else:
        # Unknown continuous: treat large absolute gaps neutrally
        direction = "similar"

    return {
        "feature": feature,
        "customer_value": cust,
        "reference_value": ref,
        "direction": direction,
        "note": _numeric_note(feature, cust, ref, direction),
    }


def _flag_note(feature: str, on: bool) -> str:
    if feature == "Complain":
        return (
            "Customer filed a complaint (high-risk signal)."
            if on
            else "No complaint on file."
        )
    if feature == "UnhappyComplain":
        return (
            "Complaint plus low satisfaction (UnhappyComplain=1)."
            if on
            else "Not in the unhappy-complaint segment."
        )
    if feature == "IsDormant":
        return (
            "Days since last order exceed the train median (dormant)."
            if on
            else "Recent enough activity vs train median (not dormant)."
        )
    return f"{feature}={'1' if on else '0'}"


def _numeric_note(feature: str, cust: float, ref: float, direction: str) -> str:
    if direction == "similar":
        return f"{feature}={cust:g} is near the train median ({ref:g})."
    if direction == "riskier":
        return (
            f"{feature}={cust:g} looks riskier than the train median ({ref:g})."
        )
    return f"{feature}={cust:g} looks safer than the train median ({ref:g})."


def build_driver_comparisons(
    raw_snapshot: dict[str, Any],
    *,
    reference: dict[str, Any] | None = None,
    top_drivers: list[dict[str, Any]] | None = None,
    preprocess_bundle: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Compare this customer's raw values to train medians on top drivers.

    ``reference`` defaults to ``numeric_fill_values`` from the preprocess bundle
    (those are the train-set medians used for imputation).
    """
    if reference is None and preprocess_bundle is not None:
        reference = dict(preprocess_bundle.get("numeric_fill_values", {}))
        # Engineered / binary defaults
        reference.setdefault("UnhappyComplain", 0)
        reference.setdefault("IsDormant", 0)
        if "recency_threshold" in preprocess_bundle:
            reference.setdefault(
                "DaySinceLastOrder",
                preprocess_bundle["recency_threshold"],
            )
    if reference is None:
        reference = {}

    if top_drivers is None:
        top_drivers = load_top_drivers()

    # Prefer permutation top list, always include focus drivers
    names: list[str] = []
    for row in top_drivers:
        name = str(row["feature"])
        if name not in names:
            names.append(name)
    for name in FOCUS_DRIVERS:
        if name not in names:
            names.append(name)

    comparisons: list[dict[str, Any]] = []
    for name in names:
        if name not in raw_snapshot and name not in reference:
            continue
        # Skip one-hot columns unless present in snapshot
        if "_" in name and name not in raw_snapshot:
            # One-hots live on the scaled matrix; skip unless UI passed them
            continue
        comparisons.append(
            compare_to_reference(
                name,
                raw_snapshot.get(name),
                reference.get(name),
            )
        )
    return comparisons


def select_relevant_plays(
    comparisons: list[dict[str, Any]],
    plays: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Pick playbook rows whose levers match this customer's riskier flags."""
    if plays is None:
        plays = load_retention_plays()

    riskier = {
        c["feature"] for c in comparisons if c.get("direction") == "riskier"
    }
    selected: list[dict[str, Any]] = []
    for play in plays:
        lever = str(play.get("lever", ""))
        lever_bits = [p.strip() for p in lever.replace("/", ",").split(",")]
        if any(bit in riskier or any(bit in f for f in riskier) for bit in lever_bits if bit):
            selected.append(play)

    # Always return something actionable for high-risk demos
    return selected or plays[:3]


def explain_customer(
    *,
    score: ScoreResult | dict[str, Any],
    raw_snapshot: dict[str, Any],
    preprocess_bundle: dict[str, Any] | None = None,
    top_drivers: list[dict[str, Any]] | None = None,
    plays: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build a structured, deterministic explanation + short retention note.

    Does not invent metrics: only uses supplied score, snapshot, and playbook.
    """
    if isinstance(score, ScoreResult):
        score_dict = score.to_dict()
    else:
        score_dict = dict(score)
        if "risk_band" not in score_dict and "probability" in score_dict:
            score_dict["risk_band"] = risk_band(float(score_dict["probability"]))

    comparisons = build_driver_comparisons(
        raw_snapshot,
        preprocess_bundle=preprocess_bundle,
        top_drivers=top_drivers,
    )
    plays_all = plays if plays is not None else load_retention_plays()
    relevant = select_relevant_plays(comparisons, plays_all)

    predicted_churn = int(score_dict.get("predicted_churn", 0))
    why_lines = _why_lines(comparisons, predicted_churn=predicted_churn)
    guide = _guide_lines(predicted_churn=predicted_churn, plays=relevant)

    return {
        "model": score_dict.get("model"),
        "probability": score_dict.get("probability"),
        "risk_band": score_dict.get("risk_band"),
        "label": score_dict.get("label"),
        "predicted_churn": predicted_churn,
        "why": why_lines,
        "guide": guide,
        "driver_comparisons": comparisons,
        "relevant_plays": relevant,
        "source": "rule_based",
    }


def format_explanation_text(explanation: dict[str, Any]) -> str:
    """Plain-text block suitable for Streamlit markdown / agent fallback."""
    label = explanation.get("label", "unknown")
    prob = explanation.get("probability")
    band = explanation.get("risk_band")
    model = explanation.get("model")
    header = (
        f"Model={model} | P(churn)={prob} | band={band} | prediction={label}"
    )
    parts = [header, "", "Why:"]
    for line in explanation.get("why", []):
        parts.append(f"- {line}")
    parts.append("")
    parts.append("Guide:" if explanation.get("predicted_churn") else "Keep healthy:")
    for line in explanation.get("guide", []):
        parts.append(f"- {line}")
    return "\n".join(parts)


def explanation_facts_payload(
    *,
    score: ScoreResult | dict[str, Any],
    raw_snapshot: dict[str, Any],
    preprocess_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Facts-only JSON for the OpenRouter agent (no free-form invention room).
    """
    explanation = explain_customer(
        score=score,
        raw_snapshot=raw_snapshot,
        preprocess_bundle=preprocess_bundle,
    )
    return {
        "selected_model": explanation["model"],
        "p_churn": explanation["probability"],
        "risk_band": explanation["risk_band"],
        "predicted_label": explanation["label"],
        "raw_feature_snapshot": raw_snapshot,
        "top_driver_comparisons": explanation["driver_comparisons"][:8],
        "retention_plays": explanation["relevant_plays"],
        "all_retention_plays": load_retention_plays(),
    }


def _why_lines(
    comparisons: list[dict[str, Any]],
    *,
    predicted_churn: int,
) -> list[str]:
    riskier = [c for c in comparisons if c.get("direction") == "riskier"]
    safer = [c for c in comparisons if c.get("direction") == "safer"]

    lines: list[str] = []
    if predicted_churn:
        for c in riskier[:5]:
            lines.append(c["note"])
        if not lines:
            lines.append(
                "Model score is elevated, but no strong raw-driver flags vs train medians."
            )
    else:
        for c in safer[:5]:
            lines.append(c["note"])
        # Still surface any residual risks honestly
        for c in riskier[:2]:
            lines.append("Residual risk: " + c["note"])
        if not lines:
            lines.append(
                "Model score is low; top drivers sit near safer train-median values."
            )
    return lines


def _guide_lines(*, predicted_churn: int, plays: list[dict[str, Any]]) -> list[str]:
    if predicted_churn:
        bullets = []
        for play in plays[:5]:
            bullets.append(
                f"{play['play']} ({play['owner']}): {play['expected_effect']} "
                f"— lever: {play['lever']}"
            )
        if not bullets:
            bullets.append("Prioritise complaint recovery before offering discounts.")
        return bullets

    return [
        "Keep service quality steady; avoid dumping blanket discounts.",
        "Watch satisfaction and complaint tickets so risk does not creep up.",
        "Use light loyalty touches only when category preference is clear.",
    ]


def dumps_facts(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)
