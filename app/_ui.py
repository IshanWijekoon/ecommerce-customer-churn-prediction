"""Shared Streamlit UI helpers for scoring cards and retention panel."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from _data import MODEL_DISPLAY

BAND_COLORS = {
    "High": "#b42318",
    "Medium": "#b54708",
    "Low": "#027a48",
}

# Fixed decimals so peaked TabNet probs never render as bare 0 / 1.
PROB_DISPLAY_DECIMALS = 6


def format_probability(probability: float) -> str:
    """Always show a decimal string (e.g. 0.999988), never bare 0/1."""
    return f"{float(probability):.{PROB_DISPLAY_DECIMALS}f}"


def format_probability_pct(probability: float) -> str:
    """Percentage with enough places that 0.999988 does not become 100.0%."""
    return f"{100.0 * float(probability):.4f}%"


def render_score_cards(scores: dict[str, Any]) -> None:
    """Side-by-side probability / label / risk for each model."""
    cols = st.columns(len(scores) or 1)
    for col, (name, result) in zip(cols, scores.items(), strict=False):
        data = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        label = MODEL_DISPLAY.get(name, name)
        band = data.get("risk_band", "—")
        color = BAND_COLORS.get(str(band), "#344054")
        pred = data.get("label", "—")
        prob = float(data.get("probability", 0.0))
        with col:
            st.markdown(f"**{label}**")
            st.metric("P(churn)", format_probability(prob))
            st.caption(format_probability_pct(prob))
            st.markdown(
                f"Prediction: **{pred}**  \n"
                f"Risk: <span style='color:{color};font-weight:600'>{band}</span>",
                unsafe_allow_html=True,
            )


def scores_table(scores: dict[str, Any]) -> pd.DataFrame:
    from src.scoring import results_as_table

    table = results_as_table(scores)
    table["model"] = table["model"].map(lambda m: MODEL_DISPLAY.get(m, m))
    table["probability"] = table["probability"].map(format_probability)
    table = table.rename(
        columns={
            "model": "Model",
            "probability": "P(churn)",
            "predicted_churn": "Predicted churn",
            "label": "Label",
            "risk_band": "Risk band",
        }
    )
    return table


def render_raw_profile(snapshot: dict[str, Any], *, title: str = "Customer profile") -> None:
    """Show unscaled attributes (not the RobustScaler matrix)."""
    st.subheader(title)
    prefer = [
        "Tenure",
        "Complain",
        "SatisfactionScore",
        "DaySinceLastOrder",
        "UnhappyComplain",
        "IsDormant",
        "CashbackAmount",
        "OrderCount",
        "CouponUsed",
        "PreferredLoginDevice",
        "PreferredPaymentMode",
        "PreferedOrderCat",
        "Gender",
        "MaritalStatus",
        "CityTier",
    ]
    ordered = [k for k in prefer if k in snapshot] + [
        k for k in snapshot if k not in prefer and k not in {"CustomerID", "Churn"}
    ]
    view = pd.DataFrame(
        [{"Feature": k, "Value": snapshot[k]} for k in ordered]
    )
    st.dataframe(view, hide_index=True, width="stretch")


def render_retention_panel(
    *,
    scores: dict[str, Any],
    raw_snapshot: dict[str, Any],
    preprocess_bundle: dict[str, Any] | None,
    default_model: str = "tabnet",
) -> None:
    """Model picker + OpenRouter / rule-based retention note."""
    st.divider()
    st.header("Retention agent")
    st.caption(
        "Uses only this customer’s score, raw features, top drivers, and the "
        "notebook-8 playbook. No invented metrics."
    )

    model_keys = list(scores.keys())
    if not model_keys:
        st.info("Score a customer first to generate a retention note.")
        return

    default_idx = (
        model_keys.index(default_model) if default_model in model_keys else 0
    )
    selected = st.selectbox(
        "Explain with model",
        options=model_keys,
        index=default_idx,
        format_func=lambda m: MODEL_DISPLAY.get(m, m),
        key="retention_model",
    )
    use_llm = st.toggle(
        "Use OpenRouter (falls back to rule-based if the key is missing)",
        value=True,
        key="retention_use_llm",
    )

    if st.button("Generate retention note", type="primary", key="retention_go"):
        from src.agent import generate_retention_note

        with st.spinner("Writing retention note…"):
            note = generate_retention_note(
                score=scores[selected],
                raw_snapshot=raw_snapshot,
                preprocess_bundle=preprocess_bundle,
                use_llm=use_llm,
            )
        source = note.get("source", "rule_based")
        if note.get("error"):
            st.warning(note["error"])
        else:
            st.success(
                "OpenRouter note"
                if source == "openrouter"
                else "Rule-based note (deterministic fallback)"
            )
        st.markdown(note["text"])

        with st.expander("Structured drivers (rule-based)"):
            explanation = note.get("explanation") or {}
            why = explanation.get("why") or []
            guide = explanation.get("guide") or []
            if why:
                st.markdown("**Why**")
                for line in why:
                    st.markdown(f"- {line}")
            if guide:
                heading = (
                    "**Retaining guide**"
                    if explanation.get("predicted_churn")
                    else "**Keep healthy**"
                )
                st.markdown(heading)
                for line in guide:
                    st.markdown(f"- {line}")
