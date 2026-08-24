"""Overview — recruiter-facing KPIs, leaderboard, and chart gallery."""

from __future__ import annotations

import streamlit as st

import _bootstrap  # noqa: F401 — puts repo root on path for ``src``
from _data import (
    GALLERY,
    figure_path,
    load_leaderboard,
    load_retention_plays,
    overview_kpis,
)

st.set_page_config(
    page_title="Churn Overview",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("E-commerce customer churn")
st.caption(
    "Recruiter demo of the Nature-Inspired Algorithms mini-project: "
    "baselines → GA/PSO feature selection → TabNet, plus a live scoring desk."
)

kpis = overview_kpis()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Churn rate", kpis["churn_rate"])
c2.metric("TabNet F1 (GA subset)", kpis["tabnet_f1"])
c3.metric("GA winning subset", kpis["ga_features"])
c4.metric("NIA winner", kpis["nia_winner"])

st.subheader("Model leaderboard")
st.caption("Held-out test metrics from the notebook pipeline (seed = 42).")
board = load_leaderboard()
if board.empty:
    st.warning(
        "Leaderboard CSV missing. Re-run notebook 8 or "
        "`scripts/export_scoring_artifacts.py`."
    )
else:
    display = board.copy()
    for col in ("f1", "recall", "pr_auc"):
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{float(x):.3f}")
    st.dataframe(display, hide_index=True, width="stretch")

st.subheader("Findings gallery")
st.caption("Same charts as the README — no notebook required to browse them.")
rows = [GALLERY[i : i + 2] for i in range(0, len(GALLERY), 2)]
for row in rows:
    cols = st.columns(2)
    for col, item in zip(cols, row, strict=False):
        path = figure_path(item["file"])
        with col:
            if path is None:
                st.info(f"Missing figure: `{item['file']}`")
            else:
                st.image(str(path), width="stretch")
            st.markdown(f"**{item['caption']}**")

st.subheader("Retention playbook")
plays = load_retention_plays()
if plays.empty:
    st.info("Retention plays CSV not found yet.")
else:
    st.dataframe(plays, hide_index=True, width="stretch")

st.divider()
st.markdown(
    "Next: open **Score a customer** in the sidebar to compare "
    "Logistic Regression, Random Forest, and TabNet on one shopper, "
    "then generate a retention note."
)
