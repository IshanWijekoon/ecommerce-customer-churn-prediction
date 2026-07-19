"""Streamlit dashboard entrypoint (implemented in Phase 10)."""

from __future__ import annotations


def main() -> None:
    """Launch placeholder until the multipage dashboard is built."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "streamlit is required to run the dashboard. "
            "Install via: conda env create -f environment.yml"
        ) from exc

    st.set_page_config(
        page_title="E-commerce Churn Prediction",
        layout="wide",
    )
    st.title("E-commerce Customer Churn Prediction")
    st.info(
        "Dashboard pages (EDA, insights, prediction, explainability) "
        "will be added in Phase 10. Core analysis lives under `notebooks/`."
    )


if __name__ == "__main__":
    main()
