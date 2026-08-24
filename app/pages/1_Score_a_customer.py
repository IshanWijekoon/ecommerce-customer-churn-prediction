"""Score one customer (test ID or form) and open the retention agent."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Entrypoint lives in app/; pages/ is one level deeper — keep app/ importable.
_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import _bootstrap  # noqa: F401 — puts repo root on path for ``src``
from _data import (
    GENDER_OPTIONS,
    LOGIN_OPTIONS,
    MARITAL_OPTIONS,
    ORDER_CAT_OPTIONS,
    PAYMENT_OPTIONS,
    form_defaults,
    get_model_bundle,
    get_preprocess_bundle,
    load_raw_test_profiles,
)
from _ui import (
    render_raw_profile,
    render_retention_panel,
    render_score_cards,
    scores_table,
)

st.set_page_config(
    page_title="Score a customer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Score a customer")
st.caption(
    "Compare Logistic Regression, Random Forest, and TabNet on one shopper. "
    "Raw attributes are shown unscaled; models receive the notebook-3 matrix."
)


def _score_raw_row(raw: dict[str, Any] | pd.DataFrame) -> tuple[dict, dict, dict]:
    from src.preprocess import raw_feature_snapshot, transform_raw_customers
    from src.scoring import score_customer

    bundle = get_preprocess_bundle()
    models = get_model_bundle()
    matrix = transform_raw_customers(raw, bundle)
    scores = score_customer(matrix, bundle=models)
    snapshot = raw_feature_snapshot(raw, bundle=bundle)
    return scores, snapshot, bundle


def _actual_badge(churn: int | None) -> None:
    if churn is None:
        return
    if int(churn) == 1:
        st.error("Actual label (held-out): **churned**")
    else:
        st.success("Actual label (held-out): **stayed**")


profiles = load_raw_test_profiles()
tab_test, tab_form = st.tabs(["Test set", "Manual form"])

with tab_test:
    if profiles.empty:
        st.warning(
            "Missing `data/clean/raw_test_profiles.csv`. "
            "Run `scripts/export_scoring_artifacts.py` or notebook 3."
        )
    else:
        ids = profiles["CustomerID"].astype(int).tolist()
        selected_id = st.selectbox(
            "CustomerID (held-out test set)",
            options=ids,
            format_func=lambda x: str(x),
            help="Search by typing a CustomerID.",
        )
        show_actual = st.checkbox("Show actual churn label", value=True)
        row = profiles.loc[profiles["CustomerID"] == selected_id].iloc[0]
        actual = int(row["Churn"]) if "Churn" in row.index and show_actual else None

        raw_for_score = row.drop(labels=["Churn"], errors="ignore").to_dict()
        preview = {
            k: v
            for k, v in row.to_dict().items()
            if k not in {"Churn"}
        }
        render_raw_profile(preview, title=f"Raw profile — CustomerID {selected_id}")
        if show_actual:
            _actual_badge(actual)

        if st.button("Run all three models", type="primary", key="score_test"):
            try:
                with st.spinner("Scoring…"):
                    scores, snapshot, _bundle = _score_raw_row(raw_for_score)
                st.session_state["last_scores"] = scores
                st.session_state["last_snapshot"] = snapshot
                st.session_state["last_source"] = f"test:{selected_id}"
            except FileNotFoundError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001 — surface scoring errors in UI
                st.exception(exc)

with tab_form:
    st.markdown("Enter original Kaggle-style fields (aliases like `Phone` / `CC` are fine).")
    try:
        defaults = form_defaults(get_preprocess_bundle())
    except FileNotFoundError:
        defaults = form_defaults()

    left, right = st.columns(2)
    with left:
        tenure = st.number_input("Tenure (months)", min_value=0.0, value=float(defaults["Tenure"]))
        warehouse = st.number_input(
            "WarehouseToHome",
            min_value=0.0,
            value=float(defaults["WarehouseToHome"]),
        )
        hours = st.number_input(
            "HourSpendOnApp",
            min_value=0.0,
            value=float(defaults["HourSpendOnApp"]),
        )
        devices = st.number_input(
            "NumberOfDeviceRegistered",
            min_value=1,
            max_value=10,
            value=int(defaults["NumberOfDeviceRegistered"]),
        )
        addresses = st.number_input(
            "NumberOfAddress",
            min_value=1,
            max_value=30,
            value=int(defaults["NumberOfAddress"]),
        )
        hike = st.number_input(
            "OrderAmountHikeFromlastYear",
            min_value=0.0,
            value=float(defaults["OrderAmountHikeFromlastYear"]),
        )
        coupons = st.number_input(
            "CouponUsed",
            min_value=0.0,
            value=float(defaults["CouponUsed"]),
        )
        orders = st.number_input(
            "OrderCount",
            min_value=0.0,
            value=float(defaults["OrderCount"]),
        )
        days = st.number_input(
            "DaySinceLastOrder",
            min_value=0.0,
            value=float(defaults["DaySinceLastOrder"]),
        )
    with right:
        cashback = st.number_input(
            "CashbackAmount",
            min_value=0.0,
            value=float(defaults["CashbackAmount"]),
        )
        city = st.selectbox(
            "CityTier",
            options=[1, 2, 3],
            index=[1, 2, 3].index(int(defaults["CityTier"]))
            if int(defaults["CityTier"]) in (1, 2, 3)
            else 1,
        )
        satisfaction = st.selectbox(
            "SatisfactionScore",
            options=[1, 2, 3, 4, 5],
            index=int(defaults["SatisfactionScore"]) - 1
            if 1 <= int(defaults["SatisfactionScore"]) <= 5
            else 2,
        )
        complain = st.selectbox(
            "Complain",
            options=[0, 1],
            index=int(defaults["Complain"]) if int(defaults["Complain"]) in (0, 1) else 0,
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
        )
        login = st.selectbox(
            "PreferredLoginDevice",
            options=LOGIN_OPTIONS,
            index=LOGIN_OPTIONS.index(defaults["PreferredLoginDevice"])
            if defaults["PreferredLoginDevice"] in LOGIN_OPTIONS
            else 1,
        )
        payment = st.selectbox(
            "PreferredPaymentMode",
            options=PAYMENT_OPTIONS,
            index=PAYMENT_OPTIONS.index(defaults["PreferredPaymentMode"])
            if defaults["PreferredPaymentMode"] in PAYMENT_OPTIONS
            else 0,
        )
        gender = st.selectbox(
            "Gender",
            options=GENDER_OPTIONS,
            index=GENDER_OPTIONS.index(defaults["Gender"])
            if defaults["Gender"] in GENDER_OPTIONS
            else 0,
        )
        order_cat = st.selectbox(
            "PreferedOrderCat",
            options=ORDER_CAT_OPTIONS,
            index=ORDER_CAT_OPTIONS.index(defaults["PreferedOrderCat"])
            if defaults["PreferedOrderCat"] in ORDER_CAT_OPTIONS
            else 0,
        )
        marital = st.selectbox(
            "MaritalStatus",
            options=MARITAL_OPTIONS,
            index=MARITAL_OPTIONS.index(defaults["MaritalStatus"])
            if defaults["MaritalStatus"] in MARITAL_OPTIONS
            else 1,
        )

    form_payload = {
        "Tenure": tenure,
        "WarehouseToHome": warehouse,
        "HourSpendOnApp": hours,
        "NumberOfDeviceRegistered": devices,
        "NumberOfAddress": addresses,
        "OrderAmountHikeFromlastYear": hike,
        "CouponUsed": coupons,
        "OrderCount": orders,
        "DaySinceLastOrder": days,
        "CashbackAmount": cashback,
        "CityTier": city,
        "SatisfactionScore": satisfaction,
        "Complain": complain,
        "PreferredLoginDevice": login,
        "PreferredPaymentMode": payment,
        "Gender": gender,
        "PreferedOrderCat": order_cat,
        "MaritalStatus": marital,
    }

    if st.button("Run all three models", type="primary", key="score_form"):
        try:
            with st.spinner("Preprocessing + scoring…"):
                scores, snapshot, _bundle = _score_raw_row(form_payload)
            st.session_state["last_scores"] = scores
            st.session_state["last_snapshot"] = snapshot
            st.session_state["last_source"] = "form"
            render_raw_profile(snapshot, title="Raw profile (form + engineered flags)")
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

# Shared results + retention panel (persist across tab switches via session_state)
if "last_scores" in st.session_state:
    st.divider()
    source = st.session_state.get("last_source", "")
    st.subheader("Model comparison")
    if source:
        st.caption(f"Last run: `{source}`")
    render_score_cards(st.session_state["last_scores"])
    st.dataframe(
        scores_table(st.session_state["last_scores"]),
        hide_index=True,
        width="stretch",
    )

    try:
        preprocess_bundle = get_preprocess_bundle()
    except FileNotFoundError:
        preprocess_bundle = None

    render_retention_panel(
        scores=st.session_state["last_scores"],
        raw_snapshot=st.session_state.get("last_snapshot", {}),
        preprocess_bundle=preprocess_bundle,
        default_model="tabnet",
    )
else:
    st.info("Pick a test CustomerID or fill the form, then run the models.")
