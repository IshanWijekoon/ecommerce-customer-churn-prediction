"""Shared fixtures for scoring-artifact tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

RAW_COLUMNS = [
    "CustomerID",
    "Churn",
    "Tenure",
    "PreferredLoginDevice",
    "CityTier",
    "WarehouseToHome",
    "PreferredPaymentMode",
    "Gender",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "PreferedOrderCat",
    "SatisfactionScore",
    "MaritalStatus",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]


@pytest.fixture
def synthetic_customers() -> pd.DataFrame:
    """Tiny labelled table with the same columns as notebook 1's step1 CSV."""
    rng = np.random.default_rng(42)
    n = 80
    login = np.array(["Mobile Phone", "Computer", "Phone"])
    pay = np.array(["Debit Card", "Credit Card", "CC", "COD", "UPI", "E wallet"])
    gender = np.array(["Male", "Female"])
    cat = np.array(["Laptop & Accessory", "Mobile", "Fashion", "Grocery", "Others"])
    marital = np.array(["Single", "Married", "Divorced"])

    churn = np.zeros(n, dtype=int)
    churn[:14] = 1
    rng.shuffle(churn)

    df = pd.DataFrame(
        {
            "CustomerID": np.arange(50001, 50001 + n),
            "Churn": churn,
            "Tenure": rng.uniform(0, 30, n).round(1),
            "PreferredLoginDevice": rng.choice(login, n),
            "CityTier": rng.choice([1, 2, 3], n),
            "WarehouseToHome": rng.uniform(5, 40, n).round(1),
            "PreferredPaymentMode": rng.choice(pay, n),
            "Gender": rng.choice(gender, n),
            "HourSpendOnApp": rng.uniform(1, 5, n).round(1),
            "NumberOfDeviceRegistered": rng.integers(1, 6, n),
            "PreferedOrderCat": rng.choice(cat, n),
            "SatisfactionScore": rng.integers(1, 6, n),
            "MaritalStatus": rng.choice(marital, n),
            "NumberOfAddress": rng.integers(1, 10, n),
            "Complain": rng.integers(0, 2, n),
            "OrderAmountHikeFromlastYear": rng.uniform(10, 25, n).round(1),
            "CouponUsed": rng.integers(0, 8, n),
            "OrderCount": rng.integers(1, 12, n),
            "DaySinceLastOrder": rng.integers(0, 20, n),
            "CashbackAmount": rng.uniform(100, 300, n).round(2),
        }
    )
    df.loc[0, "Tenure"] = np.nan
    df.loc[1, "PreferredLoginDevice"] = np.nan
    return df[RAW_COLUMNS]
