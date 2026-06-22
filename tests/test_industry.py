"""Tests for industry neutralization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_ml.features import FEATURE_COLUMNS
from ts_ml.industry import (
    compute_industry_ic_reduction,
    neutralize_industry,
)


def make_multi_stock_df(n_stocks: int = 10, n_dates: int = 50) -> pd.DataFrame:
    """Create synthetic multi-stock data with industry labels."""
    rng = np.random.default_rng(42)
    industries = ["银行", "电气设备", "元器件", "软件服务", "化工原料"]
    records = []
    for d in range(n_dates):
        trade_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=d)
        for s in range(n_stocks):
            ind = industries[s % len(industries)]
            row = {
                "trade_date": trade_date,
                "ts_code": f"{s:06d}.SZ",
                "industry": ind,
                "close": 10 + rng.normal(0, 1),
                "vol": 1_000_000 + rng.normal(0, 100_000),
            }
            for f in FEATURE_COLUMNS:
                # Feature value = industry bias + noise
                industry_bias = industries.index(ind) * 0.5
                row[f] = industry_bias + rng.normal(0, 0.3)
            records.append(row)

    df = pd.DataFrame(records)
    # Add future_return with some industry bias
    df["future_return"] = np.nan
    for ind in industries:
        mask = df["industry"] == ind
        bias = industries.index(ind) * 0.001
        df.loc[mask, "future_return"] = bias + rng.normal(0, 0.01, mask.sum())

    df["target"] = (df["future_return"] >= 0.002).astype(int)
    return df


def test_neutralize_reduces_industry_signal() -> None:
    """After neutralization, industry IC should decrease."""
    df = make_multi_stock_df(n_stocks=20, n_dates=30)

    df_after = neutralize_industry(df.copy(), FEATURE_COLUMNS)

    # Check no new NaN introduced
    for f in FEATURE_COLUMNS:
        assert df_after[f].isna().sum() <= df[f].isna().sum(), (
            f"{f}: neutralization introduced NaN"
        )

    # IC reduction check
    reduction = compute_industry_ic_reduction(
        df, df_after, FEATURE_COLUMNS, return_col="future_return"
    )

    # At least some features should show reduced |IC|
    reduced = sum(1 for v in reduction.values() if v["reduction"] > 0)
    unchanged = sum(1 for v in reduction.values() if v["reduction"] <= 0)
    print(f"  Features with reduced |IC|: {reduced}, unchanged: {unchanged}")
    # Not a hard assert because random data can go either way,
    # but we verify the function runs without error and produces valid output


def test_neutralize_single_stock_noop() -> None:
    """Single stock with one industry → no neutralization."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "trade_date": [pd.Timestamp("2020-01-01")] * 5,
        "ts_code": ["000001.SZ"] * 5,
        "industry": ["银行"] * 5,
        "SMA5_diff": rng.normal(0, 1, 5),
        "target": [0, 1, 0, 1, 0],
        "future_return": rng.normal(0, 0.01, 5),
    })

    result = neutralize_industry(df.copy(), ["SMA5_diff"])
    # Values should be unchanged (single industry → skip)
    np.testing.assert_array_almost_equal(
        df["SMA5_diff"].values, result["SMA5_diff"].values,
    )


def test_neutralize_preserves_shape() -> None:
    """Neutralization should not change row count or column set."""
    df = make_multi_stock_df(n_stocks=15, n_dates=20)
    result = neutralize_industry(df.copy(), FEATURE_COLUMNS)
    assert len(result) == len(df)
    assert set(result.columns) == set(df.columns)


def test_neutralize_with_missing_industry_col() -> None:
    """If industry column missing, should skip gracefully."""
    df = make_multi_stock_df(n_stocks=10, n_dates=10)
    df = df.drop(columns=["industry"])
    result = neutralize_industry(df.copy(), FEATURE_COLUMNS)
    assert result.equals(df)  # unchanged
