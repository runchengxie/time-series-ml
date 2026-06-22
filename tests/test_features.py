"""Tests for feature engineering — no NaN in output columns after dropna."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xgboost_ts_ashare.features import FEATURE_COLUMNS, build_features


def make_ohlcv(n_rows: int = 200) -> pd.DataFrame:
    """Synthetic OHLCV data with enough rows for all rolling windows."""
    rng = np.random.default_rng(42)
    close = 100 + rng.normal(0, 2, n_rows).cumsum()
    vol = rng.integers(1_000_000, 10_000_000, n_rows)
    noise = rng.normal(0, 0.5, n_rows)
    open_ = close + noise * 0.3
    high = np.maximum(open_, close) + np.abs(noise) * 0.5
    low = np.minimum(open_, close) - np.abs(noise) * 0.5
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "vol": vol,
    })
    df["trade_date"] = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    return df


def test_all_feature_columns_present() -> None:
    df = make_ohlcv(200)
    result = build_features(df)
    for col in FEATURE_COLUMNS:
        assert col in result.columns, f"Missing feature column: {col}"


def test_features_no_nan_after_enough_rows() -> None:
    df = make_ohlcv(200)
    result = build_features(df)
    clean = result[FEATURE_COLUMNS].dropna()
    assert len(clean) > 0, "All feature rows are NaN"
    assert clean.isna().sum().sum() == 0, "NaN values remain after dropna"


def test_volume_ratio_no_inf() -> None:
    df = make_ohlcv(200)
    result = build_features(df)
    ratio = result["Volume_SMA5_ratio"].dropna()
    assert not np.isinf(ratio).any(), "Volume ratio contains inf"


def test_rsi_bounds() -> None:
    """RSI should always be in [0, 100]."""
    df = make_ohlcv(200)
    result = build_features(df)
    rsi = result["RSI_14"].dropna()
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


def test_input_not_mutated() -> None:
    df = make_ohlcv(50)
    original_cols = set(df.columns)
    _ = build_features(df)
    assert set(df.columns) == original_cols, "build_features mutated input DataFrame"
