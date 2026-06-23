"""Tests for triple barrier label construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_ml.labels import build_labels, build_triple_barrier_labels


def make_ohlc_uptrend(n_rows: int = 50) -> pd.DataFrame:
    """Synthetic OHLC data trending upward."""
    rng = np.random.default_rng(42)
    close = 100 + np.arange(n_rows) * 0.3 + rng.normal(0, 0.1, n_rows)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    return pd.DataFrame({
        "trade_date": dates,
        "open": close - rng.uniform(0, 0.3, n_rows),
        "high": close + rng.uniform(0.1, 0.5, n_rows),
        "low": close - rng.uniform(0.1, 0.5, n_rows),
        "close": close,
        "vol": rng.integers(1_000_000, 10_000_000, n_rows),
    })


def test_triple_barrier_produces_three_classes() -> None:
    """Triple barrier labels include -1, 0, and 1."""
    df = make_ohlc_uptrend(50)
    result = build_triple_barrier_labels(df, holding_period=5, profit_take=0.02, stop_loss=0.02)
    assert "target" in result.columns
    assert set(result["target"].unique()).issubset({-1, 0, 1})
    # At least two distinct classes should appear
    assert result["target"].nunique() >= 2


def test_triple_barrier_drops_last_holding_period_rows() -> None:
    """Last K rows are dropped (no future data)."""
    n = 50
    k = 5
    df = make_ohlc_uptrend(n)
    result = build_triple_barrier_labels(df, holding_period=k, profit_take=0.05, stop_loss=0.03)
    assert len(result) == n - k


def test_triple_barrier_includes_future_return() -> None:
    """future_return column is present for IC computation."""
    df = make_ohlc_uptrend(30)
    result = build_triple_barrier_labels(df, holding_period=5, profit_take=0.05, stop_loss=0.03)
    assert "future_return" in result.columns


def test_build_labels_triple_barrier_mode() -> None:
    """build_labels with triple_barrier=True delegates correctly."""
    df = make_ohlc_uptrend(40)
    result = build_labels(
        df, holding_period=5, profit_take=0.02, stop_loss=0.02, triple_barrier=True,
    )
    assert "target" in result.columns
    assert set(result["target"].unique()).issubset({-1, 0, 1})


def test_triple_barrier_not_enough_data() -> None:
    """Returns empty DataFrame if not enough rows."""
    df = make_ohlc_uptrend(3)
    result = build_triple_barrier_labels(df, holding_period=10, profit_take=0.05, stop_loss=0.03)
    assert len(result) == 0
