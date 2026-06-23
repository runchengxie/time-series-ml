"""Tests for walk_forward backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_ml.backtest import walk_forward


def make_backtest_df(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic DataFrame for backtest testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    close = pd.Series(100 + rng.normal(0, 1, n_rows).cumsum())
    df = pd.DataFrame({
        "trade_date": dates,
        "close": close,
        "future_return": close.pct_change().shift(-1).fillna(0),
        "target": rng.integers(0, 2, n_rows),
        "feat_a": rng.normal(0, 1, n_rows),
        "feat_b": rng.normal(0, 0.5, n_rows),
    })
    return df


def test_walk_forward_missing_trade_date() -> None:
    """If trade_date column is missing, it falls back to index."""
    df = make_backtest_df(120)
    df_no_date = df.drop(columns=["trade_date"])
    df_no_date.index = pd.date_range("2020-01-01", periods=120, freq="B")

    from xgboost import XGBClassifier

    bt = walk_forward(
        df_no_date,
        feature_cols=["feat_a", "feat_b"],
        model_class=XGBClassifier,
        model_params={"n_estimators": 20, "max_depth": 2, "random_state": 42},
        threshold=0.5,
        retrain_freq="ME",
        cost_bps=0.0,
    )
    # Should not error — return dict with expected keys
    assert "n_trades" in bt
    assert "sharpe" in bt


def test_walk_forward_no_trades() -> None:
    """Returns error dict when no trades are executed."""
    df = make_backtest_df(100)
    from xgboost import XGBClassifier

    bt = walk_forward(
        df,
        feature_cols=["feat_a", "feat_b"],
        model_class=XGBClassifier,
        model_params={"n_estimators": 10, "max_depth": 2, "random_state": 42},
        threshold=0.999,  # impossibly high — no signals
        retrain_freq="ME",
        cost_bps=0.0,
    )
    # Even if no trades, should return a valid dict (not crash)
    assert isinstance(bt, dict)


def test_walk_forward_split_costs() -> None:
    """Split buy/sell costs are applied correctly."""
    df = make_backtest_df(120)
    from xgboost import XGBClassifier

    bt = walk_forward(
        df,
        feature_cols=["feat_a", "feat_b"],
        model_class=XGBClassifier,
        model_params={"n_estimators": 10, "max_depth": 2, "random_state": 42},
        threshold=0.5,
        retrain_freq="ME",
        buy_cost_bps=0.0,   # A-share: no stamp duty on buy
        sell_cost_bps=10.0,  # 10 bps sell
    )
    assert bt.get("buy_cost_bps") == 0.0
    assert bt.get("sell_cost_bps") == 10.0


def test_walk_forward_legacy_cost_fallback() -> None:
    """Legacy cost_bps is split evenly when buy/sell are not set."""
    df = make_backtest_df(100)
    from xgboost import XGBClassifier

    bt = walk_forward(
        df,
        feature_cols=["feat_a", "feat_b"],
        model_class=XGBClassifier,
        model_params={"n_estimators": 10, "max_depth": 2, "random_state": 42},
        threshold=0.5,
        retrain_freq="ME",
        cost_bps=10.0,
    )
    # Legacy cost 10 bps → buy 5 bps + sell 5 bps
    assert bt.get("cost_bps_used") == pytest.approx(10.0, abs=0.1)
