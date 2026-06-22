"""Walk-forward backtest with transaction cost analysis and signal filtering.

Implements a purged walk-forward framework: at each retrain date, the model
is refit on all prior data and used to generate signals for the next period.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_class: Any,
    model_params: dict[str, Any],
    threshold: float = 0.5,
    retrain_freq: str = "ME",
    cost_bps: float = 5.0,
    purge_days: int = 20,
    regime_labels: pd.Series | None = None,
) -> dict[str, Any]:
    """Purged walk-forward backtest.

    Parameters
    ----------
    df : DataFrame
        Must include 'trade_date' (datetime), feature columns, 'target' (0/1),
        'future_return' (float), and a 'close' column.
    threshold : float
        Probability threshold for generating buy signals (0.55+ recommended
        after calibration to filter low-confidence predictions).
    retrain_freq : str
        Pandas offset alias ('ME' = monthly, 'W' = weekly).
    cost_bps : float
        Round-trip transaction cost in basis points (e.g. 5 = 0.05%).
    regime_labels : pd.Series, optional
        Integer regime labels (1=bull, 0=range, -1=bear) aligned to df's index.
        If provided, per-regime statistics are included in the output.

    Returns
    -------
    dict with keys: total_return, annual_return, annual_vol, sharpe,
    max_drawdown, win_rate, profit_factor, n_trades, turnover, equity_curve,
    monthly_returns, signal_rate, regime_stats (if regime_labels provided).
    """
    if "trade_date" not in df.columns:
        df = df.copy()
        df["trade_date"] = pd.to_datetime(df.index)

    df = df.sort_values("trade_date").reset_index(drop=True)

    periods = pd.date_range(
        start=df["trade_date"].min(),
        end=df["trade_date"].max(),
        freq=retrain_freq,
    )

    trades: list[dict[str, Any]] = []
    total_signals = 0
    equity = 1.0
    equity_curve: list[float] = [1.0]
    dates: list[pd.Timestamp] = [df["trade_date"].iloc[0]]

    # Align regime labels to df index (regime_labels has DatetimeIndex, df has RangeIndex)
    aligned_regime: pd.Series | None = None
    if regime_labels is not None:
        aligned_regime = _align_regime_to_df_index(regime_labels, df)

    for period_start in periods:
        train_mask = df["trade_date"] < period_start
        train_idx = df[train_mask].index
        if purge_days > 0 and len(train_idx) > purge_days:
            train_idx = train_idx[:-purge_days]

        period_end = period_start + pd.tseries.frequencies.to_offset(retrain_freq)
        test_mask = (df["trade_date"] >= period_start) & (df["trade_date"] < period_end)

        if train_mask.sum() < 100 or test_mask.sum() < 5:
            continue

        X_train = df.loc[train_idx, feature_cols]
        y_train = df.loc[train_idx, "target"]
        X_test = df.loc[test_mask, feature_cols]
        returns_test = df.loc[test_mask, "future_return"]
        test_dates = df.loc[test_mask, "trade_date"]

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        signal = (prob >= threshold).astype(int)
        total_signals += signal.sum()

        for i in range(len(signal)):
            if signal[i] == 1:
                ret = returns_test.iloc[i]
                cost = cost_bps / 10000.0
                net_ret = ret - cost
                trade_record: dict[str, Any] = {
                    "date": test_dates.iloc[i],
                    "return": float(net_ret),
                }
                # Attach regime label if available
                if aligned_regime is not None:
                    df_idx = returns_test.index[i]
                    if df_idx in aligned_regime.index:
                        trade_record["regime"] = int(aligned_regime.loc[df_idx])
                trades.append(trade_record)
                equity *= (1.0 + net_ret)
            equity_curve.append(equity)
            dates.append(test_dates.iloc[i])

    metrics = _compute_backtest_metrics(trades, equity_curve, dates)

    # Signal rate: what fraction of test days generated a trade signal
    n_test_days = sum(
        1
        for p in periods
        for _ in df[
            (df["trade_date"] >= p)
            & (df["trade_date"] < p + pd.tseries.frequencies.to_offset(retrain_freq))
        ].index
    )
    metrics["signal_rate"] = total_signals / max(n_test_days, 1)
    metrics["prob_threshold_used"] = threshold

    # Per-regime statistics
    if regime_labels is not None and trades:
        metrics["regime_stats"] = _compute_regime_stats(trades, dates)

    return metrics


def _align_regime_to_df_index(
    regime_labels: pd.Series, df: pd.DataFrame
) -> pd.Series:
    """Align DatetimeIndex-backed regime labels to df's RangeIndex via trade_date."""
    date_to_regime = dict(zip(regime_labels.index, regime_labels.values, strict=False))
    aligned = df["trade_date"].map(date_to_regime)  # type: ignore[return-value]
    aligned.index = df.index  # type: ignore[assignment]
    return pd.Series(aligned, dtype=float)


def _compute_backtest_metrics(
    trades: list[dict[str, Any]],
    equity_curve: list[float],
    dates: list[pd.Timestamp],
) -> dict[str, Any]:
    n_trades = len(trades)
    if n_trades == 0:
        return {"error": "No trades executed", "n_trades": 0}

    returns = np.array([t["return"] for t in trades])
    total_return = np.prod(1.0 + returns) - 1.0

    n_years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1.0 + total_return) ** (1.0 / max(n_years, 0.25)) - 1.0
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    annual_vol = float(np.std(daily_returns) * np.sqrt(252))

    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (np.array(equity_curve) - peak) / peak
    max_drawdown = float(drawdown.min())

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / n_trades
    profit_factor = (
        abs(wins.sum()) / abs(losses.sum()) if len(losses) > 0 else float("inf")
    )

    turnover = n_trades / max(n_years, 0.1)

    eq_series = pd.Series(equity_curve, index=dates)
    monthly = eq_series.resample("ME").last().pct_change().dropna()

    return {
        "n_trades": n_trades,
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "turnover": float(turnover),
        "cost_bps_used": 5.0,
        "equity_curve": equity_curve,
        "monthly_returns": monthly.tolist(),
    }


def _compute_regime_stats(
    trades: list[dict[str, Any]],
    dates: list[pd.Timestamp],
) -> dict[int, dict[str, float]]:
    """Compute per-regime backtest statistics.

    Returns a dict keyed by regime label (1=bull, 0=range, -1=bear),
    each value is a dict with: n_trades, win_rate, total_return, sharpe.
    """
    from collections import defaultdict

    regime_trades: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        r = t.get("regime")
        if r is not None:
            regime_trades[r].append(t["return"])

    n_years = max((dates[-1] - dates[0]).days / 365.25, 0.25)
    stats: dict[int, dict[str, float]] = {}

    for regime, returns in regime_trades.items():
        if not returns:
            continue
        arr = np.array(returns)
        n = len(arr)
        wins = arr[arr > 0]
        total_ret = float(np.prod(1.0 + arr) - 1.0)
        ann_ret = (1.0 + total_ret) ** (1.0 / n_years) - 1.0
        ann_vol = float(np.std(arr) * np.sqrt(252))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        stats[regime] = {
            "n_trades": n,
            "win_rate": float(len(wins) / n) if n > 0 else 0.0,
            "total_return": total_ret,
            "sharpe": sharpe,
        }

    return stats
