"""Market regime classification using moving-average-based rules.

Regime labels: 1 = bull (uptrend), 0 = range (sideways), -1 = bear (downtrend).

The classification uses two moving averages and a deviation threshold:
  - Bull: close > MA_fast > MA_slow AND close > MA_slow * (1 + threshold)
  - Bear: close < MA_fast < MA_slow AND close < MA_slow * (1 - threshold)
  - Range: everything else

No external dependencies beyond pandas/numpy.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

REGIME_LABELS: dict[int, str] = {1: "bull", 0: "range", -1: "bear"}


def compute_market_proxy(
    dfs: list[pd.DataFrame],
    price_col: str = "close",
) -> pd.Series:
    """Compute equal-weighted daily average close from a pool of stock DataFrames.

    Each DataFrame must have 'trade_date' and `price_col` columns.

    Returns a Series indexed by trade_date with the cross-sectional mean close.
    """
    if not dfs:
        return pd.Series(dtype=float)

    # Stack all closes into a wide panel: rows=dates, columns=symbols
    panels: list[pd.DataFrame] = []
    for df in dfs:
        if "trade_date" not in df.columns or price_col not in df.columns:
            continue
        sub = df[["trade_date", price_col]].copy()
        sub = sub.set_index("trade_date")
        sub.columns = [price_col]
        panels.append(sub)

    if not panels:
        return pd.Series(dtype=float)

    combined = pd.concat(panels, axis=1)
    proxy: pd.Series = combined.mean(axis=1, skipna=True)  # type: ignore[assignment]
    proxy = cast(pd.Series, proxy.dropna())
    proxy.index = pd.to_datetime(proxy.index)
    return cast(pd.Series, proxy.sort_index())


def classify_regime(
    market_close: pd.Series,
    ma_fast: int = 20,
    ma_slow: int = 60,
    threshold: float = 0.03,
) -> pd.Series:
    """Classify each day as bull (1), range (0), or bear (-1).

    Parameters
    ----------
    market_close : pd.Series
        Daily close prices of the market proxy, indexed by date.
    ma_fast : int
        Fast moving average window (default 20 days ≈ 1 month).
    ma_slow : int
        Slow moving average window (default 60 days ≈ 1 quarter).
    threshold : float
        Minimum deviation from slow MA to qualify as bull/bear.
        Default 0.03 = ±3%.

    Returns
    -------
    pd.Series
        Integer labels: 1 (bull), 0 (range), -1 (bear).
        NaN for days where MAs are not yet available.
    """
    ma_fast_series = market_close.rolling(ma_fast, min_periods=ma_fast).mean()
    ma_slow_series = market_close.rolling(ma_slow, min_periods=ma_slow).mean()

    deviation = (market_close - ma_slow_series) / ma_slow_series

    regime = pd.Series(0, index=market_close.index, dtype=int)

    # Bull: price above fast MA, fast MA above slow MA, > threshold above slow
    bull_mask = (
        (market_close > ma_fast_series)
        & (ma_fast_series > ma_slow_series)
        & (deviation > threshold)
    )
    regime[bull_mask] = 1

    # Bear: price below fast MA, fast MA below slow MA, < -threshold below slow
    bear_mask = (
        (market_close < ma_fast_series)
        & (ma_fast_series < ma_slow_series)
        & (deviation < -threshold)
    )
    regime[bear_mask] = -1

    return regime


def regime_summary(regime: pd.Series) -> dict[str, float]:
    """Return the fraction of days in each regime."""
    total = len(regime.dropna())
    if total == 0:
        return {}
    counts = regime.value_counts()
    return {
        REGIME_LABELS.get(k, str(k)): float(counts.get(k, 0) or 0) / total
        for k in [1, 0, -1]
    }


def align_regime_to_df(
    regime: pd.Series,
    df: pd.DataFrame,
    date_col: str = "trade_date",
) -> pd.Series:
    """Align regime labels to a DataFrame's trade_date index.

    Returns a Series with the same length as df, where each row gets
    the regime label for its trade_date. Unmatched dates get NaN.
    """
    if date_col not in df.columns or regime.empty:
        return pd.Series(index=df.index, dtype=float)

    lookup = dict(zip(regime.index, regime.values, strict=False))
    aligned = df[date_col].map(lambda x: lookup.get(x))  # type: ignore[arg-type]
    aligned.index = df.index  # type: ignore[assignment]
    return cast(pd.Series, aligned)
