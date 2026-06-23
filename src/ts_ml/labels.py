"""Target-label construction with Triple Barrier Labeling.

Implements the method from Advances in Financial Machine Learning (Lopez de Prado):
  - Profit-taking barrier: +profit_take (e.g. +5%)
  - Stop-loss barrier: -stop_loss (e.g. -3%)
  - Time barrier: holding_period days (e.g. 5 days)

The label is:
  +1  if the profit barrier is hit first
  -1  if the stop barrier is hit first
   0  if neither barrier is hit within the holding period

Path reconstruction from daily OHLC (no tick data required):
  For each day in the holding period, checks whether the daily high crosses
  the profit barrier or the daily low crosses the stop barrier. When both
  barriers are crossed on the same day, uses the relative position of the
  open within the day's range to determine which barrier was breached first.

Also retains the legacy binary label mode for backward compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _first_barrier_hit(
    open_: float,
    high: float,
    low: float,
    close: float,
    entry_price: float,
    profit_price: float,
    stop_price: float,
) -> int:
    """Determine which barrier is hit first on a single day.

    Uses daily OHLC to reconstruct intraday path without tick data.

    Returns:
        +1 if profit barrier hit first
        -1 if stop barrier hit first
        0 if neither barrier hit
    """
    # Gap open — cross barrier at open
    if open_ >= profit_price:
        return 1
    if open_ <= stop_price:
        return -1

    # Intraday extremes
    profit_hit = high >= profit_price
    stop_hit = low <= stop_price

    if profit_hit and not stop_hit:
        return 1
    if stop_hit and not profit_hit:
        return -1

    if profit_hit and stop_hit:
        # Both barriers crossed — determine which was first
        # Compare relative position of open within the day's range
        daily_range = high - low
        if daily_range <= 0:
            # Degenerate case: use close as tiebreaker
            return 1 if close >= entry_price else -1

        # How far did price travel upward vs downward from open?
        up_travel = (high - open_) / daily_range
        down_travel = (open_ - low) / daily_range

        # The barrier that's closer to open is hit first
        # Distance from open to profit vs stop (normalised by barrier distances)
        dist_to_profit = (profit_price - open_) / (profit_price - entry_price + 1e-10)
        dist_to_stop = (open_ - stop_price) / (entry_price - stop_price + 1e-10)

        # If price spent more time going up relative to barrier distances,
        # profit was hit first
        if up_travel * dist_to_stop >= down_travel * dist_to_profit:
            return 1
        else:
            return -1

    return 0


def build_triple_barrier_labels(
    df: pd.DataFrame,
    holding_period: int = 5,
    profit_take: float = 0.05,
    stop_loss: float = 0.03,
    price_col: str = "close",
) -> pd.DataFrame:
    """Build triple-barrier labels for each row.

    For each trading day t with entry price = close[t]:
      - Look at the next holding_period days
      - Profit barrier = entry * (1 + profit_take)
      - Stop barrier   = entry * (1 - stop_loss)
      - Determine which barrier is hit first using daily OHLC

    Labels:
      +1 = profit barrier hit first
      -1 = stop barrier hit first
       0 = neither barrier hit within holding period

    Also adds a 'future_return' column for IC analysis:
      future_return[t] = close[t+1] / close[t] - 1  (single-day return)

    The last ``holding_period`` rows are dropped (no future data).

    Parameters
    ----------
    df : DataFrame
        Must contain: trade_date, open, high, low, close.
    holding_period : int
        Maximum number of days to hold (time barrier).
    profit_take : float
        Profit-taking threshold as a fraction (e.g. 0.05 = 5%).
    stop_loss : float
        Stop-loss threshold as a fraction (e.g. 0.03 = 3%).
    price_col : str
        Column name for the entry/close price.

    Returns
    -------
    DataFrame
        Original columns plus 'target' (int8: -1/0/+1) and 'future_return'.
        Rows near the end without enough future data are dropped.
    """
    df = df.copy()
    n = len(df)

    # We need holding_period days of future data
    if n <= holding_period:
        return df.iloc[:0]

    targets = np.zeros(n, dtype=np.int8)
    future_returns = np.full(n, np.nan)

    close_arr = df[price_col].values
    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values

    for t in range(n - holding_period):
        entry_price = close_arr[t]
        profit_price = entry_price * (1.0 + profit_take)
        stop_price = entry_price * (1.0 - stop_loss)

        # Single-day future return for IC analysis
        future_returns[t] = close_arr[t + 1] / entry_price - 1.0

        # Check each day in the holding period
        barrier_hit = 0
        for k in range(1, holding_period + 1):
            day = t + k
            hit = _first_barrier_hit(
                open_arr[day], high_arr[day], low_arr[day], close_arr[day],
                entry_price, profit_price, stop_price,
            )
            if hit != 0:
                barrier_hit = hit
                break

        targets[t] = barrier_hit

    df["target"] = targets
    df["future_return"] = future_returns

    # Drop rows without valid labels (last holding_period rows)
    df = df.iloc[: n - holding_period].copy()

    # Report label distribution
    unique, counts = np.unique(targets[: n - holding_period], return_counts=True)
    dist = dict(zip(unique, counts, strict=False))
    total = n - holding_period
    label_names = {-1: "stop_loss", 0: "timeout", 1: "profit_take"}
    parts = []
    for lbl in [-1, 0, 1]:
        cnt = dist.get(lbl, 0)
        pct = cnt / total * 100 if total > 0 else 0
        parts.append(f"{label_names[lbl]}: {cnt} ({pct:.1f}%)")
    print(f"[labels] Triple barrier K={holding_period}d "
          f"pt={profit_take:.0%} sl={stop_loss:.0%} — "
          f"{', '.join(parts)}")

    return df


def build_labels(
    df: pd.DataFrame,
    threshold: float = 0.002,
    holding_period: int = 1,
    profit_take: float = 0.002,
    stop_loss: float = 0.002,
    triple_barrier: bool = False,
) -> pd.DataFrame:
    """Compute target labels. Supports both legacy binary and triple barrier.

    Legacy mode (triple_barrier=False):
      target[t] = 1 if close[t+1] / close[t] - 1 >= threshold else 0

    Triple barrier mode (triple_barrier=True):
      Uses profit_take, stop_loss, and holding_period to generate -1/0/+1 labels.

    Parameters
    ----------
    df : DataFrame
        Must contain: trade_date, close.  For triple barrier also needs open, high, low.
    threshold : float
        Legacy binary threshold (e.g. 0.002 = +0.2%).
    holding_period : int
        Number of days to hold for triple barrier.
    profit_take : float
        Profit barrier as fraction (e.g. 0.05 = 5%).
    stop_loss : float
        Stop barrier as fraction (e.g. 0.03 = 3%).
    triple_barrier : bool
        If True, use triple barrier labeling instead of binary.

    Returns
    -------
    DataFrame
        With added 'target' (int8) and 'future_return' columns.
        Last rows without future data are dropped.
    """
    if triple_barrier:
        return build_triple_barrier_labels(
            df,
            holding_period=holding_period,
            profit_take=profit_take,
            stop_loss=stop_loss,
        )

    # Legacy binary labeling
    df = df.copy()
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1.0
    df = df.dropna(subset=["future_return"]).copy()
    df["target"] = (df["future_return"] >= threshold).astype("int8")
    return df
