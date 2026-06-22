"""Feature engineering — self-contained, no pandas_ta dependency."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, length: int) -> pd.Series:  # pyright: ignore[reportReturnType]
    """Simple Moving Average."""
    return cast(pd.Series, series.rolling(window=length, min_periods=length).mean())


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:  # pyright: ignore[reportReturnType]
    """Relative Strength Index (Wilder's smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return cast(pd.Series, 100.0 - (100.0 / (1.0 + rs)))


def compute_macd_hist(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """MACD histogram = MACD line - signal line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = cast(pd.Series, ema_fast - ema_slow)
    signal_line = cast(pd.Series, macd_line.ewm(span=signal, adjust=False).mean())
    return cast(pd.Series, macd_line - signal_line)


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = cast(pd.Series, pd.concat([tr1, tr2, tr3], axis=1).max(axis=1))
    return cast(pd.Series, true_range.rolling(window=length, min_periods=length).mean())


def compute_historical_volatility(close: pd.Series, length: int = 20) -> pd.Series:
    """Annualised historical volatility from log returns."""
    log_ret = np.log(close / close.shift(1))
    return cast(
        pd.Series,
        log_ret.rolling(window=length, min_periods=length).std() * np.sqrt(252),
    )


def compute_candle_features(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> dict[str, pd.Series]:
    """Candlestick body and shadow ratios (continuous, not discrete patterns).

    Returns dict with:
      - body_ratio: |close-open| / (high-low), 0-1.  Near 1 = marubozu.
      - upper_shadow: (high-max(open,close)) / (high-low), 0-1.
      - lower_shadow: (min(open,close)-low) / (high-low), 0-1.
    """
    body = (close - open_).abs()
    upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
    candle_range = high - low
    candle_range_safe = candle_range.replace(0, float("nan"))

    return {
        "body_ratio": body / candle_range_safe,
        "upper_shadow_ratio": upper_shadow / candle_range_safe,
        "lower_shadow_ratio": lower_shadow / candle_range_safe,
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical-indicator columns to a DataFrame with OHLCV columns.

    Expects columns: open, high, low, close, vol.
    Returns a new DataFrame (does not mutate input).
    """
    df = df.copy()

    close_s = cast(pd.Series, df["close"])
    vol_s = cast(pd.Series, df["vol"])

    has_ohlc = all(c in df.columns for c in ("open", "high", "low"))

    # --- Trend / Momentum ---
    for win in (5, 10, 20):
        sma_col = f"SMA{win}"
        df[sma_col] = compute_sma(close_s, length=win)
        df[f"{sma_col}_diff"] = df[sma_col].pct_change()

    df["RSI_14"] = compute_rsi(close_s, length=14)
    df["MACD_hist"] = compute_macd_hist(close_s)

    # --- Volume ---
    df["Volume_SMA5"] = compute_sma(vol_s, length=5)
    df["Volume_SMA5_ratio"] = vol_s / df["Volume_SMA5"].replace(0, float("nan"))

    # --- Volatility / Risk ---
    if has_ohlc:
        high_s = cast(pd.Series, df["high"])
        low_s = cast(pd.Series, df["low"])

        df["ATR_14"] = compute_atr(high_s, low_s, close_s, length=14)
        df["ATR_14_pct"] = df["ATR_14"] / close_s

    df["HistVol_20"] = compute_historical_volatility(close_s, length=20)

    # --- Candle body / shadow ratios ---
    if has_ohlc:
        candle = compute_candle_features(
            cast(pd.Series, df["open"]),
            cast(pd.Series, df["high"]),
            cast(pd.Series, df["low"]),
            close_s,
        )
        for k, v in candle.items():
            df[k] = v

    return df


# Public list of feature column names used for model input
FEATURE_COLUMNS = [
    # Trend / momentum
    "SMA5_diff",
    "SMA10_diff",
    "SMA20_diff",
    "RSI_14",
    "MACD_hist",
    # Volume
    "Volume_SMA5_ratio",
    "vol",
    # Volatility
    "ATR_14_pct",
    "HistVol_20",
    # Candle structure
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
]
