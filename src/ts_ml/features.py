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


def build_features(df: pd.DataFrame, *, use_lag: bool = False) -> pd.DataFrame:
    """Add technical-indicator columns to a DataFrame with OHLCV columns.

    Expects columns: open, high, low, close, vol.
    Returns a new DataFrame (does not mutate input).
    """
    df = df.copy()

    close_s = cast(pd.Series, df["close"])
    vol_s = cast(pd.Series, df["vol"])

    has_ohlc = all(c in df.columns for c in ("open", "high", "low"))

    # --- Trend / Momentum ---
    for win in (5, 10, 20, 60):
        sma_col = f"SMA{win}"
        df[sma_col] = compute_sma(close_s, length=win)
        df[f"{sma_col}_diff"] = df[sma_col].pct_change()

    df["RSI_14"] = compute_rsi(close_s, length=14)
    df["MACD_hist"] = compute_macd_hist(close_s)

    # --- Price distance from SMA (normalised) ---
    for win in (5, 20, 60):
        sma_col = f"SMA{win}"
        df[f"price_dist_{win}d"] = (close_s - df[sma_col]) / df[sma_col]

    # --- Volume ---
    df["Volume_SMA5"] = compute_sma(vol_s, length=5)
    df["Volume_SMA5_ratio"] = vol_s / df["Volume_SMA5"].replace(0, float("nan"))
    df["Volume_SMA20"] = compute_sma(vol_s, length=20)
    df["Volume_SMA20_ratio"] = vol_s / df["Volume_SMA20"].replace(0, float("nan"))
    # Volume trend: short/long volume ratio
    df["Volume_trend"] = df["Volume_SMA5"] / df["Volume_SMA20"].replace(0, float("nan"))

    # --- Volatility / Risk ---
    if has_ohlc:
        high_s = cast(pd.Series, df["high"])
        low_s = cast(pd.Series, df["low"])
        df["ATR_14"] = compute_atr(high_s, low_s, close_s, length=14)
        df["ATR_14_pct"] = df["ATR_14"] / close_s

    df["HistVol_20"] = compute_historical_volatility(close_s, length=20)
    df["HistVol_60"] = compute_historical_volatility(close_s, length=60)

    # --- Bollinger Band position ---
    sma20 = df.get("SMA20", compute_sma(close_s, length=20))
    vol20 = close_s.rolling(20, min_periods=20).std()
    df["BB_position"] = (close_s - sma20) / vol20.replace(0, float("nan"))

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

    # --- Gap features ---
    if "pre_close" in df.columns:
        pre_close = cast(pd.Series, df["pre_close"])
        df["gap_pct"] = (close_s.shift(1) - pre_close) / pre_close  # yesterday's gap
        # Today's gap (open vs prev_close)
        if has_ohlc:
            open_s = cast(pd.Series, df["open"])
            df["open_gap"] = (open_s - pre_close) / pre_close

    # --- Recent return skewness (20-day) ---
    ret_1d = close_s.pct_change()
    df["ret_skew_20d"] = ret_1d.rolling(20, min_periods=20).skew()
    df["ret_kurt_20d"] = ret_1d.rolling(20, min_periods=20).kurt()

    # --- Turnover rate (if available) ---
    # Already in daily_clean as turnover_rate_f — just keep as-is
    
    # --- Valuation features (from daily_basic, merged in daily_clean) ---
    if "pe_ttm" in df.columns:
        df["log_pe"] = np.log(df["pe_ttm"].clip(lower=0.01))
    if "pb" in df.columns:
        df["log_pb"] = np.log(df["pb"].clip(lower=0.01))
    if "total_mv" in df.columns:
        df["log_mcap"] = np.log(df["total_mv"].clip(lower=1))

    # --- Volume ratio (from daily_clean) ---
    if "volume_ratio" in df.columns:
        df["volume_ratio_raw"] = df["volume_ratio"]

    # --- Lag features (sequence dependency test for LSTM evaluation) ---
    # Only added when use_lag=True (--use-lag-features).
    if use_lag:
        _lag_features = [
            "SMA20_diff", "RSI_14", "MACD_hist",
            "Volume_SMA5_ratio", "ATR_14_pct", "HistVol_20", "BB_position",
        ]
        _lag_days = [3, 5]
        for lag in _lag_days:
            for feat in _lag_features:
                if feat in df.columns:
                    df[f"{feat}_lag{lag}"] = df[feat].shift(lag)

    return df


# Public list of feature column names used for model input
FEATURE_COLUMNS = [
    # Trend / momentum
    "SMA5_diff",
    "SMA10_diff",
    "SMA20_diff",
    "SMA60_diff",
    "RSI_14",
    "MACD_hist",
    # Price distance from MA
    "price_dist_5d",
    "price_dist_20d",
    "price_dist_60d",
    # Volume
    "Volume_SMA5_ratio",
    "Volume_SMA20_ratio",
    "Volume_trend",
    "vol",
    # Volatility
    "ATR_14_pct",
    "HistVol_20",
    "HistVol_60",
    "BB_position",
    # Candle structure
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    # Gap
    "gap_pct",
    "open_gap",
    # Return distribution
    "ret_skew_20d",
    "ret_kurt_20d",
    # Turnover
    "turnover_rate_f",
    # Volume ratio
    "volume_ratio_raw",
    # Valuation
    "log_pe",
    "log_pb",
    "log_mcap",
]

# Lag features (sequence dependency test for LSTM evaluation).
# Not included in FEATURE_COLUMNS by default; use get_feature_columns(use_lag=True).
_LAG_BASE = [
    "SMA20_diff", "RSI_14", "MACD_hist",
    "Volume_SMA5_ratio", "ATR_14_pct", "HistVol_20", "BB_position",
]
_LAG_DAYS = [3, 5]
_LAG_COLUMNS = [
    f"{feat}_lag{lag}"
    for lag in _LAG_DAYS
    for feat in _LAG_BASE
]


def get_feature_columns(*, use_lag: bool = False) -> list[str]:
    """Return the list of feature columns, optionally including lag features."""
    if use_lag:
        return [*FEATURE_COLUMNS, *_LAG_COLUMNS]
    return list(FEATURE_COLUMNS)
