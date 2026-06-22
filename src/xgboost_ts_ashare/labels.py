"""Target-label construction — fixed: last row is dropped, not zero-filled."""

from __future__ import annotations

import pandas as pd


def build_labels(df: pd.DataFrame, threshold: float = 0.002) -> pd.DataFrame:
    """Compute next-day binary target and drop the last row (no future data).

    Returns a DataFrame with added ``target`` column (int8) and ``future_return``.
    The last row is dropped — its future return is unknown.
    """
    df = df.copy()
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1.0

    # Drop the final row — its future return is unknown.
    df = df.dropna(subset=["future_return"]).copy()

    df["target"] = (df["future_return"] >= threshold).astype("int8")
    return df
