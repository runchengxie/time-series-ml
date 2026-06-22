"""Tests for label construction — the target bug fix is verified here."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xgboost_ts_ashare.labels import build_labels


def make_ohlcv(n_rows: int = 10) -> pd.DataFrame:
    """Synthetic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    close = 100 + rng.normal(0, 2, n_rows).cumsum()
    return pd.DataFrame({"close": close})


def test_last_row_without_future_return_is_dropped() -> None:
    """The last row has no future return — it must be removed entirely."""
    df = make_ohlcv(10)
    original_len = len(df)

    result = build_labels(df, threshold=0.002)

    assert len(result) == original_len - 1, (
        f"Expected {original_len - 1} rows after drop, got {len(result)}"
    )
    assert result["target"].isna().sum() == 0
    assert set(result["target"].unique()).issubset({0, 1})


def test_target_values_are_correct() -> None:
    """For a known price series, labels should be deterministic."""
    df = pd.DataFrame({"close": [100.0, 100.5, 100.1, 100.8, 100.3]})

    result = build_labels(df, threshold=0.002)

    expected = [1, 0, 1, 0]
    assert result["target"].tolist() == expected, f"Got {result['target'].tolist()}"


def test_future_return_column_present() -> None:
    """future_return is kept for IC computation; it should be present."""
    df = make_ohlcv(10)
    result = build_labels(df)
    assert "future_return" in result.columns


def test_threshold_respected() -> None:
    """Changing the threshold changes the labels."""
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})

    result_up = build_labels(df.copy(), threshold=0.005)
    assert result_up["target"].tolist() == [1, 1], f"Got {result_up['target'].tolist()}"

    result_down = build_labels(df.copy(), threshold=0.02)
    assert result_down["target"].tolist() == [0, 0], f"Got {result_down['target'].tolist()}"
