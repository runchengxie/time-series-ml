"""Tests for data loading from data lake."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ts_ml.config import Settings

# Data lake paths — respect env vars, fall back to hardcoded defaults for local dev.
_DATA_ROOT = Path(
    os.environ.get(
        "TIME_SERIES_ML_DATA_LAKE_ROOT",
        "/home/richard/data/market-data-platform/assets/tushare/"
        "a_share/daily/a_share_all_20150101_20260622_shadow_daily_clean/data",
    )
)
_INSTRUMENTS_PATH = Path(
    os.environ.get(
        "TIME_SERIES_ML_INSTRUMENTS_PATH",
        "/home/richard/data/market-data-platform/assets/tushare/"
        "a_share/instruments/a_share_all_instruments_latest.parquet",
    )
)
_HAS_DATA = _DATA_ROOT.exists() and _INSTRUMENTS_PATH.exists()


def test_data_lake_root_path_exists() -> None:
    """Default data_lake_root should point to a real directory."""
    s = Settings()
    if not _HAS_DATA:
        pytest.skip("Data lake not available in CI")
    assert s.data_lake_root.exists(), f"Data lake not found: {s.data_lake_root}"


def test_end_date_default() -> None:
    """end_date should default to today."""
    from datetime import datetime
    s = Settings(start_date="20250101")
    expected = datetime.now().strftime("%Y%m%d")
    assert s.end_date == expected


def test_calibrate_default_false() -> None:
    s = Settings()
    assert s.calibrate is False
    assert s.cv_method == "isotonic"


def test_prob_threshold_default() -> None:
    s = Settings()
    assert s.prob_threshold == 0.50


def test_instruments_path_exists() -> None:
    """Default instruments_path should exist."""
    s = Settings()
    if not _HAS_DATA:
        pytest.skip("Data lake not available in CI")
    assert s.instruments_path.exists(), f"Instruments not found: {s.instruments_path}"


@pytest.mark.skipif(not _HAS_DATA, reason="Data lake not available in CI")
def test_industry_join_on_real_data() -> None:
    """Loading real data should produce an 'industry' column."""
    s = Settings(symbol="000001.SZ", start_date="20250101", join_industry=True)

    from ts_ml.data import load_data
    df = load_data(s)
    assert "industry" in df.columns, "industry column missing after join"
    assert df["industry"].iloc[0] != "其他", "000001.SZ should have a real industry"
    assert df["industry"].notna().all(), "industry column has NaN"
