"""Tests for data loading from data lake."""

from __future__ import annotations

from pathlib import Path

from xgboost_ts_ashare.config import Settings


def test_data_lake_root_path_exists() -> None:
    """Default data_lake_root should point to a real directory."""
    s = Settings()
    assert s.data_lake_root.exists(), f"Data lake not found: {s.data_lake_root}"


def test_cache_file_path_is_parameterised() -> None:
    """Different symbols produce different cache paths."""
    s1 = Settings(symbol="000001.SZ", start_date="20210101", end_date="20220101")
    s2 = Settings(symbol="000002.SZ", start_date="20210101", end_date="20220101")
    assert s1.cache_file != s2.cache_file


def test_cache_file_path_mkdir() -> None:
    """cache_file should create the cache directory if it doesn't exist."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        s = Settings(
            symbol="TEST",
            start_date="20200101",
            end_date="20200110",
            cache_dir=Path(tmp),
        )
        assert s.cache_file.parent.exists()


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
    assert s.instruments_path.exists(), f"Instruments not found: {s.instruments_path}"


def test_industry_join_on_real_data() -> None:
    """Loading real data should produce an 'industry' column."""
    s = Settings(symbol="000001.SZ", start_date="20250101", join_industry=True)

    from xgboost_ts_ashare.data import load_data
    df = load_data(s)
    assert "industry" in df.columns, "industry column missing after join"
    assert df["industry"].iloc[0] != "其他", "000001.SZ should have a real industry"
    assert df["industry"].notna().all(), "industry column has NaN"
