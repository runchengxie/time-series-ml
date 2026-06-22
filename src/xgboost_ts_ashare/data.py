"""Data loading from market-data-platform A-share data lake.

Reads per-symbol parquet files from the data lake directory.
Each file: one stock, all trading days, 45 columns.

Optionally joins 申万行业分类 from the instruments table.
"""

from __future__ import annotations

import sys

import pandas as pd

from .config import Settings

# Cache for instruments table (loaded once per process)
_instruments_cache: pd.DataFrame | None = None


def _load_instruments(path: str) -> pd.DataFrame:
    """Load instruments table with caching."""
    global _instruments_cache
    if _instruments_cache is not None:
        return _instruments_cache
    df = pd.read_parquet(path)
    # 328 nulls — fill with "其他"
    df["industry"] = df["industry"].fillna("其他")
    _instruments_cache = df
    return df


def _join_industry(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Join 申万行业分类 from instruments table."""
    instruments = _load_instruments(str(settings.instruments_path))
    industry_map = instruments[["ts_code", "industry"]].copy()

    df = df.merge(industry_map, on="ts_code", how="left")
    df["industry"] = df["industry"].fillna("其他")

    if "industry" in df.columns:
        print(f"[data] Joined 申万行业: {df['industry'].nunique()} industries")
    return df


def load_data(settings: Settings) -> pd.DataFrame:
    """Load daily bars for a single A-share symbol from the data lake.

    Columns expected (subset): trade_date, open, high, low, close, pre_close, vol.

    If settings.join_industry is True, also joins 申万行业分类 from the
    instruments table, adding an 'industry' column.

    Returns a DataFrame sorted chronologically by trade_date,
    with trade_date cast to datetime.
    """
    data_path = settings.data_lake_root / f"{settings.symbol}.parquet"

    if not data_path.exists():
        sys.exit(
            f"Data file not found: {data_path}\n"
            f"  Available symbols are under: {settings.data_lake_root}"
        )

    print(f"[data] Loading {settings.symbol} from data lake ...")
    df = pd.read_parquet(data_path)

    # Filter by date range
    df = df[(df["trade_date"] >= settings.start_date) & (df["trade_date"] <= settings.end_date)]

    if df.empty:
        sys.exit(
            f"No data for {settings.symbol} in {settings.start_date}-{settings.end_date}."
        )

    # Join industry before sorting
    if settings.join_industry:
        df = _join_industry(df, settings)  # type: ignore[reportArgumentType]

    # Sort and reset
    df = df.sort_values(by="trade_date").reset_index(drop=True)  # type: ignore[reportCallIssue]

    # Cast trade_date to datetime for downstream use
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    print(
        f"  {len(df)} rows ({settings.symbol}, "
        f"{settings.start_date[:4]}-{settings.end_date[:4]})"
    )
    return df


def load_multi_symbols(settings: Settings) -> list[tuple[str, pd.DataFrame]]:
    """Load data for multiple symbols. Skips missing files quietly.

    Returns list of (symbol, DataFrame) tuples.
    """
    results: list[tuple[str, pd.DataFrame]] = []
    for i, sym in enumerate(settings.symbols):
        data_path = settings.data_lake_root / f"{sym}.parquet"
        if not data_path.exists():
            print(f"[data] SKIP {sym}: file not found")
            continue
        df = pd.read_parquet(data_path)
        df = df[(df["trade_date"] >= settings.start_date) & (df["trade_date"] <= settings.end_date)]
        if len(df) < settings.min_listed_days:
            print(f"[data] SKIP {sym}: only {len(df)} rows (< {settings.min_listed_days})")
            continue

        if settings.join_industry:
            df = _join_industry(df, settings)  # type: ignore[reportArgumentType]

        df = df.sort_values(by="trade_date").reset_index(drop=True)  # type: ignore[reportCallIssue]
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        results.append((sym, df))

        if (i + 1) % 500 == 0:
            print(f"[data]  loaded {i + 1}/{len(settings.symbols)} ...")

    print(f"[data] Loaded {len(results)} / {len(settings.symbols)} symbols")
    return results
