"""Data loading from market-data-platform A-share data lake.

Reads per-symbol parquet files from the data lake directory.
Each file: one stock, all trading days, 45 columns.

Optionally joins 申万行业分类 from the instruments table.
Optionally adds price-limit tradability flags and liquidity filters.
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


def _add_price_limit_flags(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Add is_tradable column based on A-share daily price limits.

    A-share rules:
      - Main board (60xxxx, 00xxxx): ±10%
      - ChiNext (30xxxx): ±20%
      - STAR (68xxxx): ±20%
      - BJ (8xxxxx, 4xxxxx): ±30%

    A stock is NOT tradable if it hits the daily limit (up or down),
    because you cannot buy at the up limit or sell at the down limit.

    Flag semantics:
      - is_tradable = True: neither up-limit nor down-limit hit
      - is_tradable_buy = True: not at up-limit (can buy)
      - is_tradable_sell = True: not at down-limit (can sell)
    """
    if "pre_close" not in df.columns or "close" not in df.columns:
        return df

    pre_close = df["pre_close"]
    close = df["close"]

    # Determine limit percentage by board
    symbol = settings.symbol
    if symbol.startswith(("30",)) or symbol.startswith(("68",)):
        limit_pct = 0.20
    elif symbol.startswith(("8", "4")):
        limit_pct = 0.30
    else:
        limit_pct = 0.10

    up_limit = pre_close * (1.0 + limit_pct)
    down_limit = pre_close * (1.0 - limit_pct)

    # Use a small tolerance for floating-point
    tol = 0.001
    at_up_limit = close >= up_limit - tol
    at_down_limit = close <= down_limit + tol

    df["is_tradable"] = ~(at_up_limit | at_down_limit)
    df["is_tradable_buy"] = ~at_up_limit
    df["is_tradable_sell"] = ~at_down_limit

    n_blocked = (~df["is_tradable"]).sum()
    if n_blocked > 0:
        pct = n_blocked / len(df) * 100
        print(f"[data] Price-limit blocked: {n_blocked} days ({pct:.1f}%)")

    return df


def _apply_liquidity_filter(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Mark rows with insufficient daily turnover as non-tradable.

    Uses 'amount' column (daily turnover in CNY) if available.
    Does NOT drop rows — only sets is_tradable* flags to False,
    so backtest can skip those signals while keeping pricing data.
    """
    if settings.min_daily_amount <= 0:
        return df

    if "amount" not in df.columns:
        print("[data] WARNING: 'amount' column missing, liquidity filter skipped")
        return df

    low_liq = df["amount"] < settings.min_daily_amount
    n_low = low_liq.sum()
    if n_low > 0:
        pct = n_low / len(df) * 100
        print(f"[data] Low liquidity filtered: {n_low} days ({pct:.1f}%) "
              f"(< {settings.min_daily_amount:,.0f} CNY)")

        df.loc[low_liq, "is_tradable"] = False
        df.loc[low_liq, "is_tradable_buy"] = False
        df.loc[low_liq, "is_tradable_sell"] = False

    return df


def load_data(settings: Settings) -> pd.DataFrame:
    """Load daily bars for a single A-share symbol from the data lake.

    Columns expected (subset): trade_date, open, high, low, close, pre_close, vol, amount.

    If settings.join_industry is True, also joins 申万行业分类 from the
    instruments table, adding an 'industry' column.

    If settings.backtest_enforce_price_limit is True, adds is_tradable
    columns based on A-share daily price limits.

    If settings.min_daily_amount > 0, marks low-liquidity days as non-tradable.

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

    # Add price-limit flags (before feature engineering — uses raw close/pre_close)
    if settings.backtest_enforce_price_limit:
        df = _add_price_limit_flags(df, settings)

    # Apply liquidity filter (after price-limit flags)
    df = _apply_liquidity_filter(df, settings)

    print(
        f"  {len(df)} rows ({settings.symbol}, "
        f"{settings.start_date[:4]}-{settings.end_date[:4]})"
    )
    return df
