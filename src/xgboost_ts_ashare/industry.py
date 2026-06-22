"""Industry neutralization for cross-sectional feature purification.

Per-date, cross-sectionally regress each feature on industry dummy variables
and replace the feature with the residual. This removes industry-level
confounding so the model learns stock-specific signal.

Works in two modes:
- Multi-stock: pool all symbols per date, dummify industry, regress.
- Single-stock: no-op (can't neutralize without cross-section).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def neutralize_industry(
    df: pd.DataFrame,
    feature_cols: list[str],
    industry_col: str = "industry",
    date_col: str = "trade_date",
    min_industries_per_date: int = 3,
    min_stocks_per_date: int = 10,
) -> pd.DataFrame:
    """Cross-sectional industry neutralization.

    For each trading date with enough cross-sectional breadth, fits
    feature ~ industry_dummies and replaces each feature with its residual.

    Parameters
    ----------
    df : DataFrame
        Must contain feature_cols, industry_col, date_col.
        Multiple symbols pooled together (one row per symbol per date).
    feature_cols : list[str]
        Feature columns to neutralize.
    industry_col : str
        Column with 申万 industry labels.
    date_col : str
        Column with trade dates (datetime or sortable).
    min_industries_per_date : int
        Minimum distinct industries on a date to perform neutralization.
        Dates with fewer industries are left unchanged.
    min_stocks_per_date : int
        Minimum number of stocks on a date to perform neutralization.
        Below this threshold, regression is unreliable.

    Returns
    -------
    DataFrame with industry-neutralized feature values. Rows where
    neutralization was skipped are unchanged.
    """
    if industry_col not in df.columns:
        print("[indi] Industry column not found — skipping neutralization.")
        return df

    df = df.copy()

    # Count unique symbols per date to gauge cross-sectional breadth
    by_date = df.groupby(date_col, observed=True)

    n_symbols_per_date = by_date.size()
    n_industries_per_date = by_date[industry_col].nunique()

    mask = (n_symbols_per_date >= min_stocks_per_date) & (
        n_industries_per_date >= min_industries_per_date
    )
    eligible_dates = set(n_symbols_per_date[mask].index)  # type: ignore[reportAttributeAccessIssue]

    n_neutralized = 0
    n_skipped = 0
    n_dates_processed = 0

    for date_val, group in by_date:
        if date_val not in eligible_dates:
            n_skipped += len(group)
            continue

        # One-hot encode industry
        try:
            dummies = pd.get_dummies(group[industry_col], drop_first=True, dtype=float)
        except Exception:
            n_skipped += len(group)
            continue

        if dummies.shape[1] < 1 or dummies.shape[0] < min_stocks_per_date:
            n_skipped += len(group)
            continue

        n_dates_processed += 1

        # Regress each feature on industry dummies, replace with residual
        for feat in feature_cols:
            y = group[feat].values.astype(float)
            if np.isnan(y).any():
                continue

            model = LinearRegression(fit_intercept=True)
            try:
                model.fit(dummies.values, y)
                y_pred = model.predict(dummies.values)
                residual = y - y_pred
                # Preserve NaN positions
                mask = ~np.isnan(group[feat].values)
                df.loc[group.index[mask], feat] = residual[mask]
            except Exception:
                continue

        n_neutralized += len(group)

    total = n_neutralized + n_skipped
    if total > 0:
        pct = n_neutralized / total * 100
        print(
            f"[indi] Neutralized {n_neutralized}/{total} rows "
            f"({pct:.0f}%) across {n_dates_processed} dates"
        )
    else:
        print("[indi] No rows eligible for neutralization — cross-section too thin.")

    return df


def compute_industry_ic_reduction(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    feature_cols: list[str],
    return_col: str = "future_return",
    industry_col: str = "industry",
    date_col: str = "trade_date",
) -> dict[str, dict[str, float]]:
    """Compare per-feature Rank IC before and after neutralization.

    Uses Spearman rank correlation of each feature with future returns,
    computed on a pooled cross-section. Reports IC reduction.

    Returns dict: {feature: {"ic_before": float, "ic_after": float, "reduction": float}}
    """
    from scipy.stats import spearmanr

    result: dict[str, dict[str, float]] = {}
    common_idx = df_before.index.intersection(df_after.index)

    for feat in feature_cols:
        if feat not in df_before.columns or feat not in df_after.columns:
            continue

        valid = df_before.loc[common_idx][[feat, return_col]].dropna()
        valid_a = df_after.loc[valid.index][[feat, return_col]].dropna()

        if len(valid) < 30:
            continue

        ic_before = float(spearmanr(valid[feat], valid[return_col]).correlation)  # type: ignore[arg-type]
        ic_after = float(spearmanr(valid_a[feat], valid_a[return_col]).correlation)  # type: ignore[arg-type]

        result[feat] = {
            "ic_before": ic_before,
            "ic_after": ic_after,
            "reduction": abs(ic_before) - abs(ic_after),
        }

    return result
