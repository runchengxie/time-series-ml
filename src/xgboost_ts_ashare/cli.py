"""Command-line entry point for the XGBoost A-share prediction pipeline.

Supports:
- Single or multi-symbol training
- Cross-sectional industry neutralization
- Purged time-series CV with embargo
- Threshold optimisation on validation set
- Multi-model comparison (XGBoost, LR, RF, LightGBM)
- Probability calibration (CalibratedClassifierCV)
- Signal filtering (prob_threshold)
- Walk-forward backtest with TCA
- Factor IC analysis
- YAML configuration files
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .backtest import walk_forward
from .config import Settings
from .config_yaml import load_yaml_config
from .data import load_data
from .features import FEATURE_COLUMNS, build_features
from .industry import neutralize_industry
from .labels import build_labels
from .metrics import (
    compute_factor_correlation,
    compute_factor_ic,
    evaluate,
    print_report,
)
from .model import compare_models, train_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost A-share movement prediction")

    # Symbol selection
    parser.add_argument(
        "--symbol", default="000001.SZ",
        help="Single symbol (default: 000001.SZ 平安银行)",
    )
    parser.add_argument(
        "--symbols", type=str, default="",
        help="Comma-separated symbols for multi-symbol mode (e.g. 000001.SZ,600000.SH)",
    )

    # Date range
    parser.add_argument("--start-date", default="20150101", help="Start date YYYYMMDD")
    parser.add_argument("--end-date", default="", help="End date YYYYMMDD (default: today)")

    # Data
    parser.add_argument(
        "--data-lake-root", type=str,
        default="/home/richard/data/market-data-platform/assets/tushare/"
                "a_share/daily/a_share_all_20150101_20260622_shadow_daily_clean/data",
    )
    parser.add_argument("--min-listed-days", type=int, default=252)

    # Industry neutralization
    parser.add_argument(
        "--neutralize-industry", action="store_true",
        help="Cross-sectional industry neutralization (requires --symbols)",
    )

    # Model
    parser.add_argument("--threshold", type=float, default=0.002, help="Up threshold for label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)

    # CV purge/embargo
    parser.add_argument("--purge-days", type=int, default=20)
    parser.add_argument("--embargo-days", type=int, default=1)

    # Calibration
    parser.add_argument("--calibrate", action="store_true", help="Fit CalibratedClassifierCV")
    parser.add_argument("--cv-method", default="isotonic", choices=["isotonic", "sigmoid"])

    # Signal filtering
    parser.add_argument(
        "--prob-threshold", type=float, default=0.50,
        help="Minimum predicted probability to trade (0.55+ recommended after calibration)",
    )

    # Features
    parser.add_argument(
        "--optimize-threshold", action="store_true",
        help="Grid-search optimal classification threshold",
    )
    parser.add_argument(
        "--compare-models", action="store_true",
        help="Compare XGBoost/LR/RF/LightGBM",
    )

    # Backtest
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run walk-forward backtest with TCA",
    )
    parser.add_argument("--backtest-cost-bps", type=float, default=5.0)

    # Config
    parser.add_argument("--config", type=str, default="", help="Path to YAML config file")

    return parser.parse_args(argv)


def _optimize_threshold(
    model: Any, X_val: pd.DataFrame, y_val: pd.Series, metric: str = "f1"
) -> tuple[float, float]:
    """Grid-search classification threshold on validation set."""
    prob = model.predict_proba(X_val)[:, 1]
    from sklearn.metrics import f1_score, precision_score

    scorer = {"f1": f1_score, "precision": precision_score}[metric]

    best_thresh, best_score = 0.5, 0.0
    for t in np.arange(0.30, 0.71, 0.02):
        pred = (prob >= t).astype(int)
        s = scorer(y_val, pred)
        if s > best_score:
            best_score = s
            best_thresh = t

    return float(best_thresh), float(best_score)


def _print_model_comparison(results: list[dict[str, Any]]) -> None:
    print("\n-- Model Comparison (CV accuracy) --")
    print(f"  {'Model':<20} {'CV Mean':>8}  {'CV Std':>8}")
    for r in results:
        print(f"  {r['model_name']:<20} {r['cv_mean']:>8.3f}  {r['cv_std']:>8.3f}")


def _print_backtest(bt: dict[str, Any]) -> None:
    if "error" in bt:
        print(f"\n[backtest] {bt['error']}")
        return
    print("\n-- Walk-Forward Backtest (TCA: {:.1f} bps round-trip) --".format(
        bt.get("cost_bps_used", 5.0)
    ))
    print(f"  Trades:          {bt['n_trades']}")
    print(f"  Total Return:    {bt['total_return']:.4f}")
    print(f"  Annual Return:   {bt['annual_return']:.4f}")
    print(f"  Annual Vol:      {bt['annual_vol']:.4f}")
    print(f"  Sharpe:          {bt['sharpe']:.3f}")
    print(f"  Max Drawdown:    {bt['max_drawdown']:.4f}")
    print(f"  Win Rate:        {bt['win_rate']:.3f}")
    print(f"  Profit Factor:   {bt['profit_factor']:.3f}")
    print(f"  Turnover/yr:     {bt['turnover']:.1f}")
    sr = bt.get("signal_rate", 0)
    print(f"  Signal Rate:     {sr:.1%}")
    print(f"  Prob Threshold:  {bt.get('prob_threshold_used', 0.50):.2f}")


# ---------------------------------------------------------------------------
# Data preparation (shared between single and multi-symbol paths)
# ---------------------------------------------------------------------------


def _prepare_df(settings: Settings) -> pd.DataFrame:
    """Load data, build features, build labels, keep relevant columns."""
    df = load_data(settings)
    if df is None or len(df) < settings.min_listed_days:
        return cast(pd.DataFrame, df)

    df = build_features(df)
    print("[features] Features built.")
    df = build_labels(df, threshold=settings.up_threshold)

    # Select columns and drop NaN
    keep_cols = [*FEATURE_COLUMNS, "target", "future_return"]
    for c in ("trade_date", "close", "industry", "ts_code"):
        if c in df.columns:
            keep_cols.append(c)
    collected = df[[c for c in keep_cols if c in df.columns]].dropna().reset_index(drop=True)
    result: pd.DataFrame = cast(pd.DataFrame, collected)
    return result


# ---------------------------------------------------------------------------
# Training + evaluation for one symbol (single DataFrame)
# ---------------------------------------------------------------------------


def _train_and_evaluate(
    df: pd.DataFrame,
    settings: Settings,
    args: argparse.Namespace,
    purge_days: int,
    embargo_days: int,
    label: str = "",
) -> dict[str, Any] | None:
    """Train model and evaluate on a single symbol's DataFrame.

    Expects df to already have features, labels, and all keep_cols.
    """
    if len(df) < 50:
        return None

    # Train-test split (chronological)
    split_idx = int(len(df) * (1 - settings.test_size))
    X_train = df.iloc[:split_idx][FEATURE_COLUMNS]
    X_test = df.iloc[split_idx:][FEATURE_COLUMNS]
    y_train = df.iloc[:split_idx]["target"]
    y_test = df.iloc[split_idx:]["target"]
    actual_returns_test = df.iloc[split_idx:]["future_return"].values

    print(f"[split] Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # Train
    if args.compare_models:
        print("[train] Comparing models (purged TS-CV) ...")
        results = compare_models(
            X_train, y_train,
            params=settings.xgb_params,
            cv_splits=settings.cv_splits,
            purge_days=purge_days,
            embargo_days=embargo_days,
        )
        _print_model_comparison(results)
        best = results[0]
        model = best["model"]
        cv_stats = {
            "mean": best["cv_mean"], "std": best["cv_std"],
            "scores": [], "n_folds": settings.cv_splits,
        }
        print(f"\n  Using best model: {best['model_name']}")
    else:
        print(
            f"[train] Training XGBoost "
            f"(purged TS-CV, purge={purge_days}d, embargo={embargo_days}d"
            f"{', calibrate' if settings.calibrate else ''}) ..."
        )
        model, cv_stats = train_model(
            X_train, y_train,
            params=settings.xgb_params,
            cv_splits=settings.cv_splits,
            purge_days=purge_days,
            embargo_days=embargo_days,
            calibrate=settings.calibrate,
            cv_method=settings.cv_method,
        )
        print(f"  CV folds     : {cv_stats['n_folds']}")
        print(f"  CV Accuracy  : {cv_stats['mean']:.3f} +/- {cv_stats['std'] * 2:.3f}")

    # Threshold optimisation
    pred_threshold = 0.5
    if args.optimize_threshold:
        print("[tune] Optimising classification threshold ...")
        val_split = int(len(X_train) * 0.8)
        X_train2, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
        y_train2, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

        model.fit(X_train2, y_train2)
        pred_threshold, best_f1 = _optimize_threshold(model, X_val, y_val)
        print(f"  Optimal threshold: {pred_threshold:.2f} (val F1={best_f1:.3f})")
        model.fit(X_train, y_train)

    signal_threshold = max(pred_threshold, settings.prob_threshold)
    if settings.prob_threshold > 0.50:
        print(f"[filter] Signal threshold: prob >= {signal_threshold:.2f}")

    # Evaluate
    print("[eval] Evaluating ...")
    prev_direction_test = df.iloc[split_idx:]["target"].shift(1).fillna(0).astype(int)

    report = evaluate(
        model, X_train, X_test, y_train, y_test,
        feature_names=FEATURE_COLUMNS,
        prev_direction_test=prev_direction_test,
        actual_returns_test=actual_returns_test,
    )

    # Factor IC analysis
    df_eval: pd.DataFrame = cast(pd.DataFrame, df)
    _, corr_warnings = compute_factor_correlation(df_eval, FEATURE_COLUMNS)
    factor_ic = compute_factor_ic(df_eval, FEATURE_COLUMNS)
    report["factor_ic"] = factor_ic
    report["factor_correlation_warnings"] = corr_warnings

    print_report(report)

    # Walk-forward backtest
    if args.backtest:
        print("\n[backtest] Running walk-forward backtest ...")
        from xgboost import XGBClassifier

        bt = walk_forward(
            df_eval,
            feature_cols=FEATURE_COLUMNS,
            model_class=XGBClassifier,
            model_params=settings.xgb_params,
            threshold=signal_threshold,
            retrain_freq="ME",
            cost_bps=args.backtest_cost_bps,
            purge_days=purge_days,
        )
        _print_backtest(bt)
        report["backtest"] = bt

    return report


# ---------------------------------------------------------------------------
# Single-symbol experiment (full pipeline)
# ---------------------------------------------------------------------------


def _run_single_experiment(
    settings: Settings,
    args: argparse.Namespace,
    purge_days: int,
    embargo_days: int,
) -> dict[str, Any] | None:
    """Full pipeline for a single symbol: load → prepare → train → eval."""
    df = _prepare_df(settings)
    if df is None or len(df) < 50:
        return None
    return _train_and_evaluate(df, settings, args, purge_days, embargo_days)


# ---------------------------------------------------------------------------
# Multi-symbol with optional industry neutralization
# ---------------------------------------------------------------------------


def _run_neutralized_multi(
    symbols: list[str],
    settings: Settings,
    args: argparse.Namespace,
    purge_days: int,
    embargo_days: int,
) -> None:
    """Load all symbols, pool, neutralize, split, train each."""
    # 1. Load and prepare each symbol
    print(f"\n[data] Loading {len(symbols)} symbols ...")
    all_dfs: list[tuple[str, pd.DataFrame]] = []
    for i, sym in enumerate(symbols):
        sym_settings = Settings(
            symbol=sym, symbols=[sym],
            start_date=settings.start_date, end_date=settings.end_date,
            data_lake_root=settings.data_lake_root,
            instruments_path=settings.instruments_path,
            join_industry=True,  # required for neutralization
            min_listed_days=settings.min_listed_days,
            up_threshold=settings.up_threshold,
        )
        df = _prepare_df(sym_settings)
        if df is not None and len(df) >= 50:
            df["_symbol"] = sym  # tag for splitting later
            all_dfs.append((sym, df))

        if (i + 1) % 100 == 0:
            print(f"  prepared {i + 1}/{len(symbols)} ...")

    if not all_dfs:
        print("[data] No valid symbols found.")
        return

    print(f"[data] Prepared {len(all_dfs)} / {len(symbols)} symbols")

    # 2. Pool together
    pooled = pd.concat([d for _, d in all_dfs], ignore_index=True)
    pooled = pooled.sort_values(["trade_date", "_symbol"]).reset_index(drop=True)
    print(f"[pool] Pooled DataFrame: {len(pooled)} rows, "
          f"{pooled['_symbol'].nunique()} symbols")

    # 3. Industry neutralization
    n_before = pooled[FEATURE_COLUMNS].isna().sum().sum()
    pooled = neutralize_industry(pooled, FEATURE_COLUMNS, industry_col="industry")
    n_after = pooled[FEATURE_COLUMNS].isna().sum().sum()
    if n_after > n_before:
        print(f"[indi] WARNING: neutralization introduced {n_after - n_before} NaN values")

    # 4. Split back and train each symbol
    report_count = 0
    for sym, _ in all_dfs:
        sym_df = pooled[pooled["_symbol"] == sym].drop(columns=["_symbol"])
        sym_df = sym_df.sort_values(by="trade_date").reset_index(drop=True)  # type: ignore[reportCallIssue]

        if len(sym_df) < 50:
            continue

        print(f"\n{'=' * 60}")
        print(f"  [{report_count + 1}] {sym}")
        print("=" * 60)

        sym_settings = Settings(
            symbol=sym, symbols=[sym],
            start_date=settings.start_date, end_date=settings.end_date,
            data_lake_root=settings.data_lake_root,
            join_industry=True, neutralize_industry=True,
            min_listed_days=settings.min_listed_days,
            up_threshold=settings.up_threshold,
            test_size=settings.test_size,
            cv_splits=settings.cv_splits,
            calibrate=settings.calibrate,
            cv_method=settings.cv_method,
            prob_threshold=settings.prob_threshold,
        )

        result = _train_and_evaluate(
            sym_df, sym_settings, args, purge_days, embargo_days, label=sym,
        )
        if result is not None:
            report_count += 1

    print(f"\n[multi] Completed: {report_count} symbols evaluated")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # YAML config (args take precedence)
    yaml_overrides: dict[str, Any] = {}
    if args.config:
        yaml_overrides = load_yaml_config(args.config)
        print(f"[config] Loaded YAML: {args.config}")

    # Build symbols list
    symbols = [args.symbol]
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # Settings
    settings = Settings(
        symbol=symbols[0],
        symbols=symbols,
        start_date=args.start_date or yaml_overrides.get("start_date", "20150101"),
        end_date=args.end_date or yaml_overrides.get("end_date", ""),
        data_lake_root=Path(args.data_lake_root),
        min_listed_days=args.min_listed_days,
        up_threshold=args.threshold,
        test_size=args.test_size,
        cv_splits=args.cv_splits,
        calibrate=args.calibrate,
        cv_method=args.cv_method,
        prob_threshold=args.prob_threshold,
        neutralize_industry=args.neutralize_industry,
    )

    purge_days = args.purge_days
    embargo_days = args.embargo_days

    print(f"Experiment: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
    print(f"  Data lake  : {settings.data_lake_root}")
    print(f"  Date range : {settings.start_date} - {settings.end_date}")
    print(f"  Threshold  : {settings.up_threshold:.3f}")
    print(f"  Test size  : {settings.test_size:.0%}")
    print(f"  CV purge   : {purge_days} days")
    print(f"  CV embargo : {embargo_days} day(s)")
    if settings.calibrate:
        print(f"  Calibrate  : {settings.cv_method}")
    if settings.prob_threshold > 0.50:
        print(f"  Prob thresh: {settings.prob_threshold:.2f}")
    if settings.neutralize_industry:
        print("  Neutralize : industry")

    # Route to appropriate path
    if len(symbols) > 1 and settings.neutralize_industry:
        # Multi-symbol + industry neutralization
        _run_neutralized_multi(symbols, settings, args, purge_days, embargo_days)
    elif len(symbols) > 1:
        # Multi-symbol, no neutralization (simple loop)
        print(f"\n[data] Loading {len(symbols)} symbols ...")
        for i, sym in enumerate(symbols):
            print(f"\n{'=' * 60}")
            print(f"  [{i + 1}/{len(symbols)}] {sym}")
            print("=" * 60)
            sym_settings = Settings(
                symbol=sym, symbols=[sym],
                start_date=settings.start_date, end_date=settings.end_date,
                data_lake_root=settings.data_lake_root,
                join_industry=settings.join_industry,
                min_listed_days=settings.min_listed_days,
                up_threshold=settings.up_threshold,
                test_size=settings.test_size,
                cv_splits=settings.cv_splits,
                calibrate=settings.calibrate,
                cv_method=settings.cv_method,
                prob_threshold=settings.prob_threshold,
            )
            _run_single_experiment(sym_settings, args, purge_days, embargo_days)
    else:
        # Single symbol
        _run_single_experiment(settings, args, purge_days, embargo_days)


if __name__ == "__main__":
    main()
