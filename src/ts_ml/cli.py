"""Command-line entry point for the XGBoost A-share prediction pipeline.

Supports:
- Single or multi-symbol training
- Cross-sectional industry neutralization
- Purged time-series CV with embargo
- Threshold optimisation on validation set
- Multi-model comparison (XGBoost, LR, RF, LightGBM, Ridge)
- Probability calibration (CalibratedClassifierCV)
- Signal filtering (prob_threshold)
- Walk-forward backtest with split buy/sell costs and tradability filters
- Factor IC analysis
- YAML configuration files
- Sample weighting (exp_decay)
- Rolling training windows
- Regression mode (predict future_return)
- Final OOS holdout
- Permutation test
- Feature ablation
- Result persistence (summary.json + config.used.yml)
- Multi-symbol summary CSV
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

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
    permutation_test,
    print_report,
    run_ablation,
)
from .model import compare_models, train_model
from .regime import classify_regime, compute_market_proxy
from .tracking import end_tracking, log_backtest, log_metrics, start_tracking

# Optional import: meta-labeling
try:
    from .meta_labeling import apply_meta_filter, train_meta_model
    _HAS_META = True
except ImportError:
    _HAS_META = False

# Feature families for ablation
FEATURE_FAMILIES: dict[str, list[str]] = {
    "momentum": ["SMA5_diff", "SMA10_diff", "SMA20_diff", "SMA60_diff"],
    "price_dist": ["price_dist_5d", "price_dist_20d", "price_dist_60d"],
    "relative_strength": ["RSI_14", "MACD_hist", "BB_position"],
    "volume": ["Volume_SMA5_ratio", "Volume_SMA20_ratio", "Volume_trend", "vol"],
    "volatility": ["ATR_14_pct", "HistVol_20", "HistVol_60"],
    "candle": ["body_ratio", "upper_shadow_ratio", "lower_shadow_ratio"],
    "gap": ["gap_pct", "open_gap"],
    "ret_dist": ["ret_skew_20d", "ret_kurt_20d"],
    "turnover": ["turnover_rate_f", "volume_ratio_raw"],
    "valuation": ["log_pe", "log_pb", "log_mcap"],
}


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
    parser.add_argument(
        "--min-daily-amount", type=float, default=0.0,
        help="Minimum daily turnover in CNY (e.g. 1000000 = 100万); 0 = disabled",
    )

    # Industry neutralization
    parser.add_argument(
        "--neutralize-industry", action="store_true",
        help="Cross-sectional industry neutralization (requires --symbols)",
    )

    # Model
    parser.add_argument("--threshold", type=float, default=0.002, help="Up threshold for label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--final-oos-size", type=float, default=0.0,
                       help="Final OOS holdout fraction (0=disabled, 0.10=last 10%%)")

    # Triple Barrier Labeling
    parser.add_argument(
        "--triple-barrier", action="store_true",
        help="Use triple barrier labels (-1/0/+1) instead of binary",
    )
    parser.add_argument(
        "--holding-period", type=int, default=5,
        help="Holding period in days for triple barrier (default 5)",
    )
    parser.add_argument(
        "--profit-take", type=float, default=0.05,
        help="Profit barrier as fraction (e.g. 0.05 = 5%%)",
    )
    parser.add_argument(
        "--stop-loss", type=float, default=0.03,
        help="Stop barrier as fraction (e.g. 0.03 = 3%%)",
    )
    parser.add_argument("--cv-splits", type=int, default=5)

    # CV purge/embargo
    parser.add_argument("--purge-days", type=int, default=20)
    parser.add_argument("--embargo-days", type=int, default=1)

    # Sample weighting
    parser.add_argument(
        "--sample-weight-halflife", type=int, default=0,
        help="Halflife in trading days for exp_decay weights (0=uniform, 252=1 year)",
    )

    # Training window
    parser.add_argument(
        "--train-window-days", type=int, default=0,
        help="Rolling training window in trading days (0=all, 504=~2 years)",
    )

    # Regression mode
    parser.add_argument(
        "--regression", action="store_true",
        help="Predict future_return directly (regression) instead of binary direction",
    )

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

    # Ablation
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run feature family ablation (minus-one-family experiments)",
    )

    # Meta-labeling
    parser.add_argument(
        "--meta-labeling", action="store_true",
        help="Train a secondary model to filter primary model predictions",
    )

    # Lag features
    parser.add_argument(
        "--use-lag-features", action="store_true",
        help="Add t-3 and t-5 lag features for sequence dependency testing",
    )

    # Permutation test
    parser.add_argument(
        "--permutation-test", action="store_true",
        help="Run label permutation test (n=100 shuffles)",
    )

    # Backtest
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run walk-forward backtest with TCA",
    )
    parser.add_argument("--backtest-cost-bps", type=float, default=5.0,
                       help="Legacy round-trip cost (superseded by buy/sell split)")
    parser.add_argument("--backtest-buy-cost-bps", type=float, default=0.0,
                       help="Buy-side cost in bps (A-share: 0 for no stamp duty on buy)")
    parser.add_argument("--backtest-sell-cost-bps", type=float, default=0.0,
                       help="Sell-side cost in bps; if 0, uses backtest-cost-bps/2")
    parser.add_argument(
        "--no-price-limit-filter", action="store_true",
        help="Disable A-share price-limit tradability filter",
    )

    # Regime
    parser.add_argument(
        "--regime", action="store_true",
        help="Compute market regime (bull/range/bear) and report per-regime backtest stats",
    )

    # Result persistence
    parser.add_argument(
        "--save-results", action="store_true",
        help="Save summary.json and config.used.yml to artifacts/runs/<timestamp>/",
    )
    parser.add_argument(
        "--artifacts-root", type=str, default="artifacts",
        help="Root directory for saved artifacts (default: artifacts/)",
    )

    # Config
    parser.add_argument("--config", type=str, default="", help="Path to YAML config file")

    return parser.parse_args(argv)


def parse_args_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """Alias for parse_args — used externally."""
    return parse_args(argv)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


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


def _print_model_comparison(results: list[dict[str, Any]], regression: bool = False) -> None:
    """Print model comparison results with sanity check."""
    from .metrics import _print_model_comparison as _pmc
    _pmc(results, regression=regression)


def _print_backtest(bt: dict[str, Any]) -> None:
    if "error" in bt:
        print(f"\n[backtest] {bt['error']}")
        return
    cost_used = bt.get("cost_bps_used", bt.get("buy_cost_bps", 0) + bt.get("sell_cost_bps", 0))
    buy_bps = bt.get("buy_cost_bps", 0)
    sell_bps = bt.get("sell_cost_bps", 0)
    cost_header = (
        f"\n-- Walk-Forward Backtest "
        f"(buy={buy_bps:.1f} sell={sell_bps:.1f} bps, "
        f"round-trip={cost_used:.1f} bps) --"
    )
    print(cost_header)
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
    skipped = bt.get("n_skipped_untradable", 0)
    if skipped > 0:
        print(f"  Skipped (limit): {skipped}")

    # Regime statistics
    regime_stats = bt.get("regime_stats")
    if regime_stats:
        print("\n-- Per-Regime Performance --")
        regime_names = {1: "Bull", 0: "Range", -1: "Bear"}
        header = f"  {'Regime':<8} {'Trades':>7} {'Win Rate':>9} {'Ret':>8} {'Sharpe':>7}"
        print(header)
        for regime in [1, 0, -1]:
            s = regime_stats.get(regime)
            if s:
                print(
                    f"  {regime_names.get(regime, str(regime)):<8}"
                    f" {s['n_trades']:>7}"
                    f" {s['win_rate']:>9.3f}"
                    f" {s['total_return']:>8.4f}"
                    f" {s['sharpe']:>7.2f}"
                )


# ---------------------------------------------------------------------------
# Data preparation (shared between single and multi-symbol paths)
# ---------------------------------------------------------------------------


def _prepare_df(settings: Settings, *, use_lag: bool = False) -> pd.DataFrame:
    """Load data, build features, build labels, keep relevant columns."""
    df = load_data(settings)
    if df is None or len(df) < settings.min_listed_days:
        return cast(pd.DataFrame, df)

    df = build_features(df, use_lag=use_lag)
    print("[features] Features built.")
    df = build_labels(
        df, threshold=settings.up_threshold,
        holding_period=settings.holding_period,
        profit_take=settings.profit_take,
        stop_loss=settings.stop_loss,
        triple_barrier=settings.triple_barrier,
    )

    # Select columns and drop NaN
    keep_cols = [*FEATURE_COLUMNS, "target", "future_return"]
    for c in ("trade_date", "close", "industry", "ts_code",
              "is_tradable", "is_tradable_buy", "is_tradable_sell", "amount",
              "turnover_rate", "volume_ratio",
              "pe_ttm", "pb", "total_mv", "pre_close", "open", "high", "low"):
        if c in df.columns:
            keep_cols.append(c)
    collected = df[[c for c in keep_cols if c in df.columns]].dropna().reset_index(drop=True)

    # Remap triple barrier labels from [-1,0,1] to [0,1,2] for XGBoost multiclass
    if settings.triple_barrier and "target" in collected.columns:
        collected["target"] = collected["target"] + 1

    result: pd.DataFrame = cast(pd.DataFrame, collected)
    return result


# ---------------------------------------------------------------------------
# Training + evaluation for one symbol (single DataFrame)
# ---------------------------------------------------------------------------


def _compute_pipeline_regime(
    symbols: list[str], settings: Settings
) -> pd.Series | None:
    """Compute market regime from equal-weighted pool of all symbols."""
    from .data import load_data as _load

    dfs: list[pd.DataFrame] = []
    for sym in symbols:
        sym_settings = Settings(
            symbol=sym, symbols=[sym],
            start_date=settings.start_date, end_date=settings.end_date,
            data_lake_root=settings.data_lake_root,
            min_listed_days=settings.min_listed_days,
            up_threshold=settings.up_threshold,
            backtest_enforce_price_limit=False,
            min_daily_amount=0.0,
        )
        df = _load(sym_settings)
        if df is not None and len(df) >= 50:
            dfs.append(df)

    if not dfs:
        print("[regime] Not enough data to compute regime")
        return None

    proxy = compute_market_proxy(dfs)
    regime = classify_regime(proxy)
    summary = regime.value_counts().to_dict()
    print(f"[regime] Market proxy from {len(dfs)} symbols: "
          f"bull={summary.get(1, 0)}d, range={summary.get(0, 0)}d, "
          f"bear={summary.get(-1, 0)}d")
    return regime


def _compute_single_regime(settings: Settings) -> pd.Series | None:
    """Compute regime from a single stock's own close (fallback)."""
    from .data import load_data as _load

    sym_settings = Settings(
        symbol=settings.symbol,
        start_date=settings.start_date, end_date=settings.end_date,
        data_lake_root=settings.data_lake_root,
        backtest_enforce_price_limit=False,
        min_daily_amount=0.0,
    )
    df = _load(sym_settings)
    if df is None or len(df) < 60:
        return None

    close_series: pd.Series = df.set_index("trade_date")["close"]  # type: ignore[assignment]
    regime = classify_regime(close_series)
    summary = regime.value_counts().to_dict()
    print(f"[regime] Single-stock proxy from {settings.symbol}: "
          f"bull={summary.get(1, 0)}d, range={summary.get(0, 0)}d, "
          f"bear={summary.get(-1, 0)}d")
    return regime


def _log_eval_metrics(report: dict[str, Any], cv_stats: dict[str, Any]) -> None:
    """Extract scalar metrics from evaluation report and log to MLflow."""
    metrics: dict[str, float] = {}

    # CV stats
    metrics["cv_mean"] = float(cv_stats["mean"])
    metrics["cv_std"] = float(cv_stats["std"])

    mode = report.get("mode", "classification")
    if mode == "regression":
        for k in ("train_rmse", "test_rmse", "train_r2", "test_r2", "overfitting_gap_rmse"):
            if k in report and isinstance(report[k], (int, float)):
                metrics[k] = float(report[k])
    else:
        for k in ("train_accuracy", "test_accuracy", "roc_auc", "overfitting_gap",
                  "precision", "recall", "f1"):
            if k in report and isinstance(report[k], (int, float)):
                metrics[k] = float(report[k])

    # IC
    ic = report.get("ic", {})
    if isinstance(ic, dict):
        for k in ("rank_ic", "icir"):
            v = ic.get(k)
            if isinstance(v, (int, float)) and not (
                isinstance(v, float) and v in (float("nan"), float("inf"), float("-inf"))
            ):
                metrics[k] = float(v)

    if metrics:
        log_metrics(metrics)


def _train_and_evaluate(
    df: pd.DataFrame,
    settings: Settings,
    args: argparse.Namespace,
    purge_days: int,
    embargo_days: int,
    label: str = "",
    *,
    regime_labels: pd.Series | None = None,
) -> dict[str, Any] | None:
    """Train model and evaluate on a single symbol's DataFrame.

    Supports final OOS holdout split and regression mode.
    """
    if len(df) < 50:
        return None

    regression = getattr(args, "regression", False)
    final_oos_size = getattr(args, "final_oos_size", 0.0) or settings.final_oos_size

    # Feature columns: exclude non-feature columns
    feature_cols_actual = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not feature_cols_actual:
        print("[ERROR] No feature columns found in DataFrame")
        return None

    # --- Three-layer chronological split ---
    n_total = len(df)
    if final_oos_size > 0 and final_oos_size < 1.0:
        oos_end = n_total
        oos_start = int(n_total * (1.0 - final_oos_size))
        train_val_end = oos_start
        train_end = int(train_val_end * (1.0 - settings.test_size))
        print(f"[split] Train: {train_end} | Val: {train_val_end - train_end} "
              f"| Final OOS: {oos_end - oos_start} rows")
    else:
        train_end = int(n_total * (1.0 - settings.test_size))
        train_val_end = n_total
        oos_start = n_total
        oos_end = n_total
        print(f"[split] Train: {train_end} | Test: {n_total - train_end} rows")

    # Train + val set
    X_train = df.iloc[:train_end][feature_cols_actual]
    X_val = df.iloc[train_end:train_val_end][feature_cols_actual]
    y_train = df.iloc[:train_end]["target"]
    y_val = df.iloc[train_end:train_val_end]["target"]

    # Final OOS set (if enabled)
    X_oos: pd.DataFrame | None = None
    y_oos: pd.Series | None = None
    oos_returns: np.ndarray | None = None
    oos_prev_direction: pd.Series | None = None
    if oos_end > oos_start:
        X_oos = df.iloc[oos_start:oos_end][feature_cols_actual]
        y_oos = df.iloc[oos_start:oos_end]["target"]
        oos_returns = df.iloc[oos_start:oos_end]["future_return"].values
        oos_prev_direction = df.iloc[oos_start:oos_end]["target"].shift(1).fillna(0).astype(int)

    # Use val set for evaluation (test set in original terminology)
    X_test = X_val
    y_test = y_val
    actual_returns_test = df.iloc[train_end:train_val_end]["future_return"].values
    prev_direction_test = df.iloc[train_end:train_val_end]["target"].shift(1).fillna(0).astype(int)

    # Train
    if args.compare_models:
        print("[train] Comparing models (purged TS-CV) ...")
        results = compare_models(
            X_train, y_train,
            params=settings.xgb_params,
            cv_splits=settings.cv_splits,
            purge_days=purge_days,
            embargo_days=embargo_days,
            sample_weight_halflife=settings.sample_weight_halflife,
            train_window_days=settings.train_window_days,
            regression=regression,
        )
        _print_model_comparison(results, regression=regression)
        best = results[0]
        model = best["model"]
        cv_stats = {
            "mean": best["cv_mean"], "std": best["cv_std"],
            "scores": [], "n_folds": settings.cv_splits,
        }
        print(f"\n  Using best model: {best['model_name']}")
    else:
        print(
            f"[train] Training {'XGBoostRegressor' if regression else 'XGBoost'} "
            f"(purged TS-CV, purge={purge_days}d, embargo={embargo_days}d"
            f"{', calibrate' if settings.calibrate else ''}"
            f"{', regression' if regression else ''}) ..."
        )
        model, cv_stats = train_model(
            X_train, y_train,
            params=settings.xgb_params,
            cv_splits=settings.cv_splits,
            purge_days=purge_days,
            embargo_days=embargo_days,
            calibrate=settings.calibrate,
            cv_method=settings.cv_method,
            sample_weight_halflife=settings.sample_weight_halflife,
            train_window_days=settings.train_window_days,
            regression=regression,
        )
        print(f"  CV folds     : {cv_stats['n_folds']}")
        print(f"  CV score     : {cv_stats['mean']:.3f} +/- {cv_stats['std'] * 2:.3f}"
              f" ({cv_stats.get('scoring', 'accuracy')})")

    # Threshold optimisation (classification only)
    pred_threshold = 0.5
    if not regression and args.optimize_threshold:
        print("[tune] Optimising classification threshold ...")
        val_split = int(len(X_train) * 0.8)
        X_train2, X_val_opt = X_train.iloc[:val_split], X_train.iloc[val_split:]
        y_train2, y_val_opt = y_train.iloc[:val_split], y_train.iloc[val_split:]

        model.fit(X_train2, y_train2)
        pred_threshold, best_f1 = _optimize_threshold(model, X_val_opt, y_val_opt)
        print(f"  Optimal threshold: {pred_threshold:.2f} (val F1={best_f1:.3f})")
        model.fit(X_train, y_train)

    signal_threshold = pred_threshold if args.optimize_threshold else settings.prob_threshold
    if settings.prob_threshold > 0.50:
        print(f"[filter] Signal threshold: prob >= {signal_threshold:.2f}")

    # Evaluate
    print("[eval] Evaluating ...")

    report = evaluate(
        model, X_train, X_test, y_train, y_test,
        feature_names=feature_cols_actual,
        prev_direction_test=prev_direction_test if not regression else None,
        actual_returns_test=actual_returns_test,
        regression=regression,
        triple_barrier=settings.triple_barrier,
    )

    # Factor IC analysis
    if not regression:
        df_eval: pd.DataFrame = cast(pd.DataFrame, df)
        _, corr_warnings = compute_factor_correlation(df_eval, feature_cols_actual)
        factor_ic = compute_factor_ic(df_eval, feature_cols_actual)
        report["factor_ic"] = factor_ic
        report["factor_correlation_warnings"] = corr_warnings

    # Final OOS evaluation
    if X_oos is not None and y_oos is not None:
        print("[eval] Final OOS evaluation ...")
        oos_report = evaluate(
            model, X_train, X_test, y_train, y_oos,
            feature_names=feature_cols_actual,
            prev_direction_test=oos_prev_direction,
            actual_returns_test=oos_returns,
            regression=regression,
            triple_barrier=settings.triple_barrier,
        )
        report["final_oos"] = oos_report

    print_report(report)

    # Meta-labeling: secondary model to filter primary predictions
    if getattr(args, "meta_labeling", False) and _HAS_META:
        print("\n[meta] Training meta-labeling model ...")
        # Use val set for meta-label training (first 80% of test set)
        val_meta_end = len(X_test) * 8 // 10
        if val_meta_end > 50:
            X_meta_train = X_test.iloc[:val_meta_end]
            y_meta_train = y_test.iloc[:val_meta_end]
            X_meta_val = X_test.iloc[val_meta_end:]

            meta_model, meta_info = train_meta_model(
                X_train, y_train, model,
                X_meta_train, y_meta_train,
            )
            if meta_model is not None:
                report["meta_labeling"] = meta_info
                print(f"  Primary model accuracy on val: {meta_info['primary_accuracy']:.3f}")
                print(f"  Meta model CV accuracy:       {meta_info['meta_cv_mean']:.3f} "
                      f"+/- {meta_info['meta_cv_std']:.3f}")
                print(f"  Meta positive rate:           {meta_info['meta_positive_rate']:.1%}")

                # Evaluate meta-filter on held-out set
                _, _, meta_accept = apply_meta_filter(X_meta_val, model, meta_model)
                n_filtered = meta_accept.sum()
                n_total = len(meta_accept)
                print(f"  Meta-filter acceptance rate:   {n_filtered}/{n_total} "
                      f"({n_filtered/max(n_total,1):.1%})")
            else:
                print(f"  [meta] Cannot train: {meta_info.get('error', 'unknown')}")
        else:
            print("  [meta] Not enough data for meta-labeling")

    # MLflow: log evaluation metrics
    _log_eval_metrics(report, cv_stats)

    # Walk-forward backtest
    if args.backtest:
        print("\n[backtest] Running walk-forward backtest ...")

        bt_model_class = XGBRegressor if regression else XGBClassifier

        bt = walk_forward(
            df_eval if not regression else cast(pd.DataFrame, df),
            feature_cols=feature_cols_actual,
            model_class=bt_model_class,
            model_params=settings.xgb_params,
            threshold=signal_threshold,
            retrain_freq="ME",
            cost_bps=args.backtest_cost_bps,
            purge_days=purge_days,
            regime_labels=regime_labels,
            buy_cost_bps=settings.backtest_buy_cost_bps,
            sell_cost_bps=settings.backtest_sell_cost_bps,
            tradable_col="is_tradable",
            triple_barrier=settings.triple_barrier,
        )
        _print_backtest(bt)
        report["backtest"] = bt

        # MLflow: log backtest metrics
        log_backtest(bt)

    # Permutation test
    if getattr(args, "permutation_test", False) and not regression:
        print("\n[permtest] Running label permutation test (100 shuffles) ...")
        pt_result = permutation_test(
            XGBClassifier,
            settings.xgb_params,
            X_train,
            y_train,
            n_permutations=100,
        )
        report["permutation_test"] = pt_result
        print(f"  Real accuracy:       {pt_result['real_accuracy']:.3f}")
        print(f"  Shuffled mean:       {pt_result['shuffled_mean']:.3f}")
        print(f"  Shuffled std:        {pt_result['shuffled_std']:.3f}")
        print(f"  Shuffled max:        {pt_result['shuffled_max']:.3f}")
        print(f"  Percentile:          {pt_result['percentile']:.1%}")
        if pt_result["percentile"] > 0.95:
            print("  [OK] Real accuracy significantly above shuffled distribution.")
        elif pt_result["percentile"] > 0.80:
            print("  [WARN] Real accuracy only moderately above noise baseline.")
        else:
            print("  [FAIL] Real accuracy not clearly distinguishable from shuffled labels.")

    return report


# ---------------------------------------------------------------------------
# Single-symbol experiment (full pipeline)
# ---------------------------------------------------------------------------


def _run_single_experiment(
    settings: Settings,
    args: argparse.Namespace,
    purge_days: int,
    embargo_days: int,
    *,
    regime_labels: pd.Series | None = None,
) -> dict[str, Any] | None:
    """Full pipeline for a single symbol: load → prepare → train → eval."""
    df = _prepare_df(settings, use_lag=getattr(args, "use_lag_features", False))
    if df is None or len(df) < 50:
        return None

    report = _train_and_evaluate(df, settings, args, purge_days, embargo_days,
                                 regime_labels=regime_labels)

    # Feature ablation (after main evaluation)
    if getattr(args, "ablation", False) and not getattr(args, "regression", False):
        print("\n" + "=" * 60)
        print("FEATURE ABLATION")
        print("=" * 60)
        _run_ablation_on_df(df, settings, args, purge_days, embargo_days)
        if report:
            report["ablation_ran"] = True

    return report


def _run_ablation_on_df(
    df: pd.DataFrame,
    settings: Settings,
    args: argparse.Namespace,
    purge_days: int,
    embargo_days: int,
) -> None:
    """Run feature family ablation on a prepared DataFrame."""
    feature_cols_actual = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not feature_cols_actual:
        return

    split_idx = int(len(df) * (1 - settings.test_size))
    X_train = df.iloc[:split_idx][feature_cols_actual]
    X_test = df.iloc[split_idx:][feature_cols_actual]
    y_train = df.iloc[:split_idx]["target"]
    y_test = df.iloc[split_idx:]["target"]
    actual_returns_test = df.iloc[split_idx:]["future_return"].values

    # Filter families to only those with features present
    active_families = {
        k: [f for f in v if f in feature_cols_actual]
        for k, v in FEATURE_FAMILIES.items()
    }
    active_families = {k: v for k, v in active_families.items() if v}

    ablation_results = run_ablation(
        X_train, y_train, X_test, y_test,
        feature_families=active_families,
        model_class=XGBClassifier,
        model_params=settings.xgb_params,
        actual_returns_test=actual_returns_test,
    )

    baseline = ablation_results["baseline"]
    print("\n  Baseline (all features):")
    print(f"    Accuracy: {baseline['accuracy']:.3f}  "
          f"ROC AUC: {baseline['roc_auc']:.3f}  F1: {baseline['f1']:.3f}")

    print(f"\n  {'Family':<20} {'Delta Acc':>10} {'Delta F1':>10} {'Delta IC':>10}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")
    for fam_name, fam_data in ablation_results["families"].items():
        deltas = fam_data["deltas"]
        da = deltas.get("delta_accuracy", 0)
        df1 = deltas.get("delta_f1", 0)
        di = deltas.get("delta_ic", 0)
        print(f"  {fam_name:<20} {da:>+10.4f} {df1:>+10.4f} {di:>+10.4f}")


# ---------------------------------------------------------------------------
# Multi-symbol with optional industry neutralization
# ---------------------------------------------------------------------------


def _run_neutralized_multi(
    symbols: list[str],
    settings: Settings,
    args: argparse.Namespace,
    purge_days: int,
    embargo_days: int,
) -> list[dict[str, Any]]:
    """Load all symbols, pool, neutralize, split, train each.

    Returns list of per-symbol result dicts for aggregation.
    """
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
            backtest_enforce_price_limit=settings.backtest_enforce_price_limit,
            min_daily_amount=settings.min_daily_amount,
        )
        df = _prepare_df(sym_settings, use_lag=getattr(args, "use_lag_features", False))
        if df is not None and len(df) >= 50:
            df["_symbol"] = sym  # tag for splitting later
            all_dfs.append((sym, df))

        if (i + 1) % 100 == 0:
            print(f"  prepared {i + 1}/{len(symbols)} ...")

    if not all_dfs:
        print("[data] No valid symbols found.")
        return []

    print(f"[data] Prepared {len(all_dfs)} / {len(symbols)} symbols")

    # 2. Pool together
    pooled = pd.concat([d for _, d in all_dfs], ignore_index=True)
    pooled = pooled.sort_values(["trade_date", "_symbol"]).reset_index(drop=True)
    print(f"[pool] Pooled DataFrame: {len(pooled)} rows, "
          f"{pooled['_symbol'].nunique()} symbols")

    # Regime: compute from raw pool before neutralization
    pipeline_regime: pd.Series | None = None
    if args.regime:
        raw_dfs = [d for _, d in all_dfs]
        proxy = compute_market_proxy(raw_dfs)
        if not proxy.empty:
            pipeline_regime = classify_regime(proxy)
            summary = pipeline_regime.value_counts().to_dict()
            print(f"[regime] Market proxy from {len(raw_dfs)} symbols: "
                  f"bull={summary.get(1, 0)}d, range={summary.get(0, 0)}d, "
                  f"bear={summary.get(-1, 0)}d")

    # 3. Industry neutralization
    feature_cols_actual = [c for c in FEATURE_COLUMNS if c in pooled.columns]
    n_before = pooled[feature_cols_actual].isna().sum().sum()
    pooled = neutralize_industry(pooled, feature_cols_actual, industry_col="industry")
    n_after = pooled[feature_cols_actual].isna().sum().sum()
    if n_after > n_before:
        print(f"[indi] WARNING: neutralization introduced {n_after - n_before} NaN values")

    # 4. Split back and train each symbol
    all_reports: list[dict[str, Any]] = []
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
            final_oos_size=settings.final_oos_size,
            cv_splits=settings.cv_splits,
            calibrate=settings.calibrate,
            cv_method=settings.cv_method,
            prob_threshold=settings.prob_threshold,
            xgb_params=dict(settings.xgb_params),
            sample_weight_halflife=settings.sample_weight_halflife,
            train_window_days=settings.train_window_days,
            regression=settings.regression,
            backtest_buy_cost_bps=settings.backtest_buy_cost_bps,
            backtest_sell_cost_bps=settings.backtest_sell_cost_bps,
            backtest_enforce_price_limit=settings.backtest_enforce_price_limit,
        )

        result = _train_and_evaluate(
            sym_df, sym_settings, args, purge_days, embargo_days, label=sym,
            regime_labels=pipeline_regime,
        )
        if result is not None:
            all_reports.append({"symbol": sym, "report": result})
            report_count += 1

    print(f"\n[multi] Completed: {report_count} symbols evaluated")
    return all_reports


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def _save_results(
    report: dict[str, Any],
    settings: Settings,
    args: argparse.Namespace,
) -> str:
    """Save summary.json and config.used.yml to artifacts/runs/<timestamp>/.

    Returns the run directory path.
    """
    artifacts_root = Path(getattr(args, "artifacts_root", "artifacts"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_tag = settings.symbol.replace(".", "_")
    run_dir = artifacts_root / "runs" / f"{symbol_tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Serialize report (filter out non-serializable objects)
    serializable = _make_serializable(report)
    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str, ensure_ascii=False)

    # Save config snapshot
    config_dict = {
        "symbol": settings.symbol,
        "symbols": settings.symbols,
        "start_date": settings.start_date,
        "end_date": settings.end_date,
        "up_threshold": settings.up_threshold,
        "test_size": settings.test_size,
        "final_oos_size": settings.final_oos_size,
        "cv_splits": settings.cv_splits,
        "purge_days": args.purge_days,
        "embargo_days": args.embargo_days,
        "calibrate": settings.calibrate,
        "cv_method": settings.cv_method,
        "prob_threshold": settings.prob_threshold,
        "sample_weight_halflife": settings.sample_weight_halflife,
        "train_window_days": settings.train_window_days,
        "regression": settings.regression,
        "backtest": args.backtest,
        "backtest_cost_bps": args.backtest_cost_bps,
        "backtest_buy_cost_bps": settings.backtest_buy_cost_bps,
        "backtest_sell_cost_bps": settings.backtest_sell_cost_bps,
        "backtest_enforce_price_limit": settings.backtest_enforce_price_limit,
        "min_daily_amount": settings.min_daily_amount,
        "neutralize_industry": settings.neutralize_industry,
        "compare_models": args.compare_models,
        "xgb_params": settings.xgb_params,
    }
    config_path = run_dir / "config.used.yml"
    try:
        import yaml
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    except ImportError:
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"\n[save] Results saved to {run_dir}/")
    return str(run_dir)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _make_serializable(obj.tolist())
    elif isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def _save_multi_symbol_summary(
    all_reports: list[dict[str, Any]],
    settings: Settings,
    args: argparse.Namespace,
) -> str:
    """Save a multi-symbol summary CSV."""
    artifacts_root = Path(getattr(args, "artifacts_root", "artifacts"))
    run_dir = artifacts_root / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[dict[str, Any]] = []
    for entry in all_reports:
        sym = entry["symbol"]
        r = entry["report"]
        row: dict[str, Any] = {"symbol": sym}
        mode = r.get("mode", "classification")
        if mode == "regression":
            row["test_rmse"] = r.get("test_rmse")
            row["test_r2"] = r.get("test_r2")
        else:
            row["test_accuracy"] = r.get("test_accuracy")
            row["roc_auc"] = r.get("roc_auc")
            row["overfitting_gap"] = r.get("overfitting_gap")

        ic = r.get("ic", {})
        row["rank_ic"] = ic.get("rank_ic")
        row["icir"] = ic.get("icir")

        bt = r.get("backtest", {})
        row["sharpe"] = bt.get("sharpe")
        row["total_return"] = bt.get("total_return")
        row["win_rate"] = bt.get("win_rate")
        row["n_trades"] = bt.get("n_trades")

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by rank_ic descending
    if "rank_ic" in df.columns:
        df = df.sort_values("rank_ic", ascending=False, na_position="last")

    csv_path = run_dir / f"multi_symbol_summary_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[multi] Summary saved to {csv_path}")
    return str(csv_path)


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

    # Merge YAML xgb_params
    xgb_defaults = Settings.__dataclass_fields__["xgb_params"].default_factory()  # type: ignore[attr-defined]
    xgb_params = dict(xgb_defaults)
    if "xgb_params" in yaml_overrides and isinstance(yaml_overrides["xgb_params"], dict):
        xgb_params.update(yaml_overrides["xgb_params"])
        print(f"[config] XGBoost params from YAML: {list(yaml_overrides['xgb_params'].keys())}")

    # Inject num_class for triple barrier (3-class classification)
    triple_barrier = args.triple_barrier or yaml_overrides.get("triple_barrier", False)
    if triple_barrier:
        xgb_params["num_class"] = 3
        xgb_params.setdefault("objective", "multi:softmax")

    # YAML overrides for boolean flags
    _apply_yaml_boolean_overrides(args, yaml_overrides)

    # Build symbols list
    symbols = [args.symbol]
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif "symbols" in yaml_overrides:
        symbols = [s.strip() for s in str(yaml_overrides["symbols"]).split(",") if s.strip()]
        args.symbol = symbols[0]

    # Settings
    settings = Settings(
        symbol=symbols[0],
        symbols=symbols,
        start_date=args.start_date or yaml_overrides.get("start_date", "20150101"),
        end_date=args.end_date or yaml_overrides.get("end_date", ""),
        data_lake_root=Path(args.data_lake_root),
        min_listed_days=args.min_listed_days,
        min_daily_amount=args.min_daily_amount,
        up_threshold=args.threshold,
        test_size=args.test_size,
        final_oos_size=args.final_oos_size,
        cv_splits=args.cv_splits,
        calibrate=args.calibrate,
        cv_method=args.cv_method,
        prob_threshold=_resolve_prob_threshold(args, yaml_overrides),
        neutralize_industry=args.neutralize_industry,
        xgb_params=xgb_params,
        sample_weight_halflife=args.sample_weight_halflife,
        train_window_days=args.train_window_days,
        regression=args.regression,
        triple_barrier=triple_barrier,
        holding_period=args.holding_period,
        profit_take=args.profit_take,
        stop_loss=args.stop_loss,
        backtest=args.backtest,
        backtest_cost_bps=args.backtest_cost_bps,
        backtest_buy_cost_bps=args.backtest_buy_cost_bps,
        backtest_sell_cost_bps=args.backtest_sell_cost_bps,
        backtest_enforce_price_limit=not args.no_price_limit_filter,
    )

    purge_days = args.purge_days
    embargo_days = args.embargo_days

    _print_experiment_header(settings, purge_days, embargo_days)

    # MLflow tracking
    _start_mlflow_tracking(settings, symbols, purge_days, embargo_days)

    # Run experiment
    all_multi_reports: list[dict[str, Any]] = []
    report: dict[str, Any] | None = None

    if len(symbols) > 1 and settings.neutralize_industry:
        # Multi-symbol + industry neutralization
        all_multi_reports = _run_neutralized_multi(
            symbols, settings, args, purge_days, embargo_days
        )
        # Pick first report as main (for persistence)
        if all_multi_reports:
            report = all_multi_reports[0]["report"]
    elif len(symbols) > 1:
        # Multi-symbol, no neutralization (simple loop)
        print(f"\n[data] Loading {len(symbols)} symbols ...")
        pipeline_regime: pd.Series | None = None
        if args.regime:
            pipeline_regime = _compute_pipeline_regime(symbols, settings)

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
                min_daily_amount=settings.min_daily_amount,
                up_threshold=settings.up_threshold,
                test_size=settings.test_size,
                final_oos_size=settings.final_oos_size,
                cv_splits=settings.cv_splits,
                calibrate=settings.calibrate,
                cv_method=settings.cv_method,
                prob_threshold=settings.prob_threshold,
                xgb_params=dict(settings.xgb_params),
                sample_weight_halflife=settings.sample_weight_halflife,
                train_window_days=settings.train_window_days,
                regression=settings.regression,
                backtest_buy_cost_bps=settings.backtest_buy_cost_bps,
                backtest_sell_cost_bps=settings.backtest_sell_cost_bps,
                backtest_enforce_price_limit=settings.backtest_enforce_price_limit,
            )
            result = _run_single_experiment(
                sym_settings, args, purge_days, embargo_days,
                regime_labels=pipeline_regime,
            )
            if result is not None:
                all_multi_reports.append({"symbol": sym, "report": result})
                if report is None:
                    report = result
    else:
        # Single symbol
        regime = _compute_single_regime(settings) if args.regime else None
        report = _run_single_experiment(
            settings, args, purge_days, embargo_days,
            regime_labels=regime,
        )

    # Save multi-symbol summary
    if len(all_multi_reports) > 1:
        _save_multi_symbol_summary(all_multi_reports, settings, args)

    # Save results
    if args.save_results and report is not None:
        _save_results(report, settings, args)

    end_tracking()


def _apply_yaml_boolean_overrides(
    args: argparse.Namespace, yaml_overrides: dict[str, Any]
) -> None:
    """Apply YAML boolean overrides when CLI args are at defaults."""
    bool_flags = [
        "backtest", "regime", "calibrate", "neutralize_industry",
        "compare_models", "optimize_threshold", "regression",
        "ablation", "permutation_test", "save_results",
        "triple_barrier", "meta_labeling",
    ]
    for flag in bool_flags:
        if not getattr(args, flag, False) and yaml_overrides.get(flag):
            setattr(args, flag, True)


def _resolve_prob_threshold(
    args: argparse.Namespace, yaml_overrides: dict[str, Any]
) -> float:
    """Resolve prob_threshold with YAML override awareness."""
    if args.prob_threshold != 0.50:
        return args.prob_threshold
    if "prob_threshold" in yaml_overrides:
        return yaml_overrides["prob_threshold"]
    return 0.50


def _print_experiment_header(
    settings: Settings, purge_days: int, embargo_days: int
) -> None:
    """Print experiment configuration header."""
    print(f"Experiment: {', '.join(settings.symbols[:5])}"
          f"{'...' if len(settings.symbols) > 5 else ''}")
    print(f"  Data lake  : {settings.data_lake_root}")
    print(f"  Date range : {settings.start_date} - {settings.end_date}")
    print(f"  Threshold  : {settings.up_threshold:.3f}")
    print(f"  Test size  : {settings.test_size:.0%}")
    if settings.final_oos_size > 0:
        print(f"  Final OOS  : {settings.final_oos_size:.0%}")
    print(f"  CV purge   : {purge_days} days")
    print(f"  CV embargo : {embargo_days} day(s)")
    if settings.sample_weight_halflife > 0:
        print(f"  Sample wt  : exp_decay halflife={settings.sample_weight_halflife}d")
    if settings.train_window_days > 0:
        print(f"  Train win  : rolling {settings.train_window_days}d")
    if settings.triple_barrier:
        print(f"  Labels     : triple barrier K={settings.holding_period}d "
              f"pt={settings.profit_take:.0%} sl={settings.stop_loss:.0%}")
    if settings.regression:
        print("  Mode       : regression")
    if settings.calibrate:
        print(f"  Calibrate  : {settings.cv_method}")
    if settings.prob_threshold > 0.50:
        print(f"  Prob thresh: {settings.prob_threshold:.2f}")
    if settings.neutralize_industry:
        print("  Neutralize : industry")
    if settings.min_daily_amount > 0:
        print(f"  Min amount : {settings.min_daily_amount:,.0f} CNY")
    if not settings.backtest_enforce_price_limit:
        print("  Price limit: DISABLED")
    if settings.backtest:
        print(f"  Backtest   : buy={settings.backtest_buy_cost_bps:.0f}bps "
              f"sell={settings.backtest_sell_cost_bps:.0f}bps")


def _start_mlflow_tracking(
    settings: Settings, symbols: list[str], purge_days: int, embargo_days: int
) -> None:
    """Start MLflow tracking with all experiment params."""
    run_tag = symbols[0] if len(symbols) == 1 else f"multi_{len(symbols)}s"
    start_tracking(
        "time-series-ml",
        {
            "symbol": run_tag,
            "n_symbols": str(len(symbols)),
            "start_date": settings.start_date,
            "end_date": settings.end_date,
            "up_threshold": str(settings.up_threshold),
            "test_size": str(settings.test_size),
            "final_oos_size": str(settings.final_oos_size),
            "cv_splits": str(settings.cv_splits),
            "purge_days": str(purge_days),
            "embargo_days": str(embargo_days),
            "calibrate": str(settings.calibrate),
            "cv_method": settings.cv_method,
            "prob_threshold": str(settings.prob_threshold),
            "neutralize_industry": str(settings.neutralize_industry),
            "min_listed_days": str(settings.min_listed_days),
            "sample_weight_halflife": str(settings.sample_weight_halflife),
            "train_window_days": str(settings.train_window_days),
            "regression": str(settings.regression),
            "backtest_buy_cost_bps": str(settings.backtest_buy_cost_bps),
            "backtest_sell_cost_bps": str(settings.backtest_sell_cost_bps),
        },
    )


if __name__ == "__main__":
    main()
