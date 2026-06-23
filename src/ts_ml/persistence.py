"""Result persistence — save experiment results to disk."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings


def make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    elif isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def save_results(
    report: dict[str, Any],
    settings: Settings,
    *,
    artifacts_root: str = "artifacts",
    purge_days: int = 20,
    embargo_days: int = 1,
    backtest: bool = False,
    backtest_cost_bps: float = 5.0,
    compare_models: bool = False,
) -> str:
    """Save summary.json and config.used.yml to artifacts/runs/<timestamp>/.

    Returns the run directory path.
    """
    root = Path(artifacts_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_tag = settings.symbol.replace(".", "_")
    run_dir = root / "runs" / f"{symbol_tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Serialize report (filter out non-serializable objects)
    serializable = make_serializable(report)
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
        "purge_days": purge_days,
        "embargo_days": embargo_days,
        "calibrate": settings.calibrate,
        "cv_method": settings.cv_method,
        "prob_threshold": settings.prob_threshold,
        "sample_weight_halflife": settings.sample_weight_halflife,
        "train_window_days": settings.train_window_days,
        "regression": settings.regression,
        "backtest": backtest,
        "backtest_cost_bps": backtest_cost_bps,
        "backtest_buy_cost_bps": settings.backtest_buy_cost_bps,
        "backtest_sell_cost_bps": settings.backtest_sell_cost_bps,
        "backtest_enforce_price_limit": settings.backtest_enforce_price_limit,
        "min_daily_amount": settings.min_daily_amount,
        "neutralize_industry": settings.neutralize_industry,
        "compare_models": compare_models,
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


def save_multi_symbol_summary(
    all_reports: list[dict[str, Any]],
    *,
    artifacts_root: str = "artifacts",
) -> str:
    """Save a multi-symbol summary CSV."""
    root = Path(artifacts_root)
    run_dir = root / "runs"
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
