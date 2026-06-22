"""Lightweight MLflow experiment tracking for time-series-ml.

Uses local file store (./mlruns/) — no server required.
Silently no-ops if mlflow is not installed.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

# ---- Lazy import: mlflow is an optional dependency ----
try:
    import mlflow
    _MLFLOW = mlflow
except ImportError:
    _MLFLOW = None


def _project_root() -> Path:
    """Resolve project root from this file's location."""
    return Path(__file__).resolve().parents[2]


def start_tracking(
    experiment_name: str,
    params: dict[str, Any],
    *,
    run_name: str = "",
) -> None:
    """Start an MLflow run. No-op if mlflow is not installed.

    Parameters
    ----------
    experiment_name : str
        Top-level experiment group (e.g. "time-series-ml").
    params : dict
        Flat key-value pairs logged as MLflow params.
    run_name : str
        Human-readable run identifier; auto-generated if empty.
    """
    if _MLFLOW is None:
        return

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"file://{_project_root() / 'mlruns'}",
    )
    _MLFLOW.set_tracking_uri(tracking_uri)
    _MLFLOW.set_experiment(experiment_name)
    _MLFLOW.start_run(run_name=run_name or _auto_run_name())

    for key, value in sorted(params.items()):
        _MLFLOW.log_param(key, str(value))


def log_metrics(metrics: dict[str, float]) -> None:
    """Log scalar metrics to the active MLflow run."""
    if _MLFLOW is None:
        return

    for key, value in metrics.items():
        _MLFLOW.log_metric(key, value)


def log_backtest(bt: dict[str, Any]) -> None:
    """Log walk-forward backtest metrics to MLflow."""
    if _MLFLOW is None:
        return

    bt_metrics: dict[str, float] = {}
    for k in (
        "n_trades", "total_return", "annual_return", "annual_vol",
        "sharpe", "max_drawdown", "win_rate", "profit_factor",
        "turnover", "signal_rate",
    ):
        v = bt.get(k)
        if isinstance(v, (int, float)) and not (
            isinstance(v, float) and v in (float("inf"), float("-inf"), float("nan"))
        ):
            bt_metrics[k] = float(v)

    for key, value in bt_metrics.items():
        _MLFLOW.log_metric(key, value)


def end_tracking() -> None:
    """End the active MLflow run."""
    if _MLFLOW is None:
        return
    _MLFLOW.end_run()


def is_available() -> bool:
    """Check whether mlflow is importable."""
    return _MLFLOW is not None


def _auto_run_name() -> str:
    """Generate a timestamp-based run name."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
