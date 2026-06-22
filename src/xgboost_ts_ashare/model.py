"""Model training with time-series cross-validation, multi-model comparison,
and probability calibration (CalibratedClassifierCV)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .crossval import PurgedTimeSeriesSplit, purged_cross_val_score

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def _make_xgb(params: dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(**params)


def _make_lr(params: dict[str, Any]) -> LogisticRegression:
    defaults: dict[str, Any] = {"max_iter": 1000, "random_state": params.get("random_state", 42)}
    return LogisticRegression(**defaults)


def _make_rf(params: dict[str, Any]) -> RandomForestClassifier:
    defaults = {
        "n_estimators": 200,
        "max_depth": 5,
        "random_state": params.get("random_state", 42),
        "n_jobs": -1,
    }
    return RandomForestClassifier(**defaults)


def _make_lgb(params: dict[str, Any]) -> Any:
    """LightGBM — optional dependency."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None
    defaults = {
        "n_estimators": 200,
        "learning_rate": 0.01,
        "max_depth": 3,
        "random_state": params.get("random_state", 42),
        "verbose": -1,
    }
    return LGBMClassifier(**defaults)


MODEL_REGISTRY: dict[str, Any] = {
    "xgboost": (_make_xgb, True),
    "logistic": (_make_lr, True),
    "random_forest": (_make_rf, True),
    "lightgbm": (_make_lgb, False),  # optional
}


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------


def calibrate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "isotonic",
    cv_splits: int = 5,
    purge_days: int = 20,
    embargo_days: int = 1,
) -> CalibratedClassifierCV:
    """Wrap a trained model with CalibratedClassifierCV for probability calibration.

    Uses purged time-series CV internally to avoid leakage.

    Parameters
    ----------
    method : str
        "isotonic" (non-parametric, needs more data) or "sigmoid" (Platt scaling).
    cv_splits, purge_days, embargo_days : int
        Cross-validation parameters for the calibrator's internal CV.
    """
    cv = PurgedTimeSeriesSplit(
        n_splits=min(cv_splits, 3),  # fewer splits for calibration CV
        purge_days=purge_days,
        embargo_days=embargo_days,
    )

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=cv.split(X_train.values, y_train.values),  # type: ignore[arg-type]
        n_jobs=-1,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
    cv_splits: int = 5,
    purge_days: int = 20,
    embargo_days: int = 1,
    model_type: str = "xgboost",
    calibrate: bool = False,
    cv_method: str = "isotonic",
) -> tuple[Any, dict[str, float]]:
    """Fit model with purged time-series CV, optionally calibrate probabilities.

    Parameters
    ----------
    purge_days : int
        Samples to drop from end of each training fold.
        Set to max(window) of rolling features (default 20 for SMA20).
    embargo_days : int
        Gap between train and test (default 1 for daily next-day labels).
    model_type : str
        One of: xgboost, logistic, random_forest, lightgbm.
    calibrate : bool
        If True, wrap the final model with CalibratedClassifierCV.
    cv_method : str
        "isotonic" or "sigmoid" for calibration.
    """
    maker, _ = MODEL_REGISTRY[model_type]
    estimator = maker(params)

    cv = PurgedTimeSeriesSplit(
        n_splits=cv_splits,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )

    cv_scores = purged_cross_val_score(
        estimator,
        X_train.values,
        y_train.values.astype(float),  # type: ignore[arg-type]
        cv,
        scoring="accuracy",
    )

    # Fit final model on all training data
    model = maker(params)
    model.fit(X_train, y_train)

    # Optional calibration
    if calibrate:
        print("[calib] Fitting CalibratedClassifierCV ...")
        model = calibrate_model(
            model, X_train, y_train,
            method=cv_method,
            cv_splits=cv_splits,
            purge_days=purge_days,
            embargo_days=embargo_days,
        )

    cv_stats = {
        "mean": float(np.mean(cv_scores)),
        "std": float(np.std(cv_scores, ddof=1)),
        "scores": [float(s) for s in cv_scores],
        "n_folds": len(cv_scores),
    }
    return model, cv_stats


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
    cv_splits: int = 5,
    purge_days: int = 20,
    embargo_days: int = 1,
) -> list[dict[str, Any]]:
    """Train and compare all available models.

    Returns a list of dicts with model_name, cv_mean, cv_std, model.
    """
    results: list[dict[str, Any]] = []
    for name, (_maker_fn, _required) in MODEL_REGISTRY.items():
        try:
            model, cv_stats = train_model(
                X_train, y_train,
                params=params,
                cv_splits=cv_splits,
                purge_days=purge_days,
                embargo_days=embargo_days,
                model_type=name,
            )
            results.append({
                "model_name": name,
                "cv_mean": cv_stats["mean"],
                "cv_std": cv_stats["std"],
                "model": model,
            })
        except Exception:
            continue
    results.sort(key=lambda r: r["cv_mean"], reverse=True)
    return results
