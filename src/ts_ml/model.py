"""Model training with time-series cross-validation, multi-model comparison,
probability calibration (CalibratedClassifierCV), and optional regression mode.

Supports:
- Sample weighting (exp_decay) for temporal relevance
- Rolling training windows
- Ridge as sanity baseline
- Regression mode (predict future_return directly)
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

from .crossval import PurgedTimeSeriesSplit, purged_cross_val_score

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def _make_xgb(params: dict[str, Any]) -> XGBClassifier:
    xgb_p = dict(params)
    # Handle 3-class for triple barrier
    num_class = xgb_p.pop("num_class", None)
    if num_class is not None and num_class > 2:
        xgb_p.setdefault("objective", "multi:softmax")
        return XGBClassifier(num_class=num_class, **xgb_p)
    return XGBClassifier(**xgb_p)


def _make_xgb_regressor(params: dict[str, Any]) -> XGBRegressor:
    """XGBoost regressor for --regression mode."""
    reg_params = {
        k: v for k, v in params.items()
        if k not in ("objective", "eval_metric")
    }
    reg_params.setdefault("objective", "reg:squarederror")
    reg_params.setdefault("eval_metric", "rmse")
    return XGBRegressor(**reg_params)


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


def _make_rf_regressor(params: dict[str, Any]) -> RandomForestRegressor:
    defaults = {
        "n_estimators": 200,
        "max_depth": 5,
        "random_state": params.get("random_state", 42),
        "n_jobs": -1,
    }
    return RandomForestRegressor(**defaults)


def _make_ridge(params: dict[str, Any]) -> Ridge:
    """Ridge regression — sanity baseline for regression mode."""
    return Ridge(alpha=1.0, random_state=params.get("random_state", 42))


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


def _make_lgb_regressor(params: dict[str, Any]) -> Any:
    """LightGBM regressor — optional dependency."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return None
    defaults = {
        "n_estimators": 200,
        "learning_rate": 0.01,
        "max_depth": 3,
        "random_state": params.get("random_state", 42),
        "verbose": -1,
    }
    return LGBMRegressor(**defaults)


# (maker_fn, required, is_classifier)
MODEL_REGISTRY: dict[str, tuple[Any, bool, bool]] = {
    "xgboost": (_make_xgb, True, True),
    "logistic": (_make_lr, True, True),
    "random_forest": (_make_rf, True, True),
    "lightgbm": (_make_lgb, False, True),
    "ridge": (_make_ridge, True, False),  # regression-only sanity baseline
}

REGRESSION_REGISTRY: dict[str, tuple[Any, bool]] = {
    "xgboost": (_make_xgb_regressor, True),
    "random_forest": (_make_rf_regressor, True),
    "lightgbm": (_make_lgb_regressor, False),
    "ridge": (_make_ridge, True),
}


# ---------------------------------------------------------------------------
# Sample weighting
# ---------------------------------------------------------------------------


def _compute_sample_weights(
    dates: pd.Series,
    halflife: int,
) -> np.ndarray:
    """Compute exponential decay sample weights.

    Newer samples get higher weight: weight[t] = exp(-λ * (T - t) / halflife)
    where λ = ln(2) so that weight halves every `halflife` days.

    Parameters
    ----------
    dates : pd.Series
        Chronologically ordered dates (datetime).
    halflife : int
        Number of trading days after which weight halves.
        0 or negative → uniform weights.

    Returns
    -------
    np.ndarray
        Sample weights, same length as dates.
    """
    if halflife <= 0:
        return np.ones(len(dates))

    # Days since most recent date, then divide by halflife, then exponential
    latest = dates.max()
    days_ago = (latest - dates).dt.days.values.astype(float)
    lam = np.log(2) / halflife
    weights = np.exp(-lam * days_ago)

    # Normalise to mean=1 so loss scale is preserved
    weights = weights / weights.mean()

    return weights


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
    sample_weight_halflife: int = 0,
    train_window_days: int = 0,
    regression: bool = False,
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
        One of: xgboost, logistic, random_forest, lightgbm, ridge.
    calibrate : bool
        If True, wrap the final model with CalibratedClassifierCV.
        Only effective in classification mode.
    cv_method : str
        "isotonic" or "sigmoid" for calibration.
    sample_weight_halflife : int
        Halflife in trading days for exponential decay sample weights.
        0 = uniform weights.
    train_window_days : int
        If > 0, only use the last N trading days for training.
    regression : bool
        If True, use regression models and predict future_return directly.
    """
    # --- Select model maker ---
    if regression:
        maker, _ = REGRESSION_REGISTRY[model_type]
    else:
        maker, _, _ = MODEL_REGISTRY[model_type]
    estimator = maker(params)

    # --- Apply rolling training window ---
    if train_window_days > 0 and len(X_train) > train_window_days:
        X_train = X_train.iloc[-train_window_days:]
        y_train = y_train.iloc[-train_window_days:]
        print(f"[train] Rolling window: last {train_window_days} days "
              f"({len(X_train)} rows)")

    # --- Compute sample weights ---
    sample_weight: np.ndarray | None = None
    if sample_weight_halflife > 0:
        if "trade_date" in X_train.columns:
            dates = cast(pd.Series, X_train["trade_date"])
            X_train_for_cv = X_train.drop(columns=["trade_date"])
        else:
            # Reconstruct date ordering from index position
            dates = pd.Series(
                pd.date_range(end=pd.Timestamp.today(), periods=len(X_train), freq="B"),
                index=X_train.index,
            )
            X_train_for_cv = X_train

        sample_weight = _compute_sample_weights(dates, sample_weight_halflife)
        print(f"[train] Sample weights: exp_decay halflife={sample_weight_halflife}d "
              f"(range [{sample_weight.min():.3f}, {sample_weight.max():.3f}])")
    else:
        X_train_for_cv = X_train

    # Balanced class weights for triple barrier
    if regression is False and len(np.unique(y_train)) > 2:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        cw = compute_class_weight("balanced", classes=classes, y=y_train.values)
        class_weight_dict = dict(zip(classes, cw, strict=False))
        bal_weights = np.array([class_weight_dict[v] for v in y_train.values])
        sample_weight = sample_weight * bal_weights if sample_weight is not None else bal_weights
        print(f"[train] Balanced class weights: "
              f"{', '.join(f'cls {int(k)}={v:.2f}' for k, v in class_weight_dict.items())}")

    # --- Cross-validation ---
    cv = PurgedTimeSeriesSplit(
        n_splits=cv_splits,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )

    cv_scores = purged_cross_val_score(
        estimator,
        X_train_for_cv.values,
        y_train.values.astype(float),  # type: ignore[arg-type]
        cv,
        scoring="accuracy" if not regression else "neg_mean_squared_error",
        sample_weight=sample_weight,
    )

    # --- Fit final model on all training data ---
    model = maker(params)
    if hasattr(model, "fit") and sample_weight is not None:
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        except TypeError:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    # --- Optional calibration (classification only) ---
    if calibrate and not regression:
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
        "scoring": "accuracy" if not regression else "neg_mse",
    }
    return model, cv_stats


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
    cv_splits: int = 5,
    purge_days: int = 20,
    embargo_days: int = 1,
    sample_weight_halflife: int = 0,
    train_window_days: int = 0,
    regression: bool = False,
) -> list[dict[str, Any]]:
    """Train and compare all available models.

    Returns a list of dicts with model_name, cv_mean, cv_std, model, is_classifier.
    """
    registry: dict[str, Any] = REGRESSION_REGISTRY if regression else {
        k: (m, r) for k, (m, r, _) in MODEL_REGISTRY.items()
    }

    results: list[dict[str, Any]] = []
    for name, (_maker_fn, _required) in registry.items():
        try:
            model, cv_stats = train_model(
                X_train, y_train,
                params=params,
                cv_splits=cv_splits,
                purge_days=purge_days,
                embargo_days=embargo_days,
                model_type=name,
                sample_weight_halflife=sample_weight_halflife,
                train_window_days=train_window_days,
                regression=regression,
            )
            results.append({
                "model_name": name,
                "cv_mean": cv_stats["mean"],
                "cv_std": cv_stats["std"],
                "model": model,
                "is_classifier": not regression,
            })
        except Exception:
            continue
    results.sort(key=lambda r: r["cv_mean"], reverse=not regression)
    return results
