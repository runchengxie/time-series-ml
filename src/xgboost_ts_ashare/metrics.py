"""Evaluation: classification report, baselines, ROC AUC, confusion matrix, IC/ICIR."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def majority_baseline(y_true: pd.Series) -> dict[str, Any]:
    """Always predict the most frequent class."""
    majority_class = int(y_true.mode().iloc[0])
    preds = np.full(len(y_true), majority_class)
    return {
        "name": "Majority",
        "accuracy": float(accuracy_score(y_true, preds)),
    }


def persistence_baseline(
    y_true: pd.Series, prev_direction: pd.Series
) -> dict[str, Any]:
    """Predict that tomorrow's direction equals today's direction."""
    preds = prev_direction.values
    return {
        "name": "Persistence (yesterday's direction)",
        "accuracy": float(accuracy_score(y_true, preds)),
    }


# ---------------------------------------------------------------------------
# IC / ICIR
# ---------------------------------------------------------------------------


def compute_ic_icir(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
) -> dict[str, float]:
    """Compute Rank IC (Spearman) and ICIR."""
    mask = ~(np.isnan(predictions) | np.isnan(actual_returns))
    if mask.sum() < 10:
        return {"rank_ic": float("nan"), "icir": float("nan"), "ic_p_value": float("nan")}

    pred = predictions[mask]
    ret = actual_returns[mask]

    result = spearmanr(pred, ret)
    ic = result.correlation  # type: ignore[assignment]
    p_value = result.pvalue   # type: ignore[assignment]

    # ICIR: rolling sub-windows
    window = min(63, len(pred) // 3)
    step = max(21, window // 3)
    sub_ics: list[float] = []
    for start in range(0, len(pred) - window + 1, step):
        sub_result = spearmanr(
            pred[start : start + window], ret[start : start + window]
        )
        sub_ics.append(float(sub_result.correlation))  # type: ignore[arg-type]

    if len(sub_ics) >= 2:
        ic_mean = float(np.mean(sub_ics))
        ic_std = float(np.std(sub_ics, ddof=1))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
    else:
        icir = float("nan")

    return {
        "rank_ic": float(ic),
        "icir": float(icir),
        "ic_p_value": float(p_value),
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str] | None = None,
    prev_direction_test: pd.Series | None = None,
    actual_returns_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return a dictionary with all evaluation metrics."""
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    y_pred_train = (prob_train >= 0.5).astype(int)
    y_pred_test = (prob_test >= 0.5).astype(int)

    train_acc = float(accuracy_score(y_train, y_pred_train))
    test_acc = float(accuracy_score(y_test, y_pred_test))

    roc_auc = float(roc_auc_score(y_test, prob_test))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    clf_report = classification_report(
        y_test, y_pred_test,
        target_names=["Down/Flat", "Up >=0.2%"],
        digits=3,
    )

    # Baselines
    baselines = [majority_baseline(y_test)]
    if prev_direction_test is not None:
        baselines.append(persistence_baseline(y_test, prev_direction_test))

    # IC / ICIR
    ic_info: dict[str, float] = {}
    if actual_returns_test is not None:
        ic_info = compute_ic_icir(prob_test, actual_returns_test)

    # Feature importance
    importance: dict[str, float] = {}
    if feature_names and hasattr(model, "feature_importances_"):
        importance = dict(
            sorted(
                zip(feature_names, model.feature_importances_, strict=True),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "roc_auc": roc_auc,
        "overfitting_gap": train_acc - test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": clf_report,
        "baselines": baselines,
        "ic": ic_info,
        "feature_importance": importance,
        "class_distribution_test": {
            "down_flat": int((y_test == 0).sum()),
            "up": int((y_test == 1).sum()),
            "up_pct": float((y_test == 1).mean()),
        },
    }


def print_report(report: dict[str, Any]) -> None:
    """Pretty-print an evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"  Train Accuracy:        {report['train_accuracy']:.3f}")
    print(f"  Test Accuracy:         {report['test_accuracy']:.3f}")
    print(f"  ROC AUC:               {report['roc_auc']:.3f}")
    print(f"  Overfitting Gap:       {report['overfitting_gap']:.3f}")
    print(f"  Precision:             {report['precision']:.3f}")
    print(f"  Recall:                {report['recall']:.3f}")
    print(f"  F1 Score:              {report['f1']:.3f}")

    # IC / ICIR
    ic = report.get("ic", {})
    if ic and not np.isnan(ic.get("rank_ic", float("nan"))):
        print(f"  Rank IC:               {ic['rank_ic']:.3f}")
        print(f"  ICIR (rolling):        {ic['icir']:.3f}")
        print(f"  IC p-value:            {ic['ic_p_value']:.4f}")

    # Overfitting diagnosis
    gap = report["overfitting_gap"]
    if gap < 0.05:
        print("\n  [OK] Low overfitting -- good generalisation.")
    elif gap < 0.10:
        print("\n  [WARN] Moderate overfitting -- consider tuning regularisation.")
    else:
        print("\n  [FAIL] High overfitting -- model may be memorising training data.")

    # IC diagnosis
    if ic:
        rank_ic = ic.get("rank_ic", 0)
        icir = ic.get("icir", 0)
        if not np.isnan(rank_ic):
            if rank_ic > 0.05:
                print("  [OK] Rank IC > 0.05 -- meaningful predictive signal.")
            elif rank_ic > 0:
                print("  [WARN] Rank IC > 0 but weak -- marginal signal.")
            else:
                print("  [FAIL] Rank IC <= 0 -- no directional signal.")
        if not np.isnan(icir):
            if icir > 1.0:
                print("  [OK] ICIR > 1.0 -- stable signal.")
            elif icir > 0.3:
                print("  [WARN] ICIR > 0.3 -- modest stability.")
            else:
                print("  [FAIL] ICIR < 0.3 -- signal unstable across sub-periods.")

    # Baselines
    print("\n-- Baselines --")
    for b in report["baselines"]:
        name = b["name"]
        acc = b["accuracy"]
        delta = report["test_accuracy"] - acc
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<40}: {acc:.3f}  (model {sign}{delta:.3f})")

    # Class distribution
    cd = report["class_distribution_test"]
    print("\n-- Test Class Distribution --")
    print(f"  Down/Flat: {cd['down_flat']} samples ({1 - cd['up_pct']:.1%})")
    print(f"  Up >=0.2%:  {cd['up']} samples ({cd['up_pct']:.1%})")

    # Classification report
    print("\n-- Classification Report (Test) --")
    print(report["classification_report"])

    # Confusion matrix
    cm = report["confusion_matrix"]
    print("-- Confusion Matrix --")
    print("           Pred Down   Pred Up")
    print(f"  True Down    {cm['tn']:>5}      {cm['fp']:>5}")
    print(f"  True Up      {cm['fn']:>5}      {cm['tp']:>5}")

    # Feature importance
    if report["feature_importance"]:
        print("\n-- Feature Importance --")
        for feat, imp in report["feature_importance"].items():
            print(f"  {feat:<25}: {imp:.3f}")

    # Factor IC
    factor_ic = report.get("factor_ic", {})
    if factor_ic:
        print("\n-- Factor IC Analysis --")
        print(f"  {'Feature':<25} {'Rank IC':>8}  {'Abs IC':>8}")
        for feat, ic_val in sorted(factor_ic.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feat:<25} {ic_val:>8.3f}  {abs(ic_val):>8.3f}")

    # Factor correlation
    warnings = report.get("factor_correlation_warnings", [])
    if warnings:
        print("\n-- High Factor Correlations (|r| > 0.8) --")
        for w in warnings:
            print(f"  {w}")


# ---------------------------------------------------------------------------
# Factor IC analysis
# ---------------------------------------------------------------------------


def compute_factor_ic(
    df: pd.DataFrame,
    feature_cols: list[str],
    return_col: str = "future_return",
) -> dict[str, float]:
    """Compute per-feature Rank IC (Spearman correlation with future returns)."""
    result: dict[str, float] = {}
    valid = df.dropna(subset=[return_col, *feature_cols])
    if len(valid) < 30:
        return {}

    for feat in feature_cols:
        if feat not in valid.columns:
            continue
        sr = spearmanr(valid[feat], valid[return_col])
        result[feat] = float(sr.correlation)  # type: ignore[arg-type]
    return result


def compute_ic_decay(
    df: pd.DataFrame,
    feature_cols: list[str],
    max_lag: int = 20,
) -> dict[str, list[float]]:
    """Compute IC decay: rank correlation at successive forward horizons."""
    result: dict[str, list[float]] = {}
    close = df["close"]
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        ics: list[float] = []
        for lag in range(1, max_lag + 1):
            fwd_ret = close.shift(-lag) / close - 1.0
            valid = pd.DataFrame({"f": df[feat], "r": fwd_ret}).dropna()
            if len(valid) < 30:
                ics.append(float("nan"))
            else:
                r = spearmanr(valid["f"], valid["r"])
                ics.append(float(r.correlation))  # type: ignore[arg-type]
        result[feat] = ics
    return result


def compute_factor_correlation(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Compute feature correlation matrix and flag pairs with |r| > 0.8."""
    valid = df[feature_cols].dropna()
    corr = valid.corr(numeric_only=True)  # type: ignore[call-arg]
    warnings: list[str] = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.8:
                warnings.append(
                    f"  {feature_cols[i]} <-> {feature_cols[j]}: r={r:.3f}"
                )
    return corr, warnings
