"""Evaluation: classification report, baselines, ROC AUC, confusion matrix, IC/ICIR.

Also includes:
- Permutation test (label shuffling to detect noise-fitting)
- Rolling IC window diagnostics
- Feature ablation helpers
- Regression-mode evaluation
- Ridge sanity baseline warning
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
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


def compute_rolling_ic(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    window_days: int = 126,
    step_days: int = 21,
) -> list[dict[str, float]]:
    """Compute rolling-window Rank IC for time-series diagnostics.

    Parameters
    ----------
    window_days : int
        Rolling window size in trading days (126 ≈ 6 months).
    step_days : int
        Step size between windows (21 ≈ 1 month).

    Returns
    -------
    list[dict]
        Each dict has: window_end, ic, n_samples.
    """
    mask = ~(np.isnan(predictions) | np.isnan(actual_returns))
    pred = predictions[mask]
    ret = actual_returns[mask]

    n = len(pred)
    if n < window_days:
        return []

    results: list[dict[str, float]] = []
    for start in range(0, n - window_days + 1, step_days):
        end = start + window_days
        p_win = pred[start:end]
        r_win = ret[start:end]
        if len(p_win) < 30:
            continue
        sr = spearmanr(p_win, r_win)
        results.append({
            "window_end": float(end),
            "ic": float(sr.correlation),  # type: ignore[arg-type]
            "n_samples": float(len(p_win)),
        })

    return results


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def permutation_test(
    model_class: Any,
    model_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_permutations: int = 100,
    random_state: int = 42,
) -> dict[str, Any]:
    """Shuffle labels N times, retrain, and record test accuracy.

    Returns the real accuracy, the distribution of shuffled accuracies,
    and the percentile rank of the real accuracy in the shuffled distribution.

    Parameters
    ----------
    model_class : class
        e.g. XGBClassifier (NOT an instance).
    model_params : dict
        Params to pass to model_class constructor.
    n_permutations : int
        Number of shuffled-label runs (default 100).
    """
    rng = np.random.RandomState(random_state)

    # Chronological train/test split (last 20% as test)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Real accuracy
    real_model = model_class(**model_params)
    real_model.fit(X_train, y_train)
    real_pred = real_model.predict(X_test)
    real_acc = float(accuracy_score(y_test, real_pred))

    # Shuffled accuracies
    shuffled_accs: list[float] = []
    for _ in range(n_permutations):
        y_shuffled = pd.Series(rng.permutation(y_train.values), index=y_train.index)
        model = model_class(**model_params)
        model.fit(X_train, y_shuffled)
        pred = model.predict(X_test)
        shuffled_accs.append(float(accuracy_score(y_test, pred)))

    shuffled_arr = np.array(shuffled_accs)
    percentile = float((shuffled_arr < real_acc).mean())

    return {
        "real_accuracy": real_acc,
        "shuffled_mean": float(shuffled_arr.mean()),
        "shuffled_std": float(shuffled_arr.std()),
        "shuffled_max": float(shuffled_arr.max()),
        "shuffled_min": float(shuffled_arr.min()),
        "percentile": percentile,  # fraction of shuffled accs BELOW real accuracy
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# Feature ablation
# ---------------------------------------------------------------------------


def run_ablation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_families: dict[str, list[str]],
    model_class: Any,
    model_params: dict[str, Any],
    actual_returns_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run feature-family ablation: train with all features, then minus each family.

    Parameters
    ----------
    feature_families : dict
        e.g. {"momentum": ["SMA5_diff", "SMA10_diff", ...], "volatility": [...], ...}
        Features NOT in any family are kept in all runs as base.

    Returns
    -------
    dict
        baseline results + per-family minus-results with delta.
    """
    all_features = list(X_train.columns)
    family_features: set[str] = set()
    for feats in feature_families.values():
        family_features.update(feats)

    # Baseline (all features)
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    baseline = _eval_ablation_run(model, X_test, y_test, actual_returns_test)

    results: dict[str, Any] = {
        "baseline": baseline,
        "families": {},
    }

    # Minus each family
    for fam_name, fam_feats in feature_families.items():
        keep = [f for f in all_features if f not in fam_feats]
        if len(keep) == 0:
            continue

        model = model_class(**model_params)
        model.fit(X_train[keep], y_train)
        fam_result = _eval_ablation_run(
            model,
            cast(pd.DataFrame, X_test[keep]),
            y_test,
            actual_returns_test,
        )

        # Deltas
        deltas: dict[str, float] = {}
        for k in ("accuracy", "roc_auc", "f1"):
            if k in baseline and k in fam_result:
                deltas[f"delta_{k}"] = fam_result[k] - baseline[k]
        if "ic" in baseline and "ic" in fam_result:
            deltas["delta_ic"] = fam_result["ic"] - baseline["ic"]

        results["families"][fam_name] = {
            "result": fam_result,
            "deltas": deltas,
        }

    return results


def _eval_ablation_run(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    actual_returns_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Quick evaluation for ablation comparison."""
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, pred))
    try:
        auc = float(roc_auc_score(y_test, prob))
    except ValueError:
        auc = float("nan")

    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result: dict[str, Any] = {
        "accuracy": acc,
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    if actual_returns_test is not None:
        ic_info = compute_ic_icir(prob, actual_returns_test)
        result["ic"] = ic_info.get("rank_ic", float("nan"))
        result["icir"] = ic_info.get("icir", float("nan"))

    return result


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
    regression: bool = False,
    triple_barrier: bool = False,
) -> dict[str, Any]:
    """Return a dictionary with all evaluation metrics."""
    if regression:
        return _evaluate_regression(
            model, X_train, X_test, y_train, y_test,
            feature_names=feature_names,
            actual_returns_test=actual_returns_test,
        )

    if triple_barrier:
        return _evaluate_triple_barrier(
            model, X_train, X_test, y_train, y_test,
            feature_names=feature_names,
            prev_direction_test=prev_direction_test,
            actual_returns_test=actual_returns_test,
        )

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

    # Rolling IC
    rolling_ic_6m: list[dict[str, float]] = []
    rolling_ic_12m: list[dict[str, float]] = []
    if actual_returns_test is not None:
        rolling_ic_6m = compute_rolling_ic(prob_test, actual_returns_test, window_days=126)
        rolling_ic_12m = compute_rolling_ic(prob_test, actual_returns_test, window_days=252)

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

    result: dict[str, Any] = {
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
        "rolling_ic_6m": rolling_ic_6m,
        "rolling_ic_12m": rolling_ic_12m,
        "feature_importance": importance,
        "class_distribution_test": {
            "down_flat": int((y_test == 0).sum()),
            "up": int((y_test == 1).sum()),
            "up_pct": float((y_test == 1).mean()),
        },
        "mode": "classification",
    }
    return result


def _evaluate_regression(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str] | None = None,
    actual_returns_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate a regression model (--regression mode)."""
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_rmse = float(np.sqrt(mean_squared_error(y_train, pred_train)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    train_r2 = float(r2_score(y_train, pred_train))
    test_r2 = float(r2_score(y_test, pred_test))

    # IC with actual returns (use predictions as signal)
    ic_info: dict[str, float] = {}
    if actual_returns_test is not None:
        ic_info = compute_ic_icir(
            pred_test.flatten() if pred_test.ndim > 1 else pred_test,
            actual_returns_test,
        )

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

    result: dict[str, Any] = {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "overfitting_gap_rmse": test_rmse - train_rmse,
        "ic": ic_info,
        "feature_importance": importance,
        "mode": "regression",
    }
    return result


def _evaluate_triple_barrier(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str] | None = None,
    prev_direction_test: pd.Series | None = None,
    actual_returns_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate a triple-barrier 3-class model (-1=stop, 0=timeout, +1=profit)."""
    prob_train = model.predict_proba(X_train)
    prob_test = model.predict_proba(X_test)

    # XGBoost internal: [0=stop_loss, 1=timeout, 2=profit_take]
    # Map to original: [-1, 0, 1]
    label_map = {0: -1, 1: 0, 2: 1}
    profit_col = 2  # profit_take is class 2 internally

    prob_profit = prob_test[:, profit_col]
    y_pred_train = np.argmax(prob_train, axis=1)
    y_pred_test = np.argmax(prob_test, axis=1)

    pred_test_mapped = np.array([{0: -1, 1: 0, 2: 1}[c] for c in y_pred_test], dtype=np.int8)

    # Map y_train/y_test back to original [-1,0,1] for metrics
    # y_train/y_test are already [0,1,2] internally
    y_train_orig = y_train.values.astype(int)
    y_test_orig = y_test.values.astype(int)

    train_acc = float(accuracy_score(y_train_orig, y_pred_train))
    test_acc = float(accuracy_score(y_test_orig, y_pred_test))

    cm = confusion_matrix(y_test_orig, y_pred_test, labels=[0, 1, 2])

    # Map original labels for display
    y_test_display = np.array([label_map.get(v, 0) for v in y_test_orig], dtype=np.int8)
    pred_display = pred_test_mapped

    y_buy_true = (y_test_display == 1).astype(int)
    y_buy_pred = (pred_display == 1).astype(int)
    precision = (y_buy_true & y_buy_pred).sum() / max(y_buy_pred.sum(), 1)
    recall = (y_buy_true & y_buy_pred).sum() / max(y_buy_true.sum(), 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    baselines = [majority_baseline_triple(pd.Series(y_test_orig))]

    ic_info: dict[str, float] = {}
    if actual_returns_test is not None:
        ic_info = compute_ic_icir(prob_profit, actual_returns_test)

    rolling_ic_6m: list[dict[str, float]] = []
    rolling_ic_12m: list[dict[str, float]] = []
    if actual_returns_test is not None:
        rolling_ic_6m = compute_rolling_ic(prob_profit, actual_returns_test, window_days=126)
        rolling_ic_12m = compute_rolling_ic(prob_profit, actual_returns_test, window_days=252)

    importance: dict[str, float] = {}
    if feature_names and hasattr(model, "feature_importances_"):
        importance = dict(
            sorted(
                zip(feature_names, model.feature_importances_, strict=True),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    vals, counts = np.unique(y_test_display, return_counts=True)
    label_dist: dict[str, int] = {}
    for v, c in zip(vals, counts, strict=False):
        label_dist[str(int(v))] = int(c)

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "overfitting_gap": train_acc - test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix_3class": {
            "labels": [-1, 0, 1],
            "matrix": cm.tolist(),
        },
        "baselines": baselines,
        "ic": ic_info,
        "rolling_ic_6m": rolling_ic_6m,
        "rolling_ic_12m": rolling_ic_12m,
        "feature_importance": importance,
        "label_distribution": label_dist,
        "mode": "triple_barrier",
    }


def majority_baseline_triple(y_true: pd.Series) -> dict[str, Any]:
    """Most frequent label in 3-class setting."""
    majority_val = int(y_true.mode().iloc[0])
    preds = np.full(len(y_true), majority_val)
    return {
        "name": f"Majority (always {majority_val})",
        "accuracy": float(accuracy_score(y_true, preds)),
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_report(report: dict[str, Any]) -> None:
    """Pretty-print an evaluation report."""
    mode = report.get("mode", "classification")

    if mode == "regression":
        _print_regression_report(report)
    elif mode == "triple_barrier":
        _print_triple_barrier_report(report)
    else:
        _print_classification_report(report)


def _print_classification_report(report: dict[str, Any]) -> None:
    """Print classification evaluation report."""
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

    # Rolling IC trend
    rolling_6m = report.get("rolling_ic_6m", [])
    if rolling_6m:
        ics = [r["ic"] for r in rolling_6m if not np.isnan(r["ic"])]
        if ics:
            recent_mean = float(np.mean(ics[-3:])) if len(ics) >= 3 else float(np.mean(ics))
            all_mean = float(np.mean(ics))
            print("\n-- Rolling IC (6-month windows) --")
            print(f"  Recent 3 windows mean: {recent_mean:.4f}")
            print(f"  All windows mean:      {all_mean:.4f}")
            if recent_mean < all_mean * 0.5 and all_mean > 0:
                print("  [WARN] IC has declined > 50% recently -- possible decay.")

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


def _print_regression_report(report: dict[str, Any]) -> None:
    """Print regression evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (Regression Mode)")
    print("=" * 60)

    print(f"  Train RMSE:            {report['train_rmse']:.6f}")
    print(f"  Test RMSE:             {report['test_rmse']:.6f}")
    print(f"  Train R²:              {report['train_r2']:.4f}")
    print(f"  Test R²:               {report['test_r2']:.4f}")
    print(f"  Overfitting Gap (RMSE): {report['overfitting_gap_rmse']:.6f}")

    ic = report.get("ic", {})
    if ic and not np.isnan(ic.get("rank_ic", float("nan"))):
        print(f"  Rank IC:               {ic['rank_ic']:.3f}")
        print(f"  ICIR (rolling):        {ic['icir']:.3f}")
        print(f"  IC p-value:            {ic['ic_p_value']:.4f}")

        rank_ic = ic.get("rank_ic", 0)
        icir = ic.get("icir", 0)
        print()
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

    # Feature importance
    if report["feature_importance"]:
        print("\n-- Feature Importance --")
        for feat, imp in report["feature_importance"].items():
            print(f"  {feat:<25}: {imp:.3f}")


def _print_triple_barrier_report(report: dict[str, Any]) -> None:
    """Print triple barrier evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (Triple Barrier)")
    print("=" * 60)

    print(f"  Train Accuracy:        {report['train_accuracy']:.3f}")
    print(f"  Test Accuracy:         {report['test_accuracy']:.3f}")
    print(f"  Overfitting Gap:       {report['overfitting_gap']:.3f}")
    print(f"  Precision (+1):        {report['precision']:.3f}")
    print(f"  Recall (+1):           {report['recall']:.3f}")
    print(f"  F1 Score (+1):         {report['f1']:.3f}")

    ic = report.get("ic", {})
    if ic and not np.isnan(ic.get("rank_ic", float("nan"))):
        print(f"  Rank IC (profit prob): {ic['rank_ic']:.3f}")
        print(f"  ICIR (rolling):        {ic['icir']:.3f}")

    # Overfitting diagnosis
    gap = report["overfitting_gap"]
    if gap < 0.05:
        print("\n  [OK] Low overfitting — good generalisation.")
    elif gap < 0.10:
        print("\n  [WARN] Moderate overfitting — consider tuning regularisation.")
    else:
        print("\n  [FAIL] High overfitting.")

    # IC diagnosis
    if ic:
        rank_ic = ic.get("rank_ic", 0)
        icir = ic.get("icir", 0)
        if not np.isnan(rank_ic):
            if rank_ic > 0.05:
                print("  [OK] Rank IC > 0.05 — meaningful predictive signal.")
            elif rank_ic > 0:
                print("  [WARN] Rank IC > 0 but weak — marginal signal.")
            else:
                print("  [FAIL] Rank IC <= 0 — no directional signal.")
        if not np.isnan(icir):
            if icir > 1.0:
                print("  [OK] ICIR > 1.0 — stable signal.")
            elif icir > 0.3:
                print("  [WARN] ICIR > 0.3 — modest stability.")
            else:
                print("  [FAIL] ICIR < 0.3 — signal unstable.")

    # Label distribution
    ld = report.get("label_distribution", {})
    if ld:
        print("\n-- Label Distribution (Test) --")
        label_names = {"-1": "Stop Loss", "0": "Timeout", "1": "Profit Take"}
        for lbl in ["-1", "0", "1"]:
            cnt = ld.get(lbl, 0)
            total = sum(ld.values()) or 1
            print(f"  {label_names.get(lbl, lbl):<15}: {cnt} ({cnt/total:.1%})")

    # Confusion matrix
    cm = report.get("confusion_matrix_3class", {})
    if cm and cm.get("labels"):
        print("\n-- Confusion Matrix (3-class) --")
        labels = cm["labels"]
        matrix = cm["matrix"]
        header = "            " + "".join(f" Pred {x:<3}" for x in labels)
        print(header)
        for i, lbl in enumerate(labels):
            row = f"  True {lbl:<3}  "
            row += "".join(f" {int(matrix[i][j]):>7}" for j in range(len(labels)))
            print(row)

    # Feature importance
    if report["feature_importance"]:
        print("\n-- Feature Importance --")
        for feat, imp in report["feature_importance"].items():
            print(f"  {feat:<25}: {imp:.3f}")


def _print_model_comparison(results: list[dict[str, Any]], regression: bool = False) -> None:
    """Print model comparison results with ridge sanity check."""
    mode_label = "CV RMSE" if regression else "CV Mean"
    print(f"\n-- Model Comparison ({mode_label}) --")

    if regression:
        print(f"  {'Model':<20} {'CV RMSE':>10}  {'CV Std':>10}")
        for r in results:
            print(f"  {r['model_name']:<20} {r['cv_mean']:>10.4f}  {r['cv_std']:>10.4f}")
    else:
        print(f"  {'Model':<20} {'CV Mean':>8}  {'CV Std':>8}")
        for r in results:
            print(f"  {r['model_name']:<20} {r['cv_mean']:>8.3f}  {r['cv_std']:>8.3f}")

    # Ridge sanity baseline check
    ridge_result = next((r for r in results if r["model_name"] == "ridge"), None)
    if ridge_result is not None and not regression:
        # Ridge is primarily a regression model; in classification context,
        # compare LR (linear) vs XGBoost (non-linear)
        lr_result = next((r for r in results if r["model_name"] == "logistic"), None)
        xgb_result = next((r for r in results if r["model_name"] == "xgboost"), None)
        if lr_result is not None and xgb_result is not None:
            lr_cv = lr_result["cv_mean"]
            xgb_cv = xgb_result["cv_mean"]
            nonlinear_gain = xgb_cv - lr_cv
            print("\n  [Ridge/Sanity Check]")
            print(f"  LR (linear) CV accuracy:  {lr_cv:.3f}")
            print(f"  XGBoost CV accuracy:      {xgb_cv:.3f}")
            print(f"  Non-linear gain:          {nonlinear_gain:+.3f}")
            if nonlinear_gain > 0.03:
                print("  [OK] XGBoost provides meaningful non-linear gain over LR.")
            elif nonlinear_gain > 0:
                print("  [WARN] Marginal non-linear gain -- XGBoost may be fitting noise.")
            else:
                print("  [FAIL] XGBoost not better than linear model -- check features.")


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
    warnings_list: list[str] = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.8:
                warnings_list.append(
                    f"  {feature_cols[i]} <-> {feature_cols[j]}: r={r:.3f}"
                )
    return corr, warnings_list
