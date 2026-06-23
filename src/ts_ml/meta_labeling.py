"""Meta-Labeling: secondary model that decides whether to act on primary model predictions.

From Advances in Financial Machine Learning (Lopez de Prado):
  1. Train a primary model to predict the target (e.g. triple barrier labels)
  2. Train a secondary model to predict whether the primary model's predictions
     are correct, using the primary model's confidence and original features.

The output is a meta-probability: "how likely is this primary-model prediction to be correct?"
This replaces the simple prob_threshold filter with a learned filter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit


def build_meta_labels(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Build labels for the meta-model.

    meta_label = 1 if primary model was correct, 0 otherwise.

    Parameters
    ----------
    y_true : pd.Series
        True labels (0, 1, 2 for triple barrier internally).
    y_pred : np.ndarray
        Primary model's predicted class indices.

    Returns
    -------
    np.ndarray
        Binary labels: 1 = primary correct, 0 = primary wrong.
    """
    return (y_true.values == y_pred).astype(int)


def build_meta_features(
    X: pd.DataFrame,
    proba: np.ndarray,
) -> pd.DataFrame:
    """Build feature matrix for the meta-model.

    Combines original features with primary model's predicted probabilities
    and derived confidence signals.

    Parameters
    ----------
    X : pd.DataFrame
        Original feature matrix.
    proba : np.ndarray
        Primary model's predicted probabilities, shape (n_samples, n_classes).

    Returns
    -------
    pd.DataFrame
        Feature matrix with added meta-features.
    """
    meta_X = X.copy()

    # Add per-class probabilities
    n_classes = proba.shape[1]
    for i in range(n_classes):
        meta_X[f"meta_proba_class_{i}"] = proba[:, i]

    # Add confidence: max probability as a signal of model certainty
    meta_X["meta_confidence"] = proba.max(axis=1)

    # Add entropy: how uncertain the model is
    eps = 1e-10
    entropy = -np.sum(proba * np.log(proba + eps), axis=1)
    meta_X["meta_entropy"] = entropy

    # For 3-class, add profit/stop probability ratio
    if n_classes == 3:
        # class 2 = profit_take, class 0 = stop_loss
        profit_prob = proba[:, 2]
        stop_prob = proba[:, 0]
        meta_X["meta_profit_stop_ratio"] = np.where(
            stop_prob > 0, profit_prob / (stop_prob + eps), profit_prob,
        )

    return meta_X


def train_meta_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    primary_model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    meta_model_class: Any = LogisticRegression,
    meta_model_params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train a meta-labeling model.

    Steps:
      1. Use primary model to predict on validation set
      2. Build meta-labels: 1 if primary was correct, 0 otherwise
      3. Build meta-features: original features + primary model probabilities
      4. Train a lightweight classifier on meta-features → meta-labels

    Parameters
    ----------
    X_train, y_train : training data (used to fit primary model)
    primary_model : already-trained primary model
    X_val, y_val : validation data for meta-model training
    meta_model_class : class
        Classifier for the meta-model (default LogisticRegression for simplicity).
    meta_model_params : dict
        Constructor params for meta_model_class.

    Returns
    -------
    meta_model : trained classifier
    info : dict with evaluation metrics
    """
    if meta_model_params is None:
        meta_model_params = {
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": 42,
        }

    # Primary model predictions on validation set
    primary_proba = primary_model.predict_proba(X_val)
    primary_pred = np.argmax(primary_proba, axis=1)

    # Meta labels: was primary correct?
    meta_y_raw = build_meta_labels(y_val, primary_pred)
    meta_y = pd.Series(meta_y_raw, index=y_val.index)

    # Meta features: original + primary probabilities
    meta_X = build_meta_features(X_val, primary_proba)

    # Check class balance
    n_pos = meta_y.sum()
    n_neg = len(meta_y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, {
            "error": "Meta labels are all one class — cannot train",
            "primary_accuracy": float(accuracy_score(y_val, primary_pred)),
        }

    # Train meta-model with time-series CV
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores: list[float] = []
    for train_idx, test_idx in tscv.split(meta_X):
        X_fold_train = meta_X.iloc[train_idx]  # type: ignore[union-attr]
        y_fold_train = meta_y.iloc[train_idx]  # type: ignore[union-attr]
        X_fold_test = meta_X.iloc[test_idx]  # type: ignore[union-attr]
        y_fold_test = meta_y.iloc[test_idx]  # type: ignore[union-attr]

        meta_m = meta_model_class(**meta_model_params)
        meta_m.fit(X_fold_train, y_fold_train)
        pred = meta_m.predict(X_fold_test)
        cv_scores.append(float(accuracy_score(y_fold_test, pred)))

    # Final fit
    meta_model = meta_model_class(**meta_model_params)
    meta_model.fit(meta_X, meta_y)

    info = {
        "primary_accuracy": float(accuracy_score(y_val, primary_pred)),
        "meta_cv_mean": float(np.mean(cv_scores)),
        "meta_cv_std": float(np.std(cv_scores, ddof=1)),
        "meta_positive_rate": float(n_pos / len(meta_y)),
        "n_meta_samples": len(meta_y),
    }

    return meta_model, info


def apply_meta_filter(
    X: pd.DataFrame,
    primary_model: Any,
    meta_model: Any,
    meta_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply meta-model filter to primary model predictions.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix to predict on.
    primary_model : model
        Trained primary model.
    meta_model : model
        Trained meta-labeling model.
    meta_threshold : float
        Minimum meta-probability to accept a primary prediction.

    Returns
    -------
    primary_pred : np.ndarray
        Primary model class predictions.
    primary_proba : np.ndarray
        Primary model probabilities.
    meta_accept : np.ndarray (bool)
        Whether the meta-model accepts each primary prediction.
    """
    primary_proba = primary_model.predict_proba(X)
    primary_pred = np.argmax(primary_proba, axis=1)

    meta_X = build_meta_features(X, primary_proba)
    meta_proba = meta_model.predict_proba(meta_X)[:, 1]  # prob of primary being correct
    meta_accept = meta_proba >= meta_threshold

    return primary_pred, primary_proba, meta_accept
