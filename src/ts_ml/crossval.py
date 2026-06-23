"""Purged time-series cross-validation with embargo.

Standard TimeSeriesSplit does not account for:
- Purge: training samples whose label window overlaps with the test period.
- Embargo: a gap between train and test to prevent serial-correlation leakage.

Supports sample_weight (for exp_decay weighting) and regression scoring.
"""

from __future__ import annotations

import numpy as np


class PurgedTimeSeriesSplit:
    """Time-series cross-validator with purge and embargo.

    Parameters
    ----------
    n_splits : int
        Number of folds (default 5).
    purge_days : int
        Number of samples to drop from the END of each training set.
        For rolling-window features of length W, set purge_days >= W.
    embargo_days : int
        Number of samples to skip between train and test sets.
        For daily data with next-day labels, 1 day is usually sufficient.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 0,
        embargo_days: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices with purge and embargo applied."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        folds: list[tuple[np.ndarray, np.ndarray]] = []
        test_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = n_samples - (self.n_splits - i - 1) * test_size

            train_end = test_start - self.embargo_days
            train_end = max(0, train_end)
            train_end_purged = train_end - self.purge_days
            train_end_purged = max(0, train_end_purged)

            if train_end_purged <= 0:
                continue

            train_idx = indices[:train_end_purged]
            test_idx = indices[test_start:test_end]

            folds.append((train_idx, test_idx))

        return folds

    def get_n_splits(self) -> int:
        return self.n_splits


def purged_cross_val_score(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: PurgedTimeSeriesSplit,
    scoring: str = "accuracy",
    sample_weight: np.ndarray | None = None,
) -> list[float]:
    """Compute cross-validation scores with purge/embargo.

    Parameters
    ----------
    sample_weight : np.ndarray or None
        Sample weights for training. Only the training-fold subset is passed
        to model.fit(). Default None (uniform weights).
    """
    from sklearn.metrics import (
        accuracy_score,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    _scorer_map = {
        "accuracy": accuracy_score,
        "roc_auc": roc_auc_score,
        "neg_mean_squared_error": lambda yt, yp: -mean_squared_error(yt, yp),
        "r2": r2_score,
    }
    scorer = _scorer_map[scoring]

    scores: list[float] = []

    for train_idx, test_idx in cv.split(X, y):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_test_fold = X[test_idx]
        y_test_fold = y[test_idx]

        model = estimator.__class__(**estimator.get_params())

        # Pass sample_weight for this fold
        fold_sample_weight = None
        if sample_weight is not None:
            fold_sample_weight = sample_weight[train_idx]

        if fold_sample_weight is not None:
            try:
                model.fit(X_train_fold, y_train_fold, sample_weight=fold_sample_weight)
            except TypeError:
                model.fit(X_train_fold, y_train_fold)
        else:
            model.fit(X_train_fold, y_train_fold)

        if scoring == "roc_auc":
            prob = model.predict_proba(X_test_fold)[:, 1]
            score = scorer(y_test_fold, prob)
        else:
            pred = model.predict(X_test_fold)
            score = scorer(y_test_fold, pred)

        scores.append(float(score))

    return scores
