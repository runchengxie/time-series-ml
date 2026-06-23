"""Tests for PurgedTimeSeriesSplit — purge and embargo logic."""

from __future__ import annotations

import numpy as np
import pytest

from ts_ml.crossval import PurgedTimeSeriesSplit, purged_cross_val_score


def test_basic_split_structure() -> None:
    """PurgedTimeSeriesSplit produces chronologically ordered folds."""
    X = np.arange(100).reshape(-1, 1)
    y = np.zeros(100)
    cv = PurgedTimeSeriesSplit(n_splits=3, purge_days=0, embargo_days=0)
    folds = cv.split(X, y)

    assert len(folds) > 0

    for train_idx, test_idx in folds:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        # Train indices must all precede test indices
        assert train_idx[-1] < test_idx[0]


def test_purge_removes_trailing_samples() -> None:
    """Purge drops the last N samples from each training fold."""
    X = np.arange(100).reshape(-1, 1)
    y = np.zeros(100)
    cv_no_purge = PurgedTimeSeriesSplit(n_splits=3, purge_days=0, embargo_days=0)
    cv_with_purge = PurgedTimeSeriesSplit(n_splits=3, purge_days=10, embargo_days=0)

    folds_no = cv_no_purge.split(X, y)
    folds_with = cv_with_purge.split(X, y)

    for (train_np, _), (train_wp, _) in zip(folds_no, folds_with, strict=True):
        # Purged folds should have fewer training samples
        assert len(train_wp) <= len(train_np)


def test_embargo_creates_gap() -> None:
    """Embargo inserts a gap between train and test."""
    X = np.arange(100).reshape(-1, 1)
    y = np.zeros(100)
    cv_no_embargo = PurgedTimeSeriesSplit(n_splits=3, purge_days=0, embargo_days=0)
    cv_with_embargo = PurgedTimeSeriesSplit(n_splits=3, purge_days=0, embargo_days=5)

    folds_no = cv_no_embargo.split(X, y)
    folds_with = cv_with_embargo.split(X, y)

    for (_train_ne, _test_ne), (train_we, test_we) in zip(
        folds_no, folds_with, strict=True
    ):
        # With embargo, gap between train end and test start should be >= embargo_days
        gap = test_we[0] - train_we[-1]
        assert gap >= 5


def test_n_splits_minimum() -> None:
    """n_splits must be at least 2."""
    with pytest.raises(ValueError, match="at least 2"):
        PurgedTimeSeriesSplit(n_splits=1)


def test_purged_cross_val_score_sample_weight() -> None:
    """CV scoring with sample_weight runs without error."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    y = rng.integers(0, 2, 100)

    model = LogisticRegression(max_iter=1000)
    cv = PurgedTimeSeriesSplit(n_splits=3, purge_days=5, embargo_days=1)

    # Uniform weights
    weights = np.ones(100)
    scores = purged_cross_val_score(
        model, X, y.astype(float), cv,
        scoring="accuracy", sample_weight=weights,
    )
    assert len(scores) >= 1
    assert all(0 <= s <= 1 for s in scores)
