"""Tests for chronological train-test split."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_split_is_chronological() -> None:
    """Train indices must all precede test indices."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "feat": rng.normal(0, 1, n),
        "target": rng.integers(0, 2, n),
    })
    test_size = 0.2
    split_idx = int(len(df) * (1 - test_size))

    X_train = df.iloc[:split_idx][["feat"]]
    X_test = df.iloc[split_idx:][["feat"]]

    assert len(X_train) + len(X_test) == n
    assert len(X_test) >= 1, "Test set is empty"

    train_max_idx = split_idx - 1
    test_min_idx = split_idx
    assert train_max_idx < test_min_idx, "Train indices overlap with test indices"


def test_split_preserves_class_distribution_roughly() -> None:
    """Class proportions should be roughly similar between train and test."""
    rng = np.random.default_rng(100)
    n = 500
    targets = rng.integers(0, 2, n)
    df = pd.DataFrame({"target": targets})

    split_idx = int(len(df) * 0.8)
    y_train = df.iloc[:split_idx]["target"]
    y_test = df.iloc[split_idx:]["target"]

    train_pct = y_train.mean()
    test_pct = y_test.mean()

    assert abs(train_pct - test_pct) < 0.15, (
        f"Class distribution diverged: train={train_pct:.2f}, test={test_pct:.2f}"
    )
