"""Metric correctness against known values and against sklearn where available."""

import numpy as np

from pacer.eval.metrics import auc, ece, logloss, reliability_curve


def test_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    p_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(auc(y, p_perfect) - 1.0) < 1e-9
    p_inverted = np.array([0.9, 0.8, 0.2, 0.1])
    assert abs(auc(y, p_inverted) - 0.0) < 1e-9


def test_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    assert abs(auc(y, p) - roc_auc_score(y, p)) < 1e-9


def test_auc_tie_handling():
    from sklearn.metrics import roc_auc_score

    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.7, 0.3, 0.3])
    assert abs(auc(y, p) - roc_auc_score(y, p)) < 1e-9


def test_logloss_matches_sklearn():
    from sklearn.metrics import log_loss

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = rng.random(300) * 0.98 + 0.01
    assert abs(logloss(y, p) - log_loss(y, p)) < 1e-6


def test_ece_zero_for_perfect_calibration():
    # construct data where predicted prob equals empirical rate exactly
    rng = np.random.default_rng(2)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(float)
    # well-calibrated by construction -> small ECE
    assert ece(y, p, n_bins=20) < 0.02


def test_ece_large_for_miscalibration():
    rng = np.random.default_rng(3)
    p = np.full(10000, 0.9)  # always predicts 0.9
    y = (rng.random(10000) < 0.1).astype(float)  # true rate 0.1
    assert ece(y, p, n_bins=20) > 0.7


def test_reliability_curve_shapes():
    rng = np.random.default_rng(4)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(float)
    mp, mt, w = reliability_curve(y, p, n_bins=10)
    assert len(mp) == len(mt) == len(w)
    assert abs(w.sum() - 1.0) < 1e-9
