"""Evaluation metrics for the CTR model.

ECE uses EQUAL-MASS bins (equal number of samples per bin), not equal-width.
pCTR is heavily skewed toward zero, so equal-width bins would put almost every
sample in the first bin and measure nothing.
"""

from __future__ import annotations

import numpy as np


def logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC via the rank (Mann-Whitney U) formulation, tie-aware."""
    y = np.asarray(y)
    p = np.asarray(p)
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks over ties so tied scores don't bias AUC
    _assign_tie_ranks(p, order, ranks)
    sum_pos = np.sum(ranks[y == 1])
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _assign_tie_ranks(p, order, ranks):
    sp = p[order]
    i = 0
    n = len(sp)
    while i < n:
        j = i
        while j + 1 < n and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


def equal_mass_bins(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Return bin edges giving ~equal sample count per bin."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(p, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.unique(edges)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> float:
    """Expected Calibration Error with equal-mass bins."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = equal_mass_bins(p, n_bins)
    bin_idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    total = 0.0
    n = len(p)
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        total += (mask.sum() / n) * abs(conf - acc)
    return float(total)


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 20):
    """Return (mean_pred, mean_true, weight) per equal-mass bin, for plotting."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = equal_mass_bins(p, n_bins)
    bin_idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    mean_pred, mean_true, weight = [], [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        mean_pred.append(p[mask].mean())
        mean_true.append(y[mask].mean())
        weight.append(mask.mean())
    return np.array(mean_pred), np.array(mean_true), np.array(weight)
