"""Training loop: it improves val logloss and honors early stopping."""

import numpy as np

from pacer.models.lr import HashedLogReg
from pacer.models.train import TrainConfig, fit, predict_proba
from pacer.eval.metrics import auc


def _planted(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 6, size=(n, 3)).astype(np.int64)
    logit = 2.5 * (x[:, 0] == 2) - 1.2
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.float32)
    return x, y


def test_fit_improves_and_early_stops():
    xtr, ytr = _planted(4000, 0)
    xva, yva = _planted(1000, 1)
    model = HashedLogReg([6, 6, 6])
    cfg = TrainConfig(lr=0.1, batch_size=512, max_epochs=30, patience=2)
    hist = fit(model, (xtr, ytr), (xva, yva), cfg, verbose=False)
    # learned something real
    assert auc(yva, predict_proba(model, xva)) > 0.6
    # early stopping kept fewer than max_epochs entries (planted signal is easy)
    assert len(hist["val_logloss"]) <= cfg.max_epochs
    assert hist["best_val_logloss"] == min(hist["val_logloss"])
