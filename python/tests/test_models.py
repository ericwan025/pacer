"""Smoke tests for the model modules on tiny synthetic data.

These assert the architecture can learn a trivial planted signal — they do NOT
produce any benchmark number. Real AUC/logloss come from training on the Avazu
splits via the report script.
"""

import numpy as np
import torch

from pacer.models.lr import HashedLogReg


def _planted_data(n=2000, seed=0):
    """Field 0 value 1 => high click prob; everything else noise."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 5, size=(n, 3)).astype(np.int64)
    logit = 3.0 * (x[:, 0] == 1) - 1.5
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def test_logreg_learns_planted_signal():
    x, y = _planted_data()
    model = HashedLogReg([5, 5, 5])
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).numpy()
    # AUC-ish check: mean prob for planted-positive rows > for others
    mask = (x[:, 0] == 1).numpy()
    assert probs[mask].mean() > probs[~mask].mean() + 0.2
