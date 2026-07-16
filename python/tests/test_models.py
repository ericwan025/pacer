"""Smoke tests for the model modules on tiny synthetic data.

These assert the architecture can learn a trivial planted signal — they do NOT
produce any benchmark number. Real AUC/logloss come from training on the Avazu
splits via the report script.
"""

import numpy as np
import torch

from pacer.models.deepfm import DeepFM
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


def test_deepfm_forward_shape_and_order2_trick():
    model = DeepFM([5, 5, 5], embed_dim=8, mlp_dims=(16, 16))
    x = torch.randint(0, 5, (7, 3))
    out = model(x)
    assert out.shape == (7,)

    # verify the sum-square trick equals the explicit pairwise sum
    e = model._field_embeds(x)  # [B, F, K]
    B, F, K = e.shape
    explicit = torch.zeros(B)
    for i in range(F):
        for j in range(i + 1, F):
            explicit += (e[:, i, :] * e[:, j, :]).sum(dim=1)
    sum_sq = e.sum(dim=1).pow(2)
    sq_sum = e.pow(2).sum(dim=1)
    trick = 0.5 * (sum_sq - sq_sum).sum(dim=1)
    assert torch.allclose(explicit, trick, atol=1e-5)


def test_deepfm_learns_planted_signal():
    x, y = _planted_data(3000)
    model = DeepFM([5, 5, 5], embed_dim=8, mlp_dims=(32, 32), dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(150):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).numpy()
    mask = (x[:, 0] == 1).numpy()
    assert probs[mask].mean() > probs[~mask].mean() + 0.2
