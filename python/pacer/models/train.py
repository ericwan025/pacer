"""Shared training loop for LR and DeepFM.

Mini-batch Adam on BCEWithLogits, early stopping on validation logloss with a
configurable patience. Returns the best (lowest val logloss) model state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from pacer.eval.metrics import auc, logloss


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 4096
    max_epochs: int = 20
    patience: int = 2
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 0


def _iter_batches(x, y, bs, shuffle, rng):
    n = len(y)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, bs):
        sel = idx[start : start + bs]
        yield x[sel], y[sel]


@torch.no_grad()
def predict_proba(model: nn.Module, x: np.ndarray, bs: int = 8192, device="cpu") -> np.ndarray:
    model.eval()
    out = []
    xt = torch.as_tensor(x, dtype=torch.long, device=device)
    for start in range(0, len(x), bs):
        logit = model(xt[start : start + bs])
        out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)


def fit(
    model: nn.Module,
    train_xy: tuple[np.ndarray, np.ndarray],
    val_xy: tuple[np.ndarray, np.ndarray],
    cfg: TrainConfig,
    verbose: bool = True,
) -> dict:
    """Train with early stopping. Returns history + loads best state into model."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = cfg.device
    model.to(device)

    xtr, ytr = train_xy
    xva, yva = val_xy
    xtr_t = torch.as_tensor(xtr, dtype=torch.long, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_ll = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    history = {"val_logloss": [], "val_auc": []}

    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in _iter_batches(
            np.arange(len(ytr)), ytr, cfg.batch_size, True, rng
        ):
            # xb here are indices into the pre-moved tensors
            idx = torch.as_tensor(xb, dtype=torch.long, device=device)
            opt.zero_grad()
            logit = model(xtr_t[idx])
            loss = loss_fn(logit, ytr_t[idx])
            loss.backward()
            opt.step()

        p_va = predict_proba(model, xva, device=device)
        ll = logloss(yva, p_va)
        a = auc(yva, p_va)
        history["val_logloss"].append(ll)
        history["val_auc"].append(a)
        if verbose:
            print(f"epoch {epoch}: val_logloss={ll:.5f} val_auc={a:.5f}")

        if ll < best_ll - 1e-6:
            best_ll = ll
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    history["best_val_logloss"] = best_ll
    return history
