"""Hashed logistic-regression baseline.

Each categorical field index looks up a single scalar weight (an embedding of
dim 1); the row logit is the sum of those weights plus a global bias. That is
exactly logistic regression over the multi-hot of all hashed/vocab fields, done
sparsely so we never materialize a 2^20-wide one-hot vector.

This is the floor. On Avazu expect AUC ~0.74-0.76. AUC ~0.5 means it learned
nothing; ~0.99 means leakage. Either should be reported, not worked around.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HashedLogReg(nn.Module):
    def __init__(self, cardinalities: list[int]):
        super().__init__()
        self.n_fields = len(cardinalities)
        # one scalar weight per (field, value). padding_idx=0 keeps OOV at 0 init
        # but it still trains — OOV is a real bucket, not padding, so no padding_idx.
        self.weights = nn.ModuleList(
            [nn.Embedding(card, 1) for card in cardinalities]
        )
        for emb in self.weights:
            nn.init.zeros_(emb.weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_fields] int64 -> logit [batch]
        total = self.bias
        for i, emb in enumerate(self.weights):
            total = total + emb(x[:, i]).squeeze(-1)
        return total  # raw logit; caller applies sigmoid / BCEWithLogits
