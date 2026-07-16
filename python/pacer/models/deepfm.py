"""DeepFM (Guo et al. 2017) in PyTorch.

logit = FM_order1 + FM_order2 + MLP_out

* Order-1: a scalar weight per (field, value)  -> linear term.
* Order-2: pairwise field-embedding interactions via the sum-square trick,
    0.5 * sum_k [ (sum_i v_ik)^2 - sum_i v_ik^2 ]
  which is O(fields * dim), never the O(fields^2) explicit double loop.
* Deep: concat all field embeddings -> MLP [400,400,400], ReLU, dropout.

The order-2 term and the deep tower SHARE the same embedding table, which is the
defining feature of DeepFM versus a plain Wide&Deep.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        embed_dim: int = 16,
        mlp_dims: tuple[int, ...] = (400, 400, 400),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_fields = len(cardinalities)
        self.embed_dim = embed_dim

        # shared embeddings (order-2 + deep) and order-1 scalar weights
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, embed_dim) for card in cardinalities]
        )
        self.linear = nn.ModuleList([nn.Embedding(card, 1) for card in cardinalities])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.01)
        for lin in self.linear:
            nn.init.zeros_(lin.weight)
        self.bias = nn.Parameter(torch.zeros(1))

        # deep tower
        layers: list[nn.Module] = []
        in_dim = self.n_fields * embed_dim
        for h in mlp_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def _field_embeds(self, x: torch.Tensor) -> torch.Tensor:
        # -> [batch, n_fields, embed_dim]
        return torch.stack([emb(x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # order-1 linear
        order1 = self.bias + sum(
            lin(x[:, i]).squeeze(-1) for i, lin in enumerate(self.linear)
        )

        e = self._field_embeds(x)  # [B, F, K]
        # order-2 sum-square trick
        sum_sq = e.sum(dim=1).pow(2)  # [B, K]
        sq_sum = e.pow(2).sum(dim=1)  # [B, K]
        order2 = 0.5 * (sum_sq - sq_sum).sum(dim=1)  # [B]

        deep = self.mlp(e.flatten(start_dim=1)).squeeze(-1)  # [B]

        return order1 + order2 + deep  # raw logit
