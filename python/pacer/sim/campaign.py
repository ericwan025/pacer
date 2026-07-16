"""Campaigns.

A campaign has a daily budget, a value_per_click (what a click is worth to the
advertiser), a simple targeting predicate, and mutable spend state.

Budgets and values are sampled log-normal so we get realistic heterogeneity: a
few whales and a long tail of small spenders. RNG is seeded; count configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Fields campaigns are allowed to target on. Kept deliberately simple.
TARGETABLE_FIELDS = ("banner_pos", "site_category")


@dataclass(frozen=True)
class Targeting:
    field: str
    allowed: frozenset[str]

    def matches(self, features: dict) -> bool:
        return str(features.get(self.field)) in self.allowed


@dataclass
class Campaign:
    id: int
    daily_budget: float
    value_per_click: float
    targeting: Targeting
    spend: float = 0.0
    # pacing state the controller reads/writes
    pacing_multiplier: float = 1.0

    def remaining(self) -> float:
        return max(0.0, self.daily_budget - self.spend)

    def can_afford(self, cost: float) -> bool:
        return self.spend + cost <= self.daily_budget

    def charge(self, cost: float) -> None:
        self.spend += cost

    def eligible(self, features: dict) -> bool:
        return self.remaining() > 0 and self.targeting.matches(features)


def generate_campaigns(
    n: int,
    feature_values: dict[str, list[str]],
    seed: int = 0,
    budget_mu: float = 5.0,
    budget_sigma: float = 1.0,
    value_mu: float = -0.7,
    value_sigma: float = 0.6,
) -> list[Campaign]:
    """Sample n campaigns.

    budget ~ lognormal(budget_mu, budget_sigma)  (median ~ e^5 ≈ $148/day)
    value  ~ lognormal(value_mu, value_sigma)    (median ~ $0.50/click)
    Targeting: pick one targetable field and a random nonempty subset of its
    observed values.
    """
    rng = np.random.default_rng(seed)
    budgets = rng.lognormal(budget_mu, budget_sigma, n)
    values = rng.lognormal(value_mu, value_sigma, n)

    fields = [f for f in TARGETABLE_FIELDS if feature_values.get(f)]
    if not fields:
        raise ValueError("no targetable fields have observed values")

    campaigns: list[Campaign] = []
    for i in range(n):
        fld = fields[rng.integers(0, len(fields))]
        vals = feature_values[fld]
        k = 1 + int(rng.integers(0, max(1, len(vals))))
        chosen = rng.choice(np.array(vals, dtype=object), size=min(k, len(vals)), replace=False)
        targeting = Targeting(field=fld, allowed=frozenset(map(str, chosen)))
        campaigns.append(
            Campaign(
                id=i,
                daily_budget=float(budgets[i]),
                value_per_click=float(values[i]),
                targeting=targeting,
            )
        )
    return campaigns
