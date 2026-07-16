"""Size campaign budgets to inventory so pacing is a real problem.

An advertiser budget that exceeds everything a campaign could ever win is not a
pacing scenario — the controller can only spend what it wins, so it behaves like
greedy no matter what. To make the pacing comparison meaningful we first run a
GREEDY pass (no pacing, unlimited budget) to measure each campaign's *achievable*
spend in the fully-competitive market, then set:

    daily_budget = target_utilization * achievable_spend

with target_utilization < 1 so the controller must actually pace down to hit its
target. This is experiment design, not tuning-to-win: we document it, and the
same budgets are used for every strategy.
"""

from __future__ import annotations


from pacer.sim.campaign import Campaign
from pacer.sim.engine import Engine, EngineConfig


def _clone_unbounded(campaigns: list[Campaign]) -> list[Campaign]:
    return [
        Campaign(
            id=c.id,
            daily_budget=float("inf"),
            value_per_click=c.value_per_click,
            targeting=c.targeting,
        )
        for c in campaigns
    ]


def achievable_spend(
    campaigns: list[Campaign],
    features,
    labels,
    pctrs,
    replay,
    reserve: float = 0.001,
) -> dict[int, float]:
    """Greedy, unbounded-budget run. Returns cid -> spend it could win."""
    greedy = _clone_unbounded(campaigns)
    eng = Engine(greedy, features, labels, pctrs, EngineConfig(reserve=reserve))
    eng.run(replay)
    return {c.id: c.spend for c in greedy}


def size_budgets(
    campaigns: list[Campaign],
    achievable: dict[int, float],
    target_utilization: float = 0.6,
    min_budget: float = 1e-6,
) -> None:
    """Mutate campaigns' daily_budget in place to target_utilization * achievable.

    Campaigns that could win essentially nothing get a tiny floor budget so they
    stay valid but simply don't participate meaningfully.
    """
    for c in campaigns:
        c.daily_budget = max(min_budget, target_utilization * achievable.get(c.id, 0.0))
        c.spend = 0.0
        c.pacing_multiplier = 1.0
