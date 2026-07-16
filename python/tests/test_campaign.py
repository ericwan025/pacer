"""Campaigns: targeting, budget accounting, log-normal heterogeneity, determinism."""

import numpy as np

from pacer.sim.campaign import Campaign, Targeting, generate_campaigns


def test_targeting_matches():
    t = Targeting("banner_pos", frozenset({"0", "1"}))
    assert t.matches({"banner_pos": 0})  # int coerced to str
    assert t.matches({"banner_pos": "1"})
    assert not t.matches({"banner_pos": "7"})
    assert not t.matches({})


def test_budget_accounting():
    c = Campaign(0, 10.0, 0.5, Targeting("banner_pos", frozenset({"0"})))
    assert c.remaining() == 10.0
    assert c.can_afford(6.0)
    c.charge(6.0)
    assert c.remaining() == 4.0
    assert not c.can_afford(5.0)
    c.charge(4.0)
    assert c.remaining() == 0.0
    assert not c.eligible({"banner_pos": "0"})  # out of budget


def _fvals():
    return {"banner_pos": ["0", "1", "2"], "site_category": ["a", "b", "c", "d"]}


def test_generate_count_and_positivity():
    camps = generate_campaigns(500, _fvals(), seed=0)
    assert len(camps) == 500
    assert all(c.daily_budget > 0 and c.value_per_click > 0 for c in camps)
    assert all(len(c.targeting.allowed) >= 1 for c in camps)


def test_heterogeneity_whales_and_tail():
    camps = generate_campaigns(500, _fvals(), seed=1)
    budgets = np.array([c.daily_budget for c in camps])
    # lognormal -> heavy right skew: max should dwarf the median
    assert budgets.max() > 5 * np.median(budgets)


def test_deterministic():
    a = generate_campaigns(50, _fvals(), seed=7)
    b = generate_campaigns(50, _fvals(), seed=7)
    assert [c.daily_budget for c in a] == [c.daily_budget for c in b]
    assert [c.targeting.allowed for c in a] == [c.targeting.allowed for c in b]
