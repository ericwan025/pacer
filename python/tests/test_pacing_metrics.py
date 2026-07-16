"""Pacing metrics: perfect tracking -> ~0 error, early exhaustion detection."""

import numpy as np

from pacer.eval.pacing_metrics import (
    budget_utilization,
    early_exhaustion_fraction,
    pacing_report,
)
from pacer.sim.campaign import Campaign, Targeting
from pacer.sim.target import UniformTarget

HORIZON = 24 * 3600.0


def _campaign(cid, budget):
    return Campaign(cid, budget, 1.0, Targeting("banner_pos", frozenset({"0"})))


def _perfect_trace(budget, curve, n=100):
    ts = np.linspace(0, HORIZON, n)
    return [(float(t), curve.spend_target(budget, t)) for t in ts]


def test_perfect_tracking_zero_error():
    curve = UniformTarget(HORIZON)
    camps = [_campaign(0, 100.0)]
    camps[0].spend = 100.0
    trace = {0: _perfect_trace(100.0, curve)}
    rep = pacing_report(trace, camps, curve)
    assert rep.mean_abs_pacing_error_pct < 1e-9
    assert rep.l2_spend_deviation < 1e-9
    assert abs(rep.utilization_mean - 1.0) < 1e-9


def test_pacing_error_scales_with_offset():
    curve = UniformTarget(HORIZON)
    camps = [_campaign(0, 100.0)]
    # spend curve constantly 10% of budget above target
    ts = np.linspace(0, HORIZON, 50)
    trace = {0: [(float(t), curve.spend_target(100.0, t) + 10.0) for t in ts]}
    rep = pacing_report(trace, camps, curve)
    assert abs(rep.mean_abs_pacing_error_pct - 0.10) < 1e-9


def test_early_exhaustion_detection():
    camps = [_campaign(0, 100.0), _campaign(1, 100.0)]
    # camp 0 blows budget at hour 5; camp 1 paces to hour 23
    trace = {
        0: [(5 * 3600.0, 100.0), (23 * 3600.0, 100.0)],
        1: [(23 * 3600.0, 100.0)],
    }
    camps[0].spend = 100.0
    camps[1].spend = 100.0
    assert early_exhaustion_fraction(trace, camps) == 0.5


def test_utilization():
    camps = [_campaign(0, 100.0), _campaign(1, 200.0)]
    camps[0].spend = 50.0
    camps[1].spend = 200.0
    util = budget_utilization(camps)
    assert np.allclose(util, [0.5, 1.0])
