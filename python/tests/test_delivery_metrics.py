"""Delivery metrics + settling time."""

import numpy as np

from pacer.eval.delivery_metrics import delivery_report, settling_time
from pacer.sim.engine import RunStats


def test_delivery_basic():
    s = RunStats()
    s.total_clicks = 20
    s.total_spend = 100.0
    s.wins_by_daypart = np.zeros(24, dtype=np.int64)
    s.wins_by_daypart[9] = 30
    s.wins_by_daypart[21] = 10
    rep = delivery_report(s)
    assert rep.clicks_per_dollar == 0.2
    assert rep.effective_cpc == 5.0
    assert abs(rep.impression_share_by_daypart.sum() - 1.0) < 1e-9
    assert abs(rep.impression_share_by_daypart[9] - 0.75) < 1e-9


def test_delivery_zero_spend():
    rep = delivery_report(RunStats())
    assert rep.clicks_per_dollar == 0.0
    assert rep.effective_cpc == float("inf")


def test_settling_time_returns_after_disturbance():
    # multiplier spikes to 3.0 during/after burst, decays back to 1.0
    ts = np.arange(0, 200, 10.0)
    ms = np.ones_like(ts)
    burst_end = 50.0
    # out of band (steady 1.0, band 0.1 -> tol 0.1) until t=110
    for i, t in enumerate(ts):
        if burst_end <= t <= 100:
            ms[i] = 3.0
        elif 100 < t <= 110:
            ms[i] = 1.2  # still outside 10% band
    st = settling_time(list(zip(ts, ms)), burst_end_t=burst_end, band=0.1, steady=1.0)
    assert st == 120.0 - burst_end  # first in-band sample after last bad (t=110)


def test_settling_time_never_settles():
    ts = np.arange(0, 100, 10.0)
    ms = np.full_like(ts, 3.0)
    st = settling_time(list(zip(ts, ms)), burst_end_t=0.0, band=0.1, steady=1.0)
    assert st is None
