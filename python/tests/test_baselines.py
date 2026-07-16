"""Baselines smoke test on synthetic data: every strategy runs, respects budget,
and pacing strategies exhaust budget less abruptly than greedy."""

import numpy as np

from pacer.eval.baselines import STRATEGIES, PacingRunner
from pacer.eval.pacing_metrics import pacing_report
from pacer.sim.campaign import Campaign, Targeting
from pacer.sim.engine import Engine, EngineConfig
from pacer.sim.pid import PIDConfig
from pacer.sim.target import TrafficAwareTarget, UniformTarget


def _data(n=6000, hours_n=24, seed=0):
    rng = np.random.default_rng(seed)
    # diurnal volume: reuse a fixed profile scaled to per-hour counts
    profile = np.array(
        [2, 1, 1, 1, 2, 4, 7, 10, 12, 13, 13, 12, 11, 11, 12, 13, 14, 15, 16, 14, 11, 8, 5, 3],
        float,
    )
    counts = np.maximum(1, (profile / profile.sum() * n).astype(int))
    hours = np.concatenate([[14100100 + h] * counts[h] for h in range(hours_n)])
    n = len(hours)
    features = [{"banner_pos": "0", "site_category": "a"} for _ in range(n)]
    pctrs = rng.beta(2, 8, n)
    labels = (rng.random(n) < pctrs).astype(np.int8)
    return hours, features, labels, pctrs, counts


def _run(strategy, seed=0):
    hours, features, labels, pctrs, counts = _data(seed=seed)
    from pacer.sim.traffic import TrafficReplay

    replay = TrafficReplay(hours, seed=seed)
    curve = (
        TrafficAwareTarget(counts)
        if strategy.traffic_aware
        else UniformTarget(replay.duration_seconds())
    )
    camps = [
        Campaign(i, 20.0, 1.0, Targeting("banner_pos", frozenset({"0"})))
        for i in range(5)
    ]
    gains = PIDConfig(kp=1.0, ki=0.1, kd=0.0, dt=10.0)
    runner = PacingRunner(strategy, camps, curve, gains, seed=seed)
    eng = Engine(
        camps, features, labels, pctrs, EngineConfig(reserve=0.001, control_interval_s=10.0),
        control_hook=runner.control_hook,
        throttle_hook=runner.throttle_hook,
        bid_hook=runner.bid_hook,
    )
    stats = eng.run(replay)
    return stats, camps, curve


def test_all_strategies_run_and_respect_budget():
    for strat in STRATEGIES:
        stats, camps, _ = _run(strat)
        assert stats.impressions > 0
        for c in camps:
            assert c.spend <= c.daily_budget + 1e-9


def test_pid_tracks_target_uncontested():
    """The strongest controller-correctness check in the full engine: a single
    campaign with no competition must track the traffic-aware target almost
    exactly. Multi-campaign deviation is a competitive-market effect, not a bug."""
    from pacer.eval.budget_sizing import achievable_spend, size_budgets
    from pacer.sim.traffic import TrafficReplay

    hours, features, labels, pctrs, counts = _data(seed=0)
    replay = TrafficReplay(hours, seed=0)
    curve = TrafficAwareTarget(counts)
    c = Campaign(0, 1e9, 0.5, Targeting("banner_pos", frozenset({"0"})))
    ach = achievable_spend([c], features, labels, pctrs, replay)
    size_budgets([c], ach, target_utilization=0.6)

    strat = next(s for s in STRATEGIES if s.name == "traffic_pid_bidshade")
    gains = PIDConfig(kp=4.0, ki=0.4, kd=0.0, dt=10.0)
    runner = PacingRunner(strat, [c], curve, gains, seed=0)
    eng = Engine(
        [c], features, labels, pctrs, EngineConfig(reserve=0.001, control_interval_s=10.0),
        control_hook=runner.control_hook,
        throttle_hook=runner.throttle_hook,
        bid_hook=runner.bid_hook,
    )
    stats = eng.run(replay)
    rep = pacing_report(stats.spend_trace, [c], curve)
    assert rep.mean_abs_pacing_error_pct < 0.02   # under 2% — tracks tightly
    assert rep.early_exhaustion_frac == 0.0


def test_traffic_pid_paces_better_than_greedy():
    greedy = next(s for s in STRATEGIES if s.name == "greedy")
    tpid = next(s for s in STRATEGIES if s.name == "traffic_pid_bidshade")
    g_stats, g_camps, g_curve = _run(greedy)
    t_stats, t_camps, t_curve = _run(tpid)
    g_rep = pacing_report(g_stats.spend_trace, g_camps, g_curve)
    t_rep = pacing_report(t_stats.spend_trace, t_camps, t_curve)
    # the PID strategy tracks its target curve far better than greedy tracks it
    assert t_rep.mean_abs_pacing_error_pct < g_rep.mean_abs_pacing_error_pct
