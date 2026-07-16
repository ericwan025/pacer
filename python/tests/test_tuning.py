"""Gain tuning: grid search picks the lowest-error gains and beats a bad gain."""

import numpy as np

from pacer.sim.pid import PIDConfig
from pacer.sim.tuning import grid_search, simulate_pacing

# a diurnal 24h profile (trough overnight, peak evening)
PROFILE = np.array(
    [2, 1, 1, 1, 2, 4, 7, 10, 12, 13, 13, 12, 11, 11, 12, 13, 14, 15, 16, 14, 11, 8, 5, 3],
    dtype=float,
)


def test_grid_search_returns_min_of_table():
    res = grid_search(PROFILE)
    assert res.table[0][1] == res.best_error
    assert res.best_error == min(e for _, e in res.table)
    # chosen gains match the winning row
    assert res.table[0][0] == {"kp": res.best.kp, "ki": res.best.ki, "kd": res.best.kd}


def test_tuned_beats_worst_gain_in_grid():
    res = grid_search(PROFILE)
    worst = res.table[-1][1]
    assert res.best_error < worst


def test_pacing_error_under_two_percent():
    # the reduced-order tuned controller keeps cumulative pacing error small
    res = grid_search(PROFILE)
    assert res.best_error < 0.02


def test_simulate_is_deterministic_given_seed():
    cfg = PIDConfig(kp=1.0, ki=0.1, kd=0.0, min_mult=0.0, max_mult=1.0)
    a = simulate_pacing(cfg, PROFILE, seed=3)
    b = simulate_pacing(cfg, PROFILE, seed=3)
    assert np.array_equal(a.spend, b.spend)


def test_final_utilization_reasonable():
    res = grid_search(PROFILE)
    out = simulate_pacing(res.best, PROFILE)
    assert 0.9 <= out.final_utilization <= 1.0
