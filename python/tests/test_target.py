"""Target curves: endpoints, monotonicity, traffic-awareness differs from uniform."""

import numpy as np

from pacer.sim.target import SECONDS_PER_HOUR, TrafficAwareTarget, UniformTarget


def test_uniform_endpoints_and_midpoint():
    u = UniformTarget(horizon_s=100.0)
    assert u.fraction(0) == 0.0
    assert u.fraction(50) == 0.5
    assert u.fraction(100) == 1.0
    assert u.fraction(200) == 1.0  # clamped
    assert u.spend_target(80.0, 50) == 40.0


def test_traffic_aware_endpoints():
    vol = np.array([10.0, 30.0, 60.0])  # skewed toward later hours
    ta = TrafficAwareTarget(vol)
    assert ta.fraction(0) == 0.0
    assert abs(ta.fraction(3 * SECONDS_PER_HOUR) - 1.0) < 1e-12


def test_traffic_aware_is_monotone():
    vol = np.array([5.0, 40.0, 15.0, 30.0])
    ta = TrafficAwareTarget(vol)
    ts = np.linspace(0, 4 * SECONDS_PER_HOUR, 200)
    fracs = [ta.fraction(t) for t in ts]
    assert all(b >= a - 1e-12 for a, b in zip(fracs, fracs[1:]))


def test_traffic_aware_differs_from_uniform_on_skew():
    # first hour is a trough (10%), so at end of hour 1 traffic-aware target should
    # be well below the uniform 1/3.
    vol = np.array([10.0, 45.0, 45.0])
    ta = TrafficAwareTarget(vol)
    u = UniformTarget(horizon_s=3 * SECONDS_PER_HOUR)
    t = 1 * SECONDS_PER_HOUR
    assert ta.fraction(t) == 0.1
    assert u.fraction(t) > ta.fraction(t)  # uniform over-targets in the trough


def test_from_hours():
    hours = np.array([14100100] * 5 + [14100101] * 20)
    ta = TrafficAwareTarget.from_hours(hours)
    # end of hour 0 -> 5/25 spent
    assert abs(ta.fraction(SECONDS_PER_HOUR) - 5 / 25) < 1e-12
