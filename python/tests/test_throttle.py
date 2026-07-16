"""Uniform throttle: realized rate within 3 SE of target, no lag-1 correlation."""

import numpy as np
import pytest

from pacer.sim.throttle import UniformThrottle


@pytest.mark.parametrize("p", [0.0, 0.05, 0.37, 0.5, 0.83, 1.0])
def test_realized_rate_within_3_se(p):
    n = 100_000
    draws = UniformThrottle(seed=1).participate_batch(p, n)
    rate = draws.mean()
    if p in (0.0, 1.0):
        assert rate == p
        return
    se = np.sqrt(p * (1 - p) / n)
    assert abs(rate - p) < 3 * se


def test_scalar_matches_boundaries():
    t = UniformThrottle(seed=2)
    assert t.participate(1.0) is True
    assert t.participate(0.0) is False


def test_no_lag1_autocorrelation():
    n = 200_000
    draws = UniformThrottle(seed=3).participate_batch(0.4, n).astype(float)
    x = draws - draws.mean()
    r1 = np.sum(x[:-1] * x[1:]) / np.sum(x * x)
    # independent draws -> lag-1 autocorrelation ~ 0 (within ~3/sqrt(n))
    assert abs(r1) < 3 / np.sqrt(n) + 0.005


def test_deterministic_given_seed():
    a = UniformThrottle(seed=9).participate_batch(0.3, 1000)
    b = UniformThrottle(seed=9).participate_batch(0.3, 1000)
    assert np.array_equal(a, b)
