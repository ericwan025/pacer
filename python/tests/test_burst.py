"""Burst injection: adds volume, keeps payload indices valid, stays ordered."""

import numpy as np

from pacer.sim.burst import BurstConfig, inject_bursts
from pacer.sim.traffic import TrafficReplay


def _replay(n=5000, seed=0):
    # 5 hours of uniform traffic
    hours = np.repeat(np.array([14100100 + h for h in range(5)]), n // 5)
    return TrafficReplay(hours, seed=seed)


def test_burst_adds_volume():
    r = _replay()
    b = inject_bursts(r, BurstConfig(n_bursts=3, seed=1))
    assert len(b) > len(r)


def test_payload_indices_stay_in_range():
    r = _replay()
    n_orig = len(r)
    b = inject_bursts(r, BurstConfig(n_bursts=4, seed=2))
    for pidx, _ in b.stream():
        assert 0 <= pidx < n_orig


def test_stream_time_ordered_after_burst():
    b = inject_bursts(_replay(), BurstConfig(n_bursts=4, seed=3))
    times = [t for _, t in b.stream()]
    assert times == sorted(times)


def test_deterministic():
    r = _replay()
    a = inject_bursts(r, BurstConfig(seed=5))
    b = inject_bursts(r, BurstConfig(seed=5))
    assert np.array_equal(a.timestamps, b.timestamps)
    assert np.array_equal(a.payload_idx, b.payload_idx)
