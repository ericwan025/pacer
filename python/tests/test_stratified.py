"""Stratified throttling: rate ~ p, and it delivers more clicks than uniform at
matched participation."""

import numpy as np

from pacer.sim.stratified import ReservoirQuantile, StratifiedThrottle
from pacer.sim.throttle import UniformThrottle


def test_reservoir_quantile_accurate():
    res = ReservoirQuantile(capacity=5000, seed=0)
    rng = np.random.default_rng(1)
    data = rng.random(50000)
    for x in data:
        res.add(x)
    # median estimate close to true 0.5
    assert abs(res.quantile(0.5) - 0.5) < 0.03


def _stream(n=40000, seed=0):
    rng = np.random.default_rng(seed)
    pctr = rng.beta(2, 8, n)           # skewed toward low values, CTR-like
    label = (rng.random(n) < pctr).astype(int)  # click prob == pctr
    return pctr, label


def test_stratified_rate_near_target():
    pctr, _ = _stream()
    st = StratifiedThrottle(capacity=2000, seed=0)
    p = 0.5
    # warm up so the threshold is stable before we measure
    kept = [st.participate(x, p) for x in pctr]
    rate = np.mean(kept[5000:])
    assert abs(rate - p) < 0.05


def test_stratified_delivers_more_clicks_than_uniform():
    pctr, label = _stream(seed=2)
    p = 0.5

    st = StratifiedThrottle(capacity=2000, seed=0)
    strat_keep = np.array([st.participate(x, p) for x in pctr])

    un = UniformThrottle(seed=0)
    uni_keep = un.participate_batch(p, len(pctr))

    # matched participation rate (both ~ p)
    assert abs(strat_keep.mean() - uni_keep.mean()) < 0.05
    # but stratified keeps the high-pCTR impressions -> more clicks delivered
    strat_clicks = label[strat_keep].sum()
    uni_clicks = label[uni_keep].sum()
    assert strat_clicks > uni_clicks * 1.3
