"""Pacing modes: bid-shade folds multiplier into bid; throttle gates participation."""

import numpy as np

from pacer.sim.campaign import Campaign, Targeting
from pacer.sim.engine import Engine, EngineConfig
from pacer.sim.pacing_mode import BidShadeMode, ThrottleMode
from pacer.sim.traffic import TrafficReplay


def _c(mult):
    c = Campaign(0, 1e9, 2.0, Targeting("banner_pos", frozenset({"0"})))
    c.pacing_multiplier = mult
    return c


def test_bid_shade_scales_bid():
    m = BidShadeMode()
    assert m.bid_hook(_c(0.5), 0.1) == 0.1 * 2.0 * 0.5
    assert m.throttle_hook(None) is None


def test_throttle_bids_full_value():
    m = ThrottleMode()
    # bid ignores the multiplier
    assert m.bid_hook(_c(0.3), 0.1) == 0.1 * 2.0 * 1.0


def test_throttle_participation_rate_matches_multiplier():
    m = ThrottleMode()
    rng = np.random.default_rng(0)
    hook = m.throttle_hook(rng)
    c = _c(0.3)
    n = 50000
    hits = sum(hook(c, 0.1) for _ in range(n))
    assert abs(hits / n - 0.3) < 0.01


def _run(mode, mult, seed=0):
    n = 2000
    rng = np.random.default_rng(seed)
    hours = np.repeat(np.array([14100100 + h for h in range(4)]), n // 4)
    replay = TrafficReplay(hours, seed=seed)
    features = [{"banner_pos": "0", "site_category": "a"} for _ in range(n)]
    labels = (rng.random(n) < 0.2).astype(np.int8)
    pctrs = rng.uniform(0.05, 0.4, n)
    c = _c(mult)
    hook_rng = np.random.default_rng(123)
    eng = Engine(
        [c], features, labels, pctrs, EngineConfig(reserve=0.0),
        throttle_hook=mode.throttle_hook(hook_rng),
        bid_hook=mode.bid_hook,
    )
    return eng.run(replay)


def test_throttle_serves_fewer_than_full_participation():
    full = _run(ThrottleMode(), mult=1.0)
    half = _run(ThrottleMode(), mult=0.5)
    # throttling to p=0.5 wins roughly half as many auctions
    assert half.auctions_with_winner < full.auctions_with_winner
    assert half.auctions_with_winner > 0
