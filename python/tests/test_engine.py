"""Engine: spend never exceeds budget, control hook fires on cadence, clicks
come from labels."""

import numpy as np

from pacer.sim.campaign import Campaign, Targeting
from pacer.sim.engine import Engine, EngineConfig
from pacer.sim.traffic import TrafficReplay


def _setup(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    hours = np.repeat(np.array([14100100 + h for h in range(4)]), n // 4)
    replay = TrafficReplay(hours, seed=seed)
    features = [{"banner_pos": "0", "site_category": "a"} for _ in range(n)]
    labels = (rng.random(n) < 0.2).astype(np.int8)
    pctrs = rng.uniform(0.05, 0.4, n)
    return replay, features, labels, pctrs


def test_spend_never_exceeds_budget():
    replay, features, labels, pctrs = _setup()
    camps = [
        Campaign(i, daily_budget=5.0, value_per_click=1.0,
                 targeting=Targeting("banner_pos", frozenset({"0"})))
        for i in range(10)
    ]
    eng = Engine(camps, features, labels, pctrs, EngineConfig(reserve=0.01))
    stats = eng.run(replay)
    for c in camps:
        assert c.spend <= c.daily_budget + 1e-9
    assert abs(stats.total_spend - sum(c.spend for c in camps)) < 1e-6


def test_clicks_come_from_labels_not_pctr():
    replay, features, labels, pctrs = _setup(seed=3)
    # single campaign wins everything within budget
    camps = [Campaign(0, 1e9, 1.0, Targeting("banner_pos", frozenset({"0"})))]
    eng = Engine(camps, features, labels, pctrs, EngineConfig(reserve=0.0))
    stats = eng.run(replay)
    # with no competition, clicks == number of label==1 impressions it served
    assert stats.total_clicks == int(labels.sum())


def test_control_hook_fires_on_cadence():
    replay, features, labels, pctrs = _setup()
    camps = [Campaign(0, 100.0, 1.0, Targeting("banner_pos", frozenset({"0"})))]
    fired = []
    eng = Engine(camps, features, labels, pctrs,
                 EngineConfig(control_interval_s=10.0),
                 control_hook=lambda t, cs: fired.append(t))
    eng.run(replay)
    # ticks are 10s apart and monotonic
    assert fired == sorted(fired)
    assert all(abs((b - a) - 10.0) < 1e-6 for a, b in zip(fired, fired[1:]))


def test_targeting_excludes_nonmatching():
    replay, features, labels, pctrs = _setup()
    camps = [Campaign(0, 100.0, 1.0, Targeting("banner_pos", frozenset({"9"})))]
    eng = Engine(camps, features, labels, pctrs, EngineConfig())
    stats = eng.run(replay)
    assert stats.total_spend == 0.0
    assert stats.auctions_with_winner == 0
