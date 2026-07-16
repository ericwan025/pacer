"""Traffic replay: envelope preserved, timestamps land in the right hour, ordered."""

import numpy as np

from pacer.sim.traffic import (
    SECONDS_PER_HOUR,
    TrafficReplay,
    hourly_volume,
    synthesize_timestamps,
)


def _hours():
    # 3 hours with heterogeneous volume (envelope must be preserved)
    return np.array([14100100] * 5 + [14100101] * 20 + [14100102] * 8)


def test_envelope_counts_preserved():
    uniq, counts = hourly_volume(_hours())
    assert list(uniq) == [14100100, 14100101, 14100102]
    assert list(counts) == [5, 20, 8]


def test_timestamps_within_their_hour():
    hours = _hours()
    ts = synthesize_timestamps(hours, seed=1)
    # hour 0 -> [0,3600), hour 1 -> [3600,7200), hour 2 -> [7200,10800)
    order = {14100100: 0, 14100101: 1, 14100102: 2}
    for h, t in zip(hours, ts):
        lo = order[h] * SECONDS_PER_HOUR
        assert lo <= t < lo + SECONDS_PER_HOUR


def test_deterministic_given_seed():
    h = _hours()
    assert np.array_equal(synthesize_timestamps(h, 7), synthesize_timestamps(h, 7))
    assert not np.array_equal(synthesize_timestamps(h, 7), synthesize_timestamps(h, 8))


def test_stream_is_time_ordered():
    r = TrafficReplay(_hours(), seed=2)
    times = [t for _, t in r.stream()]
    assert times == sorted(times)
    assert len(list(r.stream())) == len(_hours())
