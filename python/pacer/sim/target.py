"""Spend target curves.

The naive pacer targets uniform spend: target(t) = budget * t / horizon. That is
wrong because traffic is not uniform. Targeting uniform spend during a 3am trough
forces over-bidding on garbage inventory; during a 9pm peak it under-bids and
misses the best impressions of the day.

Traffic-aware target: target(t) = budget * cumulative_expected_traffic(t) / total,
where the expected-traffic curve is estimated from the training days' hourly
volumes. We implement BOTH so the uniform pacer is a real baseline to beat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SECONDS_PER_HOUR = 3600.0


class TargetCurve:
    """fraction(t) in [0,1] = fraction of the daily budget that *should* be spent
    by time t. spend_target = budget * fraction(t)."""

    def fraction(self, t: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def spend_target(self, budget: float, t: float) -> float:
        return budget * self.fraction(t)


@dataclass
class UniformTarget(TargetCurve):
    horizon_s: float

    def fraction(self, t: float) -> float:
        if self.horizon_s <= 0:
            return 1.0
        return float(np.clip(t / self.horizon_s, 0.0, 1.0))


class TrafficAwareTarget(TargetCurve):
    """Cumulative-traffic target built from per-hour expected volumes."""

    def __init__(self, hourly_volume: np.ndarray):
        vol = np.asarray(hourly_volume, dtype=np.float64)
        if vol.sum() <= 0:
            raise ValueError("hourly_volume must have positive total")
        self.vol = vol
        self.prefix = np.concatenate([[0.0], np.cumsum(vol)])  # len n+1
        self.total = float(vol.sum())
        self.horizon_s = len(vol) * SECONDS_PER_HOUR

    def fraction(self, t: float) -> float:
        if t <= 0:
            return 0.0
        if t >= self.horizon_s:
            return 1.0
        h = int(t // SECONDS_PER_HOUR)
        frac_in = (t - h * SECONDS_PER_HOUR) / SECONDS_PER_HOUR
        cum = self.prefix[h] + frac_in * self.vol[h]
        return float(cum / self.total)

    @classmethod
    def from_hours(cls, hours: np.ndarray) -> "TrafficAwareTarget":
        from pacer.sim.traffic import hourly_volume

        _, counts = hourly_volume(hours)
        return cls(counts)
