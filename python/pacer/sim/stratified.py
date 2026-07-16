"""Stratified throttling.

Uniform throttling drops impressions evenly, but the objective is VALUE, not
smoothness. A campaign would rather drop its lowest-pCTR impressions first. So
instead of a coin flip, participate iff  pCTR >= threshold, where the threshold is
the (1 - p) quantile of the recent pCTR distribution — i.e. keep the top-p
fraction by pCTR. That keeps the same participation *rate* while delivering more
clicks per participation.

The quantile is maintained online by a bounded reservoir sample (Vitter's
Algorithm R). A reservoir is simpler to reason about and to port than a t-digest,
and for a threshold at a moderate quantile it is plenty accurate.
"""

from __future__ import annotations

import numpy as np


class ReservoirQuantile:
    """Fixed-size uniform reservoir sample supporting quantile queries."""

    def __init__(self, capacity: int = 2000, seed: int = 0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.buf = np.empty(capacity, dtype=np.float64)
        self.n_seen = 0
        self.size = 0

    def add(self, x: float) -> None:
        if self.size < self.capacity:
            self.buf[self.size] = x
            self.size += 1
        else:
            # replace a random element with prob capacity/n_seen (Algorithm R)
            j = self.rng.integers(0, self.n_seen + 1)
            if j < self.capacity:
                self.buf[j] = x
        self.n_seen += 1

    def quantile(self, q: float) -> float:
        if self.size == 0:
            return 0.0
        return float(np.quantile(self.buf[: self.size], q))


class StratifiedThrottle:
    """Keep the top-p fraction of impressions by pCTR, updating the estimator as
    it goes."""

    def __init__(self, capacity: int = 2000, seed: int = 0):
        self.res = ReservoirQuantile(capacity, seed)

    def participate(self, pctr: float, p: float) -> bool:
        self.res.add(pctr)
        if p >= 1.0:
            return True
        if p <= 0.0:
            return False
        threshold = self.res.quantile(1.0 - p)
        return pctr >= threshold
