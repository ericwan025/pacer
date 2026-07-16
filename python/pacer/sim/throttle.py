"""Probabilistic throttling layer.

Given a participation probability p from the controller, decide participation per
request with a single PRNG draw: participate iff U[0,1) < p. We use numpy's
Generator (PCG64) — a fast, statistically sound, NON-cryptographic PRNG. Crypto
randomness would be pointless overhead on the hot path.

The Go serving layer mirrors this with its own fast PRNG (math/rand, not
crypto/rand); the *statistical* properties are what must match, not the exact
stream.
"""

from __future__ import annotations

import numpy as np


class UniformThrottle:
    """Each request participates independently with probability p."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def participate(self, p: float) -> bool:
        if p >= 1.0:
            return True
        if p <= 0.0:
            return False
        return bool(self.rng.random() < p)

    def participate_batch(self, p: float, n: int) -> np.ndarray:
        """Vectorized draws — used by tests and offline analysis."""
        if p >= 1.0:
            return np.ones(n, dtype=bool)
        if p <= 0.0:
            return np.zeros(n, dtype=bool)
        return self.rng.random(n) < p
