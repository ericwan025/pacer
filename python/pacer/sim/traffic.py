"""Traffic replay.

Avazu is hourly. The pacer needs sub-hour resolution, so we take the REAL hourly
impression volume as the envelope and, within each hour, scatter that hour's
impressions uniformly at random across the 3600 seconds to synthesize
second-level arrival times.

Honest labeling (see README Limitations): the diurnal *volume shape* is real; the
intra-hour *arrival times* are synthetic. We never imply the timestamps are real.
"""

from __future__ import annotations

import numpy as np

SECONDS_PER_HOUR = 3600


def hour_index(hours: np.ndarray) -> np.ndarray:
    """Map YYMMDDHH values to a dense chronological hour index 0,1,2,...

    Uses the sorted unique hours, so gaps in the data don't create empty hours
    in the timeline (we index observed hours, not wall-clock hours)."""
    hours = np.asarray(hours)
    uniq = np.unique(hours)
    lookup = {h: i for i, h in enumerate(uniq.tolist())}
    return np.array([lookup[h] for h in hours.tolist()], dtype=np.int64)


def hourly_volume(hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique_hours_sorted, counts) — the traffic envelope."""
    uniq, counts = np.unique(np.asarray(hours), return_counts=True)
    return uniq, counts


def synthesize_timestamps(hours: np.ndarray, seed: int = 0) -> np.ndarray:
    """Assign each impression a timestamp in seconds since the first hour.

    second = hour_index * 3600 + U[0, 3600).  Deterministic given seed.
    """
    hidx = hour_index(hours)
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(0.0, SECONDS_PER_HOUR, size=len(hidx))
    return hidx.astype(np.float64) * SECONDS_PER_HOUR + offsets


class TrafficReplay:
    """Ordered stream of impression indices at synthetic timestamps.

    Holds only the timestamps + the chronological order; the caller keeps the
    actual impression payloads (features, labels) and indexes into them.
    """

    def __init__(self, hours: np.ndarray, seed: int = 0):
        self.hours = np.asarray(hours)
        self.timestamps = synthesize_timestamps(self.hours, seed)
        self.order = np.argsort(self.timestamps, kind="mergesort")
        # payload_idx[i] is which original impression row position i refers to.
        # Identity here; burst injection remaps it so duplicated rows still point
        # at real payloads.
        self.payload_idx = np.arange(len(self.hours), dtype=np.int64)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.hours)

    def stream(self):
        """Yield (payload_index, timestamp_seconds) in time order."""
        for i in self.order:
            yield int(self.payload_idx[i]), float(self.timestamps[i])

    def duration_seconds(self) -> float:
        return float(self.timestamps.max()) if len(self.timestamps) else 0.0
