"""Burst injection.

A pacer that only ever sees smooth traffic proves nothing. Bursts are what stress
the controller and make integral windup show up. We pick random windows and
multiply the arrival rate by a factor for a few minutes by DUPLICATING existing
impressions inside the window (real feature/label payloads, just more of them).

Every experiment runs in both smooth and bursty mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pacer.sim.traffic import TrafficReplay


@dataclass
class BurstConfig:
    n_bursts: int = 4
    min_factor: float = 2.0
    max_factor: float = 5.0
    min_duration_s: float = 5 * 60
    max_duration_s: float = 30 * 60
    seed: int = 0


def inject_bursts(replay: TrafficReplay, cfg: BurstConfig) -> TrafficReplay:
    """Return a NEW TrafficReplay with extra impressions inside burst windows.

    The added impressions reuse indices of impressions already inside the window,
    so payloads stay real. Timestamps are jittered within the window.
    """
    rng = np.random.default_rng(cfg.seed)
    ts = replay.timestamps
    duration = replay.duration_seconds()
    if duration <= 0:
        return replay

    new_idx: list[int] = []
    new_ts: list[float] = []
    for _ in range(cfg.n_bursts):
        win = rng.uniform(cfg.min_duration_s, cfg.max_duration_s)
        start = rng.uniform(0, max(1e-9, duration - win))
        end = start + win
        factor = rng.uniform(cfg.min_factor, cfg.max_factor)

        in_win = np.where((ts >= start) & (ts < end))[0]
        if len(in_win) == 0:
            continue
        # add (factor - 1) x the in-window volume
        n_add = int(round((factor - 1.0) * len(in_win)))
        picks = rng.choice(in_win, size=n_add, replace=True)
        new_idx.extend(picks.tolist())
        new_ts.extend(rng.uniform(start, end, size=n_add).tolist())

    if not new_idx:
        return replay

    new_idx_arr = np.array(new_idx, dtype=np.int64)
    merged = TrafficReplay.__new__(TrafficReplay)
    merged.hours = np.concatenate([replay.hours, replay.hours[new_idx_arr]])
    merged.timestamps = np.concatenate([replay.timestamps, np.array(new_ts)])
    merged.order = np.argsort(merged.timestamps, kind="mergesort")
    # duplicated rows point back at the ORIGINAL payloads they were copied from
    merged.payload_idx = np.concatenate(
        [replay.payload_idx, replay.payload_idx[new_idx_arr]]
    )
    merged.seed = replay.seed
    return merged
