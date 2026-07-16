"""Delivery-value metrics + controller settling time.

A pacer that produces a beautiful smooth spend curve while buying worse inventory
is a WORSE pacer. So we report value alongside pacing quality: clicks, clicks per
dollar, effective CPC, and impression share by daypart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DeliveryReport:
    total_clicks: int
    total_spend: float
    clicks_per_dollar: float
    effective_cpc: float                  # spend / clicks
    impression_share_by_daypart: np.ndarray  # len-24, sums to 1


def delivery_report(stats) -> DeliveryReport:
    clicks = int(stats.total_clicks)
    spend = float(stats.total_spend)
    wins = stats.wins_by_daypart.astype(np.float64)
    share = wins / wins.sum() if wins.sum() > 0 else np.zeros(24)
    return DeliveryReport(
        total_clicks=clicks,
        total_spend=spend,
        clicks_per_dollar=(clicks / spend) if spend > 0 else 0.0,
        effective_cpc=(spend / clicks) if clicks > 0 else float("inf"),
        impression_share_by_daypart=share,
    )


def settling_time(
    mult_trace: list[tuple[float, float]],
    burst_end_t: float,
    band: float = 0.1,
    steady: float | None = None,
) -> float | None:
    """Time (seconds) after burst_end for the multiplier to return and STAY within
    `band` (relative) of its steady-state value.

    steady defaults to the median multiplier over the tail of the trace (post-
    burst equilibrium). Returns None if it never settles.
    """
    if not mult_trace:
        return None
    ts = np.array([t for t, _ in mult_trace])
    ms = np.array([m for _, m in mult_trace])

    post = ms[ts >= burst_end_t]
    if len(post) == 0:
        return None
    if steady is None:
        steady = float(np.median(post[-max(1, len(post) // 4):]))
    tol = band * abs(steady) if steady != 0 else band

    within = np.abs(ms - steady) <= tol
    # find the last time it was OUTSIDE the band at/after the burst; it settles
    # on the next sample. If always within, it settled at burst_end.
    idxs = np.where((ts >= burst_end_t) & (~within))[0]
    if len(idxs) == 0:
        return 0.0
    last_bad = idxs[-1]
    if last_bad + 1 >= len(ts):
        return None  # still out of band at the end
    return float(ts[last_bad + 1] - burst_end_t)
