"""Pacing-quality metrics.

All operate on the per-campaign spend trace the engine records at each control
tick: cid -> [(t_seconds, cumulative_spend), ...], plus the target curve and the
campaigns themselves. Definitions are fixed here so every strategy is measured the
same way and the README numbers are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pacer.sim.campaign import Campaign
from pacer.sim.target import TargetCurve

SECONDS_PER_HOUR = 3600.0
EARLY_EXHAUSTION_HOUR = 20.0     # "early" = exhausted before this hour
EXHAUSTION_FRACTION = 0.999      # spent >= this fraction of budget = exhausted


@dataclass
class PacingReport:
    mean_abs_pacing_error_pct: float      # time+campaign-averaged |spend-target|/budget
    l2_spend_deviation: float             # RMS of (spend-target)/budget
    utilization_mean: float
    utilization: np.ndarray               # per-campaign spend/budget
    early_exhaustion_frac: float


def _campaign_error_series(
    trace: list[tuple[float, float]], curve: TargetCurve, budget: float
) -> np.ndarray:
    """|spend(t) - target(t)| / budget at each recorded tick."""
    if not trace or budget <= 0:
        return np.array([0.0])
    ts = np.array([t for t, _ in trace])
    spends = np.array([s for _, s in trace])
    targets = np.array([curve.spend_target(budget, t) for t in ts])
    return np.abs(spends - targets) / budget


def budget_utilization(campaigns: list[Campaign]) -> np.ndarray:
    return np.array([c.spend / c.daily_budget if c.daily_budget > 0 else 0.0
                     for c in campaigns])


def early_exhaustion_fraction(
    spend_trace: dict, campaigns: list[Campaign]
) -> float:
    """Fraction of campaigns that spent >= EXHAUSTION_FRACTION of budget before
    EARLY_EXHAUSTION_HOUR."""
    cutoff_t = EARLY_EXHAUSTION_HOUR * SECONDS_PER_HOUR
    early = 0
    for c in campaigns:
        trace = spend_trace.get(c.id, [])
        thresh = EXHAUSTION_FRACTION * c.daily_budget
        for t, spend in trace:
            if t < cutoff_t and spend >= thresh:
                early += 1
                break
    return early / len(campaigns) if campaigns else 0.0


def pacing_report(
    spend_trace: dict, campaigns: list[Campaign], curve: TargetCurve
) -> PacingReport:
    per_campaign_mae = []
    per_campaign_l2 = []
    for c in campaigns:
        err = _campaign_error_series(spend_trace.get(c.id, []), curve, c.daily_budget)
        per_campaign_mae.append(float(np.mean(err)))
        per_campaign_l2.append(float(np.sqrt(np.mean(err**2))))

    util = budget_utilization(campaigns)
    return PacingReport(
        mean_abs_pacing_error_pct=float(np.mean(per_campaign_mae)),
        l2_spend_deviation=float(np.mean(per_campaign_l2)),
        utilization_mean=float(np.mean(util)),
        utilization=util,
        early_exhaustion_frac=early_exhaustion_fraction(spend_trace, campaigns),
    )
