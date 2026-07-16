"""PID gain tuning by grid search on a reduced-order pacing model.

We don't hand-wave the gains. We sweep (Kp, Ki, Kd) over a grid, score each on
mean absolute pacing error against the traffic-aware target, and pick the best.
Tuning is done on the VALIDATION days' hourly volumes and reported on TEST — the
honest split.

The scoring model is a reduced-order pacer: at each control tick the achievable
spend is proportional to the instantaneous traffic rate, and actual spend =
achievable * participation, where participation is the clamped multiplier. Supply
exceeds budget (supply_ratio > 1), so the controller MUST pace down to hit the
target instead of just spending everything. This is fast enough to grid-search
and captures the diurnal pacing problem the full engine faces.

Caveat: pacing error is measured on CUMULATIVE spend vs a cumulative target, so
per-tick traffic noise largely averages out and the absolute errors here are
optimistic. Use this model to RANK gains, not to quote the headline pacing number
— that comes from the full engine in the Phase 6 harness.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from pacer.sim.pid import PIDConfig, PIDController
from pacer.sim.target import SECONDS_PER_HOUR, TrafficAwareTarget


@dataclass
class PacingSimResult:
    mean_abs_pacing_error_pct: float
    final_utilization: float
    times: np.ndarray
    spend: np.ndarray
    target: np.ndarray


def simulate_pacing(
    cfg: PIDConfig,
    hourly_volume: np.ndarray,
    budget: float = 1000.0,
    control_interval_s: float = 10.0,
    supply_ratio: float = 3.0,
    noise_cv: float = 0.5,
    seed: int = 0,
) -> PacingSimResult:
    vol = np.asarray(hourly_volume, dtype=np.float64)
    total_vol = vol.sum()
    horizon = len(vol) * SECONDS_PER_HOUR
    target_curve = TrafficAwareTarget(vol)
    rng = np.random.default_rng(seed)
    # lognormal multiplier with the requested coefficient of variation, so that
    # available inventory is bursty tick-to-tick and sluggish/aggressive gains
    # actually track differently.
    sigma = float(np.sqrt(np.log(1.0 + noise_cv**2)))

    pid = PIDController(cfg)
    spend = 0.0
    times, spends, targets = [], [], []

    t = 0.0
    while t < horizon:
        h = min(int(t // SECONDS_PER_HOUR), len(vol) - 1)
        # achievable spend this tick if fully participating (supply > budget)
        rate = vol[h] / total_vol  # fraction of daily traffic per hour
        noise = rng.lognormal(-0.5 * sigma**2, sigma)  # mean 1
        achievable = (
            rate * supply_ratio * budget
            * (control_interval_s / SECONDS_PER_HOUR)
            * noise
        )

        target = target_curve.spend_target(budget, t)
        # feed BUDGET-NORMALIZED error so tuned gains transfer to the full engine,
        # which normalizes the same way (see eval/baselines.py control_hook).
        mult = pid.update(setpoint=target_curve.fraction(t), measurement=spend / budget)
        participation = min(1.0, max(0.0, mult))
        delivered = min(achievable * participation, max(0.0, budget - spend))
        spend += delivered

        times.append(t)
        spends.append(spend)
        targets.append(target)
        t += control_interval_s

    times = np.array(times)
    spends = np.array(spends)
    targets = np.array(targets)
    mape = float(np.mean(np.abs(spends - targets)) / budget)
    return PacingSimResult(mape, spend / budget, times, spends, targets)


DEFAULT_GRID = {
    "kp": [0.5, 1.0, 2.0, 4.0],
    "ki": [0.05, 0.1, 0.2, 0.4],
    "kd": [0.0, 0.5, 1.0],
}


@dataclass
class TuningResult:
    best: PIDConfig
    best_error: float
    table: list[tuple[dict, float]]  # (params, error), sorted best-first


def grid_search(
    hourly_volume: np.ndarray,
    grid: dict | None = None,
    base_cfg: PIDConfig | None = None,
) -> TuningResult:
    grid = grid or DEFAULT_GRID
    base = base_cfg or PIDConfig(kp=1.0, ki=0.1, kd=0.0, min_mult=0.0, max_mult=1.0)

    table: list[tuple[dict, float]] = []
    best_err = float("inf")
    best_params = None
    for kp, ki, kd in itertools.product(grid["kp"], grid["ki"], grid["kd"]):
        cfg = PIDConfig(
            kp=kp, ki=ki, kd=kd, dt=base.dt,
            min_mult=base.min_mult, max_mult=base.max_mult,
            deriv_tau=base.deriv_tau, anti_windup=base.anti_windup,
            mapping=base.mapping,
        )
        err = simulate_pacing(cfg, hourly_volume).mean_abs_pacing_error_pct
        table.append(({"kp": kp, "ki": ki, "kd": kd}, err))
        if err < best_err:
            best_err = err
            best_params = (kp, ki, kd)

    table.sort(key=lambda x: x[1])
    kp, ki, kd = best_params
    best = PIDConfig(
        kp=kp, ki=ki, kd=kd, dt=base.dt,
        min_mult=base.min_mult, max_mult=base.max_mult,
        deriv_tau=base.deriv_tau, anti_windup=base.anti_windup, mapping=base.mapping,
    )
    return TuningResult(best, best_err, table)
