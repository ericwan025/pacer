"""Anti-windup demonstration scenario.

A traffic trough starves spend while the target keeps rising, so the error grows
and the controller saturates at max_mult. With anti-windup OFF the integral term
winds up over the whole trough; when traffic returns the controller stays pinned
at max long after spend has caught the target, overshooting badly. With
anti-windup ON the integral is frozen during saturation, so recovery is prompt.

This scenario is used by both the test and the Phase 6 plot, so the compelling
artifact and the asserted behavior come from the exact same code.
"""

from __future__ import annotations

from dataclasses import dataclass

from pacer.sim.pid import PIDConfig, PIDController


@dataclass
class WindupResult:
    mults: list[float]
    spend: list[float]
    target: list[float]
    caught_step: int | None      # first step after the trough where spend >= target
    recovery_steps: int | None   # steps from caught until multiplier leaves the max rail


def run_windup_scenario(
    anti_windup: bool,
    steps: int = 300,
    trough: tuple[int, int] = (40, 90),
    target_rate: float = 1.0,
    max_mult: float = 3.0,
) -> WindupResult:
    cfg = PIDConfig(
        kp=0.02, ki=0.01, kd=0.0, dt=1.0,
        min_mult=0.0, max_mult=max_mult, anti_windup=anti_windup,
    )
    pid = PIDController(cfg)

    spend = 0.0
    mults: list[float] = []
    spends: list[float] = []
    targets: list[float] = []
    caught_step: int | None = None
    recovery_steps: int | None = None

    for k in range(steps):
        target = target_rate * k
        u = pid.update(setpoint=target, measurement=spend)
        # available spend per step: a deep trough throttles delivery mid-run
        avail = 0.05 if trough[0] <= k < trough[1] else 2.0
        delivered = min(avail, u * 1.0)  # price = 1
        spend += delivered

        mults.append(u)
        spends.append(spend)
        targets.append(target)

        if k >= trough[1] and caught_step is None and spend >= target:
            caught_step = k
        if (
            caught_step is not None
            and recovery_steps is None
            and u < max_mult * 0.99
        ):
            recovery_steps = k - caught_step

    return WindupResult(mults, spends, targets, caught_step, recovery_steps)
