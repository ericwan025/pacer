"""The PID pacing controller.

    error(t)   = target_spend(t) - actual_spend(t)
    P          = Kp * error
    I         += Ki * error * dt      (with anti-windup)
    D          = Kd * filtered_derivative_on_measurement
    output     = P + I + D
    multiplier = clamp(map(output), min_mult, max_mult)

Design choices, all interview material:

* Anti-windup by CONDITIONAL INTEGRATION: when the mapped output is saturated
  against a clamp and the error would push it further into that rail, we freeze
  the integral. Without this, a bursty morning drives I huge and the controller
  stays pinned long after it should have recovered.

* DERIVATIVE ON MEASUREMENT, not error, so a moving setpoint doesn't cause a
  derivative "kick". Plus a first-order low-pass filter (time constant tau) so
  raw noise on the spend signal doesn't wreck the D term.

* The controller runs every dt seconds (default 10), NOT per impression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PIDConfig:
    kp: float
    ki: float
    kd: float = 0.0
    dt: float = 10.0
    min_mult: float = 0.0
    max_mult: float = 10.0
    deriv_tau: float = 30.0  # low-pass time constant for the derivative
    anti_windup: bool = True
    mapping: str = "linear"  # "linear" or "sigmoid"


def _map(mapping: str, raw: float, lo: float, hi: float) -> float:
    if mapping == "sigmoid":
        return lo + (hi - lo) * (1.0 / (1.0 + math.exp(-raw)))
    return raw  # linear/identity; clamp applied by caller


class PIDController:
    def __init__(self, cfg: PIDConfig):
        self.cfg = cfg
        self.integral = 0.0
        self.filtered_deriv = 0.0
        self.prev_measurement: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.filtered_deriv = 0.0
        self.prev_measurement = None

    def update(self, setpoint: float, measurement: float) -> float:
        c = self.cfg
        error = setpoint - measurement

        # --- derivative on measurement (not error), low-pass filtered ---
        if self.prev_measurement is None:
            raw_deriv = 0.0
        else:
            # d(error)/dt = -d(measurement)/dt when setpoint is slow-moving
            raw_deriv = -(measurement - self.prev_measurement) / c.dt
        alpha = c.dt / (c.deriv_tau + c.dt)
        self.filtered_deriv += alpha * (raw_deriv - self.filtered_deriv)
        D = c.kd * self.filtered_deriv

        P = c.kp * error

        # --- anti-windup: test provisional output for saturation first ---
        inc = c.ki * error * c.dt
        provisional = P + (self.integral + inc) + D
        mapped_prov = _map(c.mapping, provisional, c.min_mult, c.max_mult)
        sat_high = mapped_prov > c.max_mult
        sat_low = mapped_prov < c.min_mult
        pushing_out = (sat_high and error > 0) or (sat_low and error < 0)
        if not (c.anti_windup and pushing_out):
            self.integral += inc

        output = P + self.integral + D
        mult = _map(c.mapping, output, c.min_mult, c.max_mult)
        mult = min(c.max_mult, max(c.min_mult, mult))

        self.prev_measurement = measurement
        return mult
