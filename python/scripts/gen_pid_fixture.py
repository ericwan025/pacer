"""Generate the Go<->Python PID parity fixture.

Runs the Python PIDController over a trace that includes saturation (to exercise
anti-windup) and a moving setpoint (to exercise the filtered derivative), and
writes config + trace + outputs. The Go controller must reproduce the outputs.

Run: python -m scripts.gen_pid_fixture
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pacer.sim.pid import PIDConfig, PIDController

OUT = Path("../go/internal/controller/testdata")


def main() -> None:
    cfg = PIDConfig(
        kp=1.5, ki=0.2, kd=0.8, dt=1.0,
        min_mult=0.0, max_mult=3.0, deriv_tau=5.0, anti_windup=True, mapping="linear",
    )
    pid = PIDController(cfg)

    rng = np.random.default_rng(0)
    trace = []
    outputs = []
    spend = 0.0
    for k in range(400):
        setpoint = k / 400.0                      # ramps 0 -> ~1 (moving setpoint)
        measurement = spend + rng.normal(0, 0.01)  # noisy measurement
        out = pid.update(setpoint, measurement)
        trace.append([setpoint, measurement])
        outputs.append(out)
        # crude plant so measurement moves and sometimes saturates the controller
        spend += 0.002 * out

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "kp": cfg.kp, "ki": cfg.ki, "kd": cfg.kd, "dt": cfg.dt,
            "min_mult": cfg.min_mult, "max_mult": cfg.max_mult,
            "deriv_tau": cfg.deriv_tau, "anti_windup": cfg.anti_windup,
            "mapping": cfg.mapping,
        },
        "trace": trace,
        "outputs": outputs,
    }
    (OUT / "pid_parity.json").write_text(json.dumps(payload))
    print(f"wrote PID fixture: {len(outputs)} steps -> {OUT}/pid_parity.json")


if __name__ == "__main__":
    main()
