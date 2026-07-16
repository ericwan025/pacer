"""PID core: zero steady-state error on a step, derivative filter behavior."""

from pacer.sim.pid import PIDConfig, PIDController


def _run_closed_loop(cfg, r, steps, a=0.7, b=0.3, y0=0.0):
    """Stable first-order plant y_{k+1} = a*y + b*u driven by the controller.
    Returns the final measurement and the full trace."""
    pid = PIDController(cfg)
    y = y0
    trace = [y]
    for _ in range(steps):
        u = pid.update(setpoint=r, measurement=y)
        y = a * y + b * u
        trace.append(y)
    return y, trace


def test_zero_steady_state_error_on_step():
    # PI controller, wide clamp so it never saturates; must drive error -> 0.
    cfg = PIDConfig(kp=0.5, ki=0.1, kd=0.0, dt=1.0, min_mult=-100, max_mult=100)
    y, _ = _run_closed_loop(cfg, r=5.0, steps=400)
    assert abs(y - 5.0) < 1e-2


def test_integral_off_leaves_steady_state_error():
    # pure P controller cannot remove steady-state error on this plant
    cfg = PIDConfig(kp=0.5, ki=0.0, kd=0.0, dt=1.0, min_mult=-100, max_mult=100)
    y, _ = _run_closed_loop(cfg, r=5.0, steps=400)
    assert abs(y - 5.0) > 0.1


def test_derivative_filter_smooths_noise():
    # feed a noisy measurement; filtered derivative magnitude stays bounded and
    # far smaller than the raw step-to-step change would imply.
    cfg = PIDConfig(kp=0.0, ki=0.0, kd=1.0, dt=1.0, deriv_tau=50.0,
                    min_mult=-1e9, max_mult=1e9)
    pid = PIDController(cfg)
    import random

    random.seed(0)
    outs = []
    for _ in range(200):
        meas = 5.0 + random.uniform(-1, 1)  # noisy around constant
        outs.append(pid.update(setpoint=5.0, measurement=meas))
    # with heavy filtering the D output stays small despite unit-scale noise
    assert max(abs(o) for o in outs[10:]) < 0.5


def test_no_derivative_kick_on_setpoint_change():
    # derivative is on measurement, so a setpoint jump alone must not spike output
    cfg = PIDConfig(kp=0.0, ki=0.0, kd=5.0, dt=1.0, deriv_tau=1.0,
                    min_mult=-1e9, max_mult=1e9)
    pid = PIDController(cfg)
    pid.update(setpoint=0.0, measurement=2.0)
    out = pid.update(setpoint=100.0, measurement=2.0)  # setpoint jumps, meas same
    assert abs(out) < 1e-9
