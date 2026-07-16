"""Anti-windup: the failure mode (off) and the fix (on).

This is the single most compelling artifact in the repo. A traffic trough drives
the controller into saturation; without anti-windup the integral winds up and the
controller stays pinned at max_mult long after spend has caught the target.
"""

from pacer.sim.windup_demo import run_windup_scenario


def test_both_saturate_during_trough():
    on = run_windup_scenario(anti_windup=True)
    off = run_windup_scenario(anti_windup=False)
    # the scenario must actually stress the controller, else it proves nothing
    assert max(on.mults) >= 3.0 - 1e-9
    assert max(off.mults) >= 3.0 - 1e-9


def test_both_eventually_catch_target():
    on = run_windup_scenario(anti_windup=True)
    off = run_windup_scenario(anti_windup=False)
    assert on.caught_step is not None
    assert off.caught_step is not None


def test_anti_windup_recovers_much_faster():
    on = run_windup_scenario(anti_windup=True)
    off = run_windup_scenario(anti_windup=False)
    assert on.recovery_steps is not None and off.recovery_steps is not None
    # the whole point: windup keeps the controller saturated far longer
    assert off.recovery_steps > 10 * on.recovery_steps


def test_windup_causes_larger_overshoot():
    on = run_windup_scenario(anti_windup=True)
    off = run_windup_scenario(anti_windup=False)
    # overshoot = how far spend runs past target after catching up
    on_over = max(s - t for s, t in zip(on.spend, on.target))
    off_over = max(s - t for s, t in zip(off.spend, off.target))
    assert off_over > on_over
