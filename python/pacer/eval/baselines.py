"""The five pacing strategies compared by the eval harness.

  1. greedy                      — ASAP: no pacing, bid full value until budget gone
  2. uniform_pid_bidshade        — PID on a traffic-BLIND (uniform) target curve
  3. traffic_pid_uniform_throttle— traffic-aware PID, uniform probabilistic throttle
  4. traffic_pid_stratified_throttle — traffic-aware PID, drop low-pCTR first
  5. traffic_pid_bidshade        — traffic-aware PID, multiplier shades the bid

2 vs 5 isolates the target curve (uniform vs traffic-aware) at a fixed mode.
3 vs 4 isolates uniform vs stratified throttling. 3/4 vs 5 isolates throttle vs
bid-shade. Each runs in smooth and bursty traffic.

A PacingRunner owns per-campaign PID controllers (each campaign paces its own
budget) and per-campaign stratified estimators, and exposes the three engine
hooks (control / throttle / bid).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from pacer.sim.auction import compute_bid
from pacer.sim.campaign import Campaign
from pacer.sim.pid import PIDConfig, PIDController
from pacer.sim.stratified import StratifiedThrottle
from pacer.sim.target import TargetCurve
from pacer.sim.throttle import UniformThrottle


@dataclass(frozen=True)
class Strategy:
    name: str
    pid: bool             # does a controller run at all?
    traffic_aware: bool   # traffic-aware target vs uniform (informational; curve is passed in)
    mode: str             # 'greedy' | 'bid_shade' | 'throttle' | 'stratified'
    max_mult: float


STRATEGIES = [
    Strategy("greedy", pid=False, traffic_aware=False, mode="greedy", max_mult=1.0),
    Strategy("uniform_pid_bidshade", pid=True, traffic_aware=False, mode="bid_shade", max_mult=5.0),
    Strategy("traffic_pid_uniform_throttle", pid=True, traffic_aware=True, mode="throttle", max_mult=1.0),
    Strategy("traffic_pid_stratified_throttle", pid=True, traffic_aware=True, mode="stratified", max_mult=1.0),
    Strategy("traffic_pid_bidshade", pid=True, traffic_aware=True, mode="bid_shade", max_mult=5.0),
]


class PacingRunner:
    def __init__(
        self,
        strategy: Strategy,
        campaigns: list[Campaign],
        curve: TargetCurve,
        gains: PIDConfig,
        seed: int = 0,
    ):
        self.strategy = strategy
        self.curve = curve
        self.pids: dict[int, PIDController] = {}
        if strategy.pid:
            for c in campaigns:
                cfg = replace(gains, min_mult=0.0, max_mult=strategy.max_mult, mapping="linear")
                self.pids[c.id] = PIDController(cfg)
        self.uni = UniformThrottle(seed)
        self.strat = (
            {c.id: StratifiedThrottle(seed=seed + 1 + c.id) for c in campaigns}
            if strategy.mode == "stratified"
            else {}
        )

    # -- engine hooks ------------------------------------------------------
    def control_hook(self, t: float, campaigns: list[Campaign]) -> None:
        if not self.strategy.pid:
            return
        for c in campaigns:
            target = self.curve.spend_target(c.daily_budget, t)
            c.pacing_multiplier = self.pids[c.id].update(setpoint=target, measurement=c.spend)

    def throttle_hook(self, c: Campaign, pctr: float) -> bool:
        m = self.strategy.mode
        if m == "throttle":
            return self.uni.participate(c.pacing_multiplier)
        if m == "stratified":
            return self.strat[c.id].participate(pctr, c.pacing_multiplier)
        return True  # greedy / bid_shade always participate

    def bid_hook(self, c: Campaign, pctr: float) -> float:
        m = self.strategy.mode
        if m in ("throttle", "stratified"):
            return compute_bid(pctr, c.value_per_click, 1.0)  # full-value bid
        if m == "greedy":
            return compute_bid(pctr, c.value_per_click, 1.0)
        return compute_bid(pctr, c.value_per_click, c.pacing_multiplier)  # bid_shade
