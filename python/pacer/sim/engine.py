"""Discrete-event simulation engine.

Streams impressions in timestamp order. For each impression:
  1. find eligible campaigns (targeting match + budget left + not throttled out)
  2. each computes bid = pCTR * value * pacing_multiplier
  3. run the second-price auction
  4. charge the winner the clearing price
  5. sample the click from the TRUE label; if the winner "wins" the click, that is
     just recorded — the charge is per-impression (CPM-style) in this simulator.

Pacing control (the PID controller writing multipliers) and throttling are passed
in as pluggable callbacks so we can swap strategies for the eval harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from pacer.sim.auction import compute_bid, run_auction
from pacer.sim.campaign import Campaign

# A control hook is called every control_interval_s seconds with (t, campaigns).
ControlHook = Callable[[float, list[Campaign]], None]
# A throttle hook decides participation: (campaign, pctr) -> bool.
ThrottleHook = Callable[[Campaign, float], bool]
# A bid hook computes the bid: (campaign, pctr) -> bid_amount. Lets pacing modes
# decide whether the multiplier shades the bid or gates participation.
BidHook = Callable[[Campaign, float], float]


@dataclass
class EngineConfig:
    reserve: float = 0.01
    control_interval_s: float = 10.0


@dataclass
class RunStats:
    impressions: int = 0
    auctions_with_winner: int = 0
    total_spend: float = 0.0
    total_clicks: int = 0
    # per-campaign spend trace sampled at each control tick: cid -> [(t, spend)]
    spend_trace: dict = field(default_factory=dict)


class Engine:
    def __init__(
        self,
        campaigns: list[Campaign],
        features: list[dict],
        labels: np.ndarray,
        pctrs: np.ndarray,
        cfg: EngineConfig,
        control_hook: Optional[ControlHook] = None,
        throttle_hook: Optional[ThrottleHook] = None,
        bid_hook: Optional[BidHook] = None,
    ):
        self.campaigns = campaigns
        self.by_id = {c.id: c for c in campaigns}
        self.features = features
        self.labels = labels
        self.pctrs = pctrs
        self.cfg = cfg
        self.control_hook = control_hook
        self.throttle_hook = throttle_hook
        # default: fold the multiplier into the bid (bid-shading behavior)
        self.bid_hook = bid_hook or (
            lambda c, pctr: compute_bid(pctr, c.value_per_click, c.pacing_multiplier)
        )

    def run(self, replay) -> RunStats:
        stats = RunStats()
        stats.spend_trace = {c.id: [] for c in self.campaigns}
        next_control = 0.0

        for pidx, t in replay.stream():
            # fire the controller on its own cadence, not per impression
            while t >= next_control:
                if self.control_hook is not None:
                    self.control_hook(next_control, self.campaigns)
                for c in self.campaigns:
                    stats.spend_trace[c.id].append((next_control, c.spend))
                next_control += self.cfg.control_interval_s

            feats = self.features[pidx]
            pctr = float(self.pctrs[pidx])
            label = int(self.labels[pidx])
            stats.impressions += 1

            bids: list[tuple[int, float]] = []
            for c in self.campaigns:
                if not c.eligible(feats):
                    continue
                if self.throttle_hook is not None and not self.throttle_hook(c, pctr):
                    continue
                bid = self.bid_hook(c, pctr)
                if bid > 0:
                    bids.append((c.id, bid))

            result = run_auction(bids, reserve=self.cfg.reserve)
            if not result.won:
                continue
            winner = self.by_id[result.winner]
            # never let a charge exceed remaining budget (hard invariant)
            cost = min(result.price, winner.remaining())
            winner.charge(cost)
            stats.total_spend += cost
            stats.auctions_with_winner += 1
            if label == 1:
                stats.total_clicks += 1

        return stats
