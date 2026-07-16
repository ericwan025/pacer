"""Two ways to apply the controller's multiplier.

* THROTTLE mode: the multiplier is a participation probability in [0,1]. The
  campaign enters the auction with probability p and, when it does, bids its full
  value. Preserves win-rate-per-participation; gives up volume.

* BID-SHADE mode: the multiplier scales the bid directly. The campaign always
  enters and bids pCTR * value * multiplier. Keeps volume; gives up the top of
  the auction.

Opinion (see README): throttle is better when you care about not distorting the
auction's price signal and can tolerate lumpier delivery; bid-shading is better
when inventory is scarce and you want to stay in every auction. Throttle also
composes with stratified dropping (Phase 5) to shed low-value impressions first.
"""

from __future__ import annotations

from pacer.sim.auction import compute_bid
from pacer.sim.campaign import Campaign


class BidShadeMode:
    name = "bid_shade"

    def bid_hook(self, c: Campaign, pctr: float) -> float:
        return compute_bid(pctr, c.value_per_click, c.pacing_multiplier)

    def throttle_hook(self, rng):
        return None  # always participate


class ThrottleMode:
    name = "throttle"

    def bid_hook(self, c: Campaign, pctr: float) -> float:
        # full-value bid; the multiplier gates participation instead of the bid
        return compute_bid(pctr, c.value_per_click, 1.0)

    def throttle_hook(self, rng):
        def hook(c: Campaign, pctr: float) -> bool:
            return rng.random() < c.pacing_multiplier

        return hook
