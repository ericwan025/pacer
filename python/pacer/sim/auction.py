"""Second-price auction with a reserve.

The auction is scaffolding, not the point of the project — kept deliberately
simple. For each impression:

  1. eligible campaigns each submit  bid = pCTR * value_per_click * multiplier
  2. the highest bid wins
  3. the winner pays  max(second_highest_bid, reserve)
  4. if the highest bid is below the reserve, nobody wins

The click outcome is NOT decided here — it is sampled from the TRUE label by the
engine. Using pCTR to sample clicks would hide the model's own errors.
"""

from __future__ import annotations

from dataclasses import dataclass


def compute_bid(pctr: float, value_per_click: float, multiplier: float) -> float:
    """The core equation. One place, so it can't drift."""
    return pctr * value_per_click * multiplier


@dataclass
class AuctionResult:
    winner: int | None
    price: float
    won: bool


def run_auction(bids: list[tuple[int, float]], reserve: float = 0.0) -> AuctionResult:
    """bids: list of (campaign_id, bid_amount). Returns the winner + clearing price.

    Ties on the top bid are broken by lowest campaign id (deterministic). A tie
    for the top means the second-highest bid equals the top, so the winner pays
    its own bid (capped by reserve from below) — standard second-price behavior.
    """
    if not bids:
        return AuctionResult(None, 0.0, False)

    # sort by (bid desc, id asc) for deterministic tie-breaking
    ordered = sorted(bids, key=lambda b: (-b[1], b[0]))
    top_id, top_bid = ordered[0]

    if top_bid < reserve:
        return AuctionResult(None, 0.0, False)

    second_bid = ordered[1][1] if len(ordered) > 1 else 0.0
    price = max(second_bid, reserve)
    return AuctionResult(winner=top_id, price=price, won=True)
