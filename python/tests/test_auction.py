"""Auction: second-price payment, reserve enforcement, tie handling, empties."""

from pacer.sim.auction import compute_bid, run_auction


def test_compute_bid():
    assert compute_bid(0.1, 2.0, 0.5) == 0.1 * 2.0 * 0.5


def test_second_price_payment():
    r = run_auction([(1, 5.0), (2, 3.0), (3, 1.0)])
    assert r.winner == 1
    assert r.price == 3.0  # pays second-highest


def test_reserve_raises_price():
    r = run_auction([(1, 5.0), (2, 1.0)], reserve=2.0)
    assert r.winner == 1
    assert r.price == 2.0  # reserve beats the second bid


def test_reserve_blocks_low_bids():
    r = run_auction([(1, 1.0), (2, 0.5)], reserve=2.0)
    assert not r.won
    assert r.winner is None


def test_tie_breaks_by_lowest_id_and_pays_bid():
    r = run_auction([(5, 4.0), (2, 4.0)])
    assert r.winner == 2  # lowest id wins the tie
    assert r.price == 4.0  # second-highest equals the top


def test_single_bidder_pays_reserve():
    r = run_auction([(1, 5.0)], reserve=1.5)
    assert r.winner == 1
    assert r.price == 1.5
    # no reserve -> pays 0 (second price with no competition)
    assert run_auction([(1, 5.0)]).price == 0.0


def test_empty():
    r = run_auction([])
    assert not r.won and r.winner is None
