from fractions import Fraction

import numpy as np
import pytest

from labor_opt import casino_math as cm


def test_roulette_exact():
    assert cm.roulette_edge("straight") == Fraction(1, 37)
    assert cm.roulette_edge("red_black") == Fraction(1, 37)
    assert cm.roulette_edge("straight", "american") == Fraction(2, 38)
    assert float(cm.roulette_edge("top_line", "american")) == pytest.approx(0.078947, abs=1e-6)


def test_craps_exact():
    r = cm.craps_pass_line()
    assert r["p_win"] == Fraction(244, 495)
    assert r["house_edge"] == Fraction(7, 495)
    assert r["p_come_out_win"] == Fraction(2, 9)
    assert r["p_come_out_loss"] == Fraction(1, 9)


def test_baccarat_published_values():
    p = cm.baccarat_probabilities(8)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-12)
    e = cm.baccarat_edges(8)
    assert e["banker"] == pytest.approx(0.010579, abs=2e-6)
    assert e["player"] == pytest.approx(0.012351, abs=2e-6)
    assert e["tie"] == pytest.approx(0.143596, abs=2e-6)


def _within(sample_mean, exact, sd, n, k=4.0):
    return abs(sample_mean - exact) < k * sd / np.sqrt(n)


def test_monte_carlo_agrees_with_exact():
    n = 60_000
    r = cm.simulate_roulette(n, "straight", seed=1)
    assert _within(-r.mean(), float(cm.roulette_edge("straight")), r.std(), n)
    c = cm.simulate_pass_line(n, seed=2)
    assert _within(-c.mean(), float(cm.craps_pass_line()["house_edge"]), c.std(), n)
    b = cm.simulate_baccarat(n, seed=3)
    p = cm.baccarat_probabilities()
    for code, key in ((1, "player"), (-1, "banker"), (0, "tie")):
        x = (b == code).astype(float)
        assert _within(x.mean(), p[key], x.std(), n)


def test_blackjack():
    assert cm.blackjack_natural_probability(1) == pytest.approx(2 * 4 / 52 * 16 / 51)
    assert cm.bust_probability(16, decks=1000) == pytest.approx(8 / 13, abs=1e-3)
    assert cm.bust_probability(16, 1, [10, 6, 7]) == pytest.approx(29 / 49)
    assert cm.bust_probability(11) == 0


def test_poker_pot_odds():
    assert cm.required_equity(100, 20) == pytest.approx(1 / 6)
    assert cm.call_ev(100, 20, 9 / 46) > 0
    assert cm.call_ev(100, 20, cm.required_equity(100, 20)) == pytest.approx(0)


def test_slot_paytable():
    pt = cm.example_paytable(0.96)
    assert pt.rtp == pytest.approx(0.96)
    assert pt.session(1000)["expected_loss"] == pytest.approx(40)
    sim = pt.simulate(200_000, seed=4)
    assert _within(-sim.mean(), pt.house_edge, pt.sd_per_spin, len(sim))
