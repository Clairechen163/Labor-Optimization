"""Casino mathematics: exact odds, house edge and payback, verified by Monte Carlo.

Exact values use rational arithmetic (fractions) or full enumeration of the shoe, so they are
reproducible to every digit. Each game also has a simulator; the tests check that simulated edges
fall inside a few standard errors of the exact figure.

House edge = -(expected value per unit wagered). Payback (RTP) = 1 - house edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------ roulette
ROULETTE_BETS = {  # name: (numbers covered, payout x:1)
    "straight": (1, 35), "split": (2, 17), "street": (3, 11), "corner": (4, 8),
    "six_line": (6, 5), "dozen": (12, 2), "column": (12, 2), "red_black": (18, 1),
    "odd_even": (18, 1), "high_low": (18, 1),
}


def roulette_edge(bet: str = "straight", wheel: str = "european") -> Fraction:
    pockets = {"european": 37, "american": 38}[wheel]
    covered, pay = (5, 6) if bet == "top_line" else ROULETTE_BETS[bet]
    ev = Fraction(covered * pay - (pockets - covered), pockets)
    return -ev


def simulate_roulette(n: int, bet: str = "straight", wheel: str = "european", seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pockets = {"european": 37, "american": 38}[wheel]
    covered, pay = ROULETTE_BETS[bet]
    spins = rng.integers(0, pockets, n)
    return np.where(spins < covered, pay, -1).astype(float)


# ------------------------------------------------------------------------------------ craps
def _dice_dist() -> dict[int, Fraction]:
    d: dict[int, Fraction] = {}
    for a, b in product(range(1, 7), repeat=2):
        d[a + b] = d.get(a + b, Fraction(0)) + Fraction(1, 36)
    return d


def craps_pass_line() -> dict:
    """Exact pass-line probabilities: win on 7/11, lose on 2/3/12, else point before 7."""
    d = _dice_dist()
    win = d[7] + d[11]
    for point in (4, 5, 6, 8, 9, 10):
        win += d[point] * d[point] / (d[point] + d[7])
    return {"p_win": win, "p_come_out_win": d[7] + d[11], "p_come_out_loss": d[2] + d[3] + d[12],
            "house_edge": 1 - 2 * win}


def simulate_pass_line(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        s = rng.integers(1, 7) + rng.integers(1, 7)
        if s in (7, 11):
            out[i] = 1
        elif s in (2, 3, 12):
            out[i] = -1
        else:
            while True:
                r = rng.integers(1, 7) + rng.integers(1, 7)
                if r == s:
                    out[i] = 1
                    break
                if r == 7:
                    out[i] = -1
                    break
    return out


# ------------------------------------------------------------------------------------ baccarat
def _banker_draws(banker_total: int, player_third: int | None) -> bool:
    if player_third is None:
        return banker_total <= 5
    if banker_total <= 2:
        return True
    if banker_total == 3:
        return player_third != 8
    if banker_total == 4:
        return 2 <= player_third <= 7
    if banker_total == 5:
        return 4 <= player_third <= 7
    if banker_total == 6:
        return 6 <= player_third <= 7
    return False


@lru_cache(maxsize=8)
def baccarat_probabilities(decks: int = 8) -> dict:
    """Exact Player/Banker/Tie probabilities by enumerating every deal from a fresh shoe."""
    counts0 = [16 * decks] + [4 * decks] * 9          # card values 0..9 (tens and faces count 0)
    total0 = 52 * decks
    pw = bw = tie = 0.0
    c = counts0[:]

    def draw(prob, v, n):
        return prob * c[v] / n

    n0 = total0
    for p1 in range(10):
        if not c[p1]:
            continue
        q1 = c[p1] / n0; c[p1] -= 1
        for b1 in range(10):
            if not c[b1]:
                continue
            q2 = q1 * c[b1] / (n0 - 1); c[b1] -= 1
            for p2 in range(10):
                if not c[p2]:
                    continue
                q3 = q2 * c[p2] / (n0 - 2); c[p2] -= 1
                for b2 in range(10):
                    if not c[b2]:
                        continue
                    q4 = q3 * c[b2] / (n0 - 3); c[b2] -= 1
                    pt, bt = (p1 + p2) % 10, (b1 + b2) % 10
                    if pt >= 8 or bt >= 8 or (pt >= 6 and bt >= 6):
                        outcomes = [(q4, pt, bt)]
                    elif pt >= 6:                       # player stands, banker draws on 0-5
                        outcomes = []
                        if bt <= 5:
                            for b3 in range(10):
                                if c[b3]:
                                    outcomes.append((q4 * c[b3] / (n0 - 4), pt, (bt + b3) % 10))
                        else:
                            outcomes = [(q4, pt, bt)]
                    else:                               # player draws a third card
                        outcomes = []
                        for p3 in range(10):
                            if not c[p3]:
                                continue
                            q5 = q4 * c[p3] / (n0 - 4); c[p3] -= 1
                            pt3 = (pt + p3) % 10
                            if _banker_draws(bt, p3):
                                for b3 in range(10):
                                    if c[b3]:
                                        outcomes.append((q5 * c[b3] / (n0 - 5), pt3, (bt + b3) % 10))
                            else:
                                outcomes.append((q5, pt3, bt))
                            c[p3] += 1
                    for q, a, b in outcomes:
                        if a > b:
                            pw += q
                        elif b > a:
                            bw += q
                        else:
                            tie += q
                    c[b2] += 1
                c[p2] += 1
            c[b1] += 1
        c[p1] += 1
    return {"player": pw, "banker": bw, "tie": tie}


def baccarat_edges(decks: int = 8, banker_commission: float = 0.05, tie_pays: int = 8) -> dict:
    p = baccarat_probabilities(decks)
    return {
        "player": -(p["player"] - p["banker"]),
        "banker": -((1 - banker_commission) * p["banker"] - p["player"]),
        "tie": -(tie_pays * p["tie"] - (1 - p["tie"])),
    }


def simulate_baccarat(n: int, decks: int = 8, seed: int = 0) -> np.ndarray:
    """Outcome codes per hand from a fresh shoe each hand: 1 player win, -1 banker win, 0 tie."""
    rng = np.random.default_rng(seed)
    shoe = np.array(([0] * 16 + list(range(1, 10)) * 4) * decks)
    out = np.empty(n, dtype=int)
    for i in range(n):
        cards = rng.choice(shoe, 6, replace=False)
        p1, b1, p2, b2, x1, x2 = cards
        pt, bt = (p1 + p2) % 10, (b1 + b2) % 10
        if not (pt >= 8 or bt >= 8):
            if pt <= 5:
                p3 = x1
                pt = (pt + p3) % 10
                if _banker_draws(bt, int(p3)):
                    bt = (bt + x2) % 10
            elif bt <= 5:
                bt = (bt + x1) % 10
        out[i] = 1 if pt > bt else (-1 if bt > pt else 0)
    return out


# ------------------------------------------------------------------------------------ blackjack
def blackjack_natural_probability(decks: int = 6) -> float:
    """P(two-card 21): an ace and a ten-value card in either order."""
    n = 52 * decks
    aces, tens = 4 * decks, 16 * decks
    return 2 * (aces / n) * (tens / (n - 1))


def bust_probability(hard_total: int, decks: int = 1, removed: list[int] | None = None) -> float:
    """P(bust) when hitting a hard total once, given cards already seen (values 2-10, ace = 1)."""
    counts = {v: 4 * decks for v in range(1, 10)}
    counts[10] = 16 * decks
    for v in removed or []:
        counts[v] -= 1
    n = sum(counts.values())
    busting = sum(k for v, k in counts.items() if hard_total + v > 21)
    return busting / n


# ------------------------------------------------------------------------------------ poker
def required_equity(pot: float, call: float) -> float:
    """Minimum win probability for a call to break even: call / (pot + call)."""
    return call / (pot + call)


def call_ev(pot: float, call: float, equity: float) -> float:
    """EV of calling: win the pot with probability `equity`, lose the call otherwise."""
    return equity * pot - (1 - equity) * call


# ------------------------------------------------------------------------------------ slots
@dataclass
class Paytable:
    """Discrete slot outcome distribution: payouts (multiples of the stake, total returned) and probabilities."""
    payouts: tuple[float, ...]
    probs: tuple[float, ...]

    def __post_init__(self):
        if abs(sum(self.probs) - 1) > 1e-9:
            raise ValueError("probabilities must sum to 1")

    @property
    def rtp(self) -> float:
        return float(np.dot(self.payouts, self.probs))

    @property
    def house_edge(self) -> float:
        return 1 - self.rtp

    @property
    def hit_frequency(self) -> float:
        return float(sum(p for x, p in zip(self.payouts, self.probs) if x > 0))

    @property
    def sd_per_spin(self) -> float:
        x, p = np.array(self.payouts), np.array(self.probs)
        return float(np.sqrt(np.dot(p, (x - self.rtp) ** 2)))

    def session(self, spins: int, stake: float = 1.0) -> dict:
        """Expected loss and a normal-approximation 90% range for a session of `spins` spins."""
        mean_loss = spins * stake * self.house_edge
        sd = np.sqrt(spins) * stake * self.sd_per_spin
        return {"expected_loss": mean_loss, "sd": sd, "p05_net": -mean_loss - 1.645 * sd,
                "p95_net": -mean_loss + 1.645 * sd}

    def simulate(self, spins: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.choice(np.array(self.payouts), size=spins, p=np.array(self.probs)) - 1.0


def example_paytable(rtp: float = 0.96) -> Paytable:
    """An illustrative paytable tuned to the given RTP (the 3x line absorbs the remainder)."""
    fixed = {1000: 0.00002, 100: 0.0002, 20: 0.003, 10: 0.015, 5: 0.03, 2: 0.12, 1: 0.15}
    p3 = (rtp - sum(k * v for k, v in fixed.items())) / 3
    payouts = tuple(list(fixed) + [3, 0])
    probs = list(fixed.values()) + [p3]
    probs.append(1 - sum(probs))
    return Paytable(payouts, tuple(probs))


# ------------------------------------------------------------------------------------ summary
def game_summary() -> pd.DataFrame:
    craps = craps_pass_line()
    bac = baccarat_edges()
    bacp = baccarat_probabilities()
    rows = [
        ("European roulette", "Straight up (35:1)", 1 / 37, float(roulette_edge("straight"))),
        ("European roulette", "Red/black (1:1)", 18 / 37, float(roulette_edge("red_black"))),
        ("American roulette", "Straight up (35:1)", 1 / 38, float(roulette_edge("straight", "american"))),
        ("American roulette", "Top line 0-00-1-2-3 (6:1)", 5 / 38, float(roulette_edge("top_line", "american"))),
        ("Craps", "Pass line (1:1)", float(craps["p_win"]), float(craps["house_edge"])),
        ("Baccarat (8 decks)", "Banker (0.95:1)", bacp["banker"], bac["banker"]),
        ("Baccarat (8 decks)", "Player (1:1)", bacp["player"], bac["player"]),
        ("Baccarat (8 decks)", "Tie (8:1)", bacp["tie"], bac["tie"]),
        ("Slot (example paytable)", "1 unit per spin", example_paytable().hit_frequency, example_paytable().house_edge),
    ]
    df = pd.DataFrame(rows, columns=["game", "bet", "p_win", "house_edge"])
    df["payback"] = 1 - df.house_edge
    return df
