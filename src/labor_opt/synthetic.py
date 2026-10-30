"""Synthetic operating data with known ground truth.

Every dataset here is simulated with a fixed seed. No real company data is used. Simulating lets the
analyses be checked against the true effects that generated the data (e.g. an experiment estimate
can be compared with the effect that was actually planted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Relative pit traffic by hour of day (casino floor: quiet before dawn, peak late evening).
HOURLY_PROFILE = np.array([
    0.80, 0.70, 0.55, 0.42, 0.35, 0.33, 0.38, 0.48, 0.58, 0.66, 0.72, 0.78,
    0.84, 0.88, 0.92, 0.95, 0.98, 1.05, 1.15, 1.25, 1.32, 1.35, 1.25, 1.05,
])
DOW_EFFECT = np.array([0.92, 0.90, 0.93, 0.97, 1.12, 1.22, 1.04])


def pit_traffic(weeks: int = 30, base_players: float = 19.0, seed: int = 42,
                event_days: int = 6, dispersion: float = 0.03) -> pd.DataFrame:
    """Hourly players seated in a blackjack pit, with weekly seasonality, events and a mild trend.

    Counts are negative-binomial (overdispersed Poisson); `dispersion` is the NB2 alpha.
    """
    rng = np.random.default_rng(seed)
    n = weeks * 168
    ts = pd.date_range("2025-01-06", periods=n, freq="h")  # starts on a Monday
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    week = np.arange(n) // 168
    days = np.arange(n) // 24
    ev_days = rng.choice(np.arange(7, weeks * 7), size=event_days, replace=False)
    event = np.isin(days, ev_days).astype(int)
    trend = 1 + 0.004 * week
    peak_evening = ((dow >= 4) & (dow <= 5) & (hour >= 19)).astype(float)
    mu = base_players * HOURLY_PROFILE[hour] * DOW_EFFECT[dow] * trend * (1 + 0.35 * event) * (1 + 0.08 * peak_evening)
    # NB2 via gamma-Poisson mixture
    lam = rng.gamma(shape=1 / dispersion, scale=mu * dispersion)
    players = rng.poisson(lam)
    return pd.DataFrame({"timestamp": ts, "week": week, "dow": dow, "day": [DAYS[d] for d in dow],
                         "hour": hour, "event": event, "players": players, "true_mean": mu})


def dealer_experiment(weeks: int = 8, tables: int = 12, effect_hands: float = 6.0, seed: int = 5) -> pd.DataFrame:
    """Table-shift panel for testing +15% dealers at peak (more relief dealers -> faster game).

    Treatment is randomized at the table-day level. Hands/hour falls with crowding (players per table)
    and with fatigue late in the shift; the planted treatment effect is `effect_hands`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    table_skill = rng.normal(0, 3, tables)
    for day in range(weeks * 7):
        dow = day % 7
        for tb in range(tables):
            for shift in ("off_peak", "peak"):
                treated = int(rng.random() < 0.5) if shift == "peak" else 0
                players = rng.poisson(5.8 if shift == "peak" else 3.6) + 1
                players = min(players, 7)
                game = "blackjack" if tb < 8 else "baccarat"
                base = 72 - 3.0 * players + (4 if game == "baccarat" else 0) + table_skill[tb] - (2 if dow in (4, 5) else 0)
                hands = base + effect_hands * treated + rng.normal(0, 7)
                sat = np.clip(np.round(6.8 + 0.06 * (hands - 55) + 0.35 * treated + rng.normal(0, 1.1)), 1, 10)
                rows.append({"day": day, "dow": DAYS[dow], "table": f"T{tb + 1:02d}", "game": game,
                             "shift": shift, "treated": treated, "players": players,
                             "hands_per_hour": round(hands, 1), "satisfaction": int(sat)})
    return pd.DataFrame(rows)


def textile_workflow_panel(lines: int = 8, weeks_before: int = 8, weeks_after: int = 6,
                           effect_uph: float = 1.5, seed: int = 9) -> pd.DataFrame:
    """Weekly production-line panel for a difference-in-differences test of a new workflow.

    Half the lines adopt the workflow at `weeks_before`. All lines share a seasonal demand swing
    (common shock) that a naive before/after comparison wrongly attributes to the workflow.
    """
    rng = np.random.default_rng(seed)
    rows = []
    line_fx = rng.normal(0, 1.2, lines)
    weeks = weeks_before + weeks_after
    # Common ramp shared by every line (seasonal volume plus learning). It lifts productivity on all
    # lines after the switch date, which a before/after test on treated lines wrongly credits to the change.
    common = np.linspace(0, 2.0, weeks)
    for ln in range(lines):
        treated_line = int(ln < lines // 2)
        for w in range(weeks):
            post = int(w >= weeks_before)
            d = treated_line * post
            uph = 25 + line_fx[ln] + common[w] + effect_uph * d + rng.normal(0, 0.7)
            labor_hours = rng.normal(320, 12)
            units = uph * labor_hours
            p_def = 0.055 - 0.012 * d + 0.004 * rng.standard_normal()
            defects = rng.binomial(int(units), np.clip(p_def, 0.005, 0.2))
            rows.append({"line": f"L{ln + 1}", "week": w, "treated_line": treated_line, "post": post,
                         "workflow_new": d, "units": round(units), "labor_hours": round(labor_hours, 1),
                         "units_per_labor_hour": round(uph, 3), "defects": defects,
                         "defect_rate": defects / units})
    return pd.DataFrame(rows)


def retail_staffing_test(stores: int = 10, weeks: int = 8, effect_logit: float = 0.12, seed: int = 21) -> pd.DataFrame:
    """Store-day data for an A/B test of dynamic (traffic-based) staffing on conversion."""
    rng = np.random.default_rng(seed)
    rows = []
    store_fx = rng.normal(0, 0.15, stores)
    for s in range(stores):
        treated = int(s < stores // 2)
        for day in range(weeks * 7):
            dow = day % 7
            visitors = rng.poisson(420 * (1.35 if dow >= 5 else 1.0) * np.exp(store_fx[s]))
            post = int(day >= weeks * 7 // 2)
            logit = -1.45 + store_fx[s] - 0.25 * (dow >= 5) + effect_logit * treated * post
            p = 1 / (1 + np.exp(-logit))
            buyers = rng.binomial(visitors, p)
            atv = rng.normal(9800, 900)
            rows.append({"store": f"S{s + 1:02d}", "day": day, "dow": DAYS[dow], "weekend": int(dow >= 5),
                         "treated_store": treated, "post": post, "dynamic_staffing": treated * post,
                         "visitors": visitors, "transactions": buyers,
                         "conversion": buyers / visitors, "sales": round(buyers * atv)})
    return pd.DataFrame(rows)


def textile_daily_kpis(lines: int = 6, days: int = 90, seed: int = 4, shift_day: int = 60,
                       bad_line: str = "L4") -> pd.DataFrame:
    """Daily line KPIs. Line `bad_line` develops tool wear from `shift_day` (defects and downtime creep up)."""
    rng = np.random.default_rng(seed)
    rows = []
    for ln in range(lines):
        name = f"L{ln + 1}"
        for d in range(days):
            wear = max(0, d - shift_day) / 30 if name == bad_line else 0.0
            labor_hours = rng.normal(160, 6)
            downtime = np.clip(rng.normal(0.045 + 0.06 * wear, 0.012), 0, 0.5)
            uph = rng.normal(25.5, 1.2) * (1 - 0.6 * (downtime - 0.045))
            units = int(uph * labor_hours)
            p = 0.018 + 0.03 * wear
            defects = rng.binomial(units, p)
            rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d), "line": name,
                         "units": units, "labor_hours": round(labor_hours, 1), "defects": defects,
                         "downtime_rate": round(downtime, 4),
                         "fulfillment_days": round(rng.normal(6.5 + 4 * wear, 0.8), 1)})
    return pd.DataFrame(rows)


def retail_daily_kpis(stores: int = 5, days: int = 60, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(stores):
        name = f"S{s + 1:02d}"
        roadworks = name == "S03"
        for d in range(days):
            dow = d % 7
            weekend = dow >= 5
            visitors = rng.poisson(450 * (1.3 if weekend else 1.0) * (0.7 if roadworks and d >= 40 else 1.0))
            conv = np.clip(rng.normal(0.205 if not (name == "S05" and d >= 35) else 0.13, 0.02), 0.02, 0.6)
            tx = rng.binomial(visitors, conv)
            sales = tx * rng.normal(1400, 120)
            labor_hours = rng.normal(150 if weekend else 120, 5)
            rows.append({"date": pd.Timestamp("2026-03-01") + pd.Timedelta(days=d), "store": name,
                         "visitors": visitors, "transactions": tx, "sales": round(sales, 0),
                         "labor_hours": round(labor_hours, 1),
                         "avg_transaction_minutes": round(rng.normal(5.0 if name != "S02" else 9.5, 0.6), 2),
                         "nps": round(rng.normal(78 if name != "S01" else 63, 4), 1)})
    return pd.DataFrame(rows)


def casino_shift_kpis(dealers: int = 10, shifts: int = 28, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(dealers):
        name = f"DLR-{i + 1:02d}"
        skill = rng.normal(0, 3)
        slow = name == "DLR-07"
        for s in range(shifts):
            target = 70
            hands = rng.normal(target + skill - (11 if slow and s >= 20 else 0), 3.5)
            rows.append({"shift": s, "dealer": name, "game": "blackjack", "hands_per_hour": round(hands, 1),
                         "utilization": round(np.clip(rng.normal(0.86 - (0.2 if name == "DLR-09" else 0), 0.04), 0, 1), 3),
                         "revenue_per_labor_hour": round(rng.normal(540 - (160 if name == "DLR-09" else 0), 40), 0)})
    return pd.DataFrame(rows)
