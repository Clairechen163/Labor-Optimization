"""Labor demand models: turn a business driver (units, customers, players) into staff required.

Three standard approaches, one per operating environment:

* workload model (manufacturing): workers = target / (units per hour x hours x efficiency)
* Erlang C queueing model (retail service desk, cashier, guest services): the smallest number of
  servers that meets a service level such as "80% of customers served within 2 minutes"
* table-game model (casino): players -> open tables at a target occupancy -> dealers incl. relief
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------- manufacturing workload model
def workload_staff(target_units: float, units_per_hour: float, hours_per_worker: float,
                   efficiency: float = 1.0) -> dict:
    """Workers needed to produce `target_units` in a period.

    Example: 100,000 garments/month, 25 garments/hour, 160 hours/worker, 90% efficiency -> 27.8 -> 28.
    """
    if min(units_per_hour, hours_per_worker, efficiency) <= 0:
        raise ValueError("productivity, hours and efficiency must be positive")
    exact = target_units / (units_per_hour * hours_per_worker * efficiency)
    return {"exact": exact, "workers": math.ceil(exact - 1e-9)}


# ---------------------------------------------------------------- Erlang C (M/M/N queue)
def erlang_c(servers: int, load: float) -> float:
    """Probability an arriving customer has to wait (Erlang C), for offered load A = lambda * AHT."""
    if servers <= load:
        return 1.0
    # Erlang B by the stable recursion, then convert to C.
    b = 1.0
    for n in range(1, servers + 1):
        b = load * b / (n + load * b)
    rho = load / servers
    return b / (1 - rho + rho * b)


def service_level(servers: int, arrivals_per_hour: float, handle_minutes: float, target_wait_minutes: float) -> float:
    """P(wait <= target) for an M/M/N queue."""
    load = arrivals_per_hour * handle_minutes / 60.0
    if servers <= load:
        return 0.0
    pw = erlang_c(servers, load)
    return 1.0 - pw * math.exp(-(servers - load) * target_wait_minutes / handle_minutes)


def erlang_staff(arrivals_per_hour: float, handle_minutes: float, target_wait_minutes: float,
                 target_service_level: float, shrinkage: float = 0.0, max_servers: int = 500) -> dict:
    """Smallest number of staff meeting the service-level target, before and after shrinkage.

    Shrinkage is the share of paid time not available to serve (breaks, training, admin).
    """
    load = arrivals_per_hour * handle_minutes / 60.0
    n = max(1, math.floor(load) + 1)
    while n <= max_servers:
        sl = service_level(n, arrivals_per_hour, handle_minutes, target_wait_minutes)
        if sl >= target_service_level:
            pw = erlang_c(n, load)
            asa = pw * handle_minutes / (n - load)  # average speed of answer (minutes)
            return {"offered_load": load, "servers": n, "service_level": sl, "prob_wait": pw,
                    "avg_wait_minutes": asa, "occupancy": load / n,
                    "scheduled": math.ceil(n / (1 - shrinkage) - 1e-9)}
        n += 1
    raise ValueError("service level not reachable within max_servers")


# ---------------------------------------------------------------- casino table-game model
def table_game_dealers(players: float | np.ndarray, seats_per_table: int = 7, target_occupancy: float = 0.8,
                       relief_ratio: float = 1.25, min_tables: int = 1, max_tables: int | None = None) -> dict:
    """Dealers needed on duty for a pit.

    tables  = ceil(players / (seats_per_table * target_occupancy)), clipped to [min_tables, max_tables]
    dealers = ceil(tables * relief_ratio)   (relief dealers rotate in so each dealer gets regular breaks;
                                             1.25 means 5 dealers for every 4 tables)
    """
    players = np.asarray(players, dtype=float)
    tables = np.ceil(players / (seats_per_table * target_occupancy) - 1e-9)
    tables = np.maximum(tables, min_tables)
    if max_tables is not None:
        tables = np.minimum(tables, max_tables)
    dealers = np.ceil(tables * relief_ratio - 1e-9)
    if players.ndim == 0:
        return {"tables": int(tables), "dealers": int(dealers)}
    return {"tables": tables.astype(int), "dealers": dealers.astype(int)}


def theoretical_win_per_hour(hands_per_hour: float, avg_players: float, avg_bet: float, house_edge: float) -> float:
    """Expected casino win per table-hour: hands x players x average bet x house edge.

    This is the link between labor productivity (hands dealt per hour) and revenue.
    """
    return hands_per_hour * avg_players * avg_bet * house_edge
