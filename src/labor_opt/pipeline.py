"""End-to-end planning run: forecast -> staffing requirement -> optimized roster -> comparison."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Scenario, load_scenario, scenario_path
from .forecast import backtest, forecast_requirements, summarize_backtest
from .roster import RosterResult, solve_roster
from .synthetic import pit_traffic

# Table-game assumptions for the casino pit: 7-seat tables at 80% target occupancy, 2-8 tables open,
# 5 dealers per 4 open tables so every dealer gets regular breaks.
PIT = {"seats_per_table": 7, "target_occupancy": 0.8, "relief_ratio": 1.25, "min_tables": 2, "max_tables": 8}


def newsvendor_quantile(cost_under: float, cost_over: float) -> float:
    """Optimal staffing quantile when an understaffed hour costs `cost_under` and an idle one `cost_over`."""
    return cost_under / (cost_under + cost_over)


@dataclass
class PlanResult:
    scenario: Scenario
    requirements: pd.DataFrame
    backtest: pd.DataFrame
    roster: RosterResult
    quantile: float


def forecast_driven_plan(config: str = "casino", weeks: int = 30, seed: int = 42, time_limit: int = 60) -> PlanResult:
    scn = load_scenario(scenario_path(config))
    call_in = float(scn.rules.get("call_in_multiplier", 3.0))
    q = newsvendor_quantile(cost_under=call_in - 1.0, cost_over=1.0)   # an uncovered hour costs (m-1) wages extra
    traffic = pit_traffic(weeks=weeks, seed=seed)
    bt = summarize_backtest(backtest(traffic, staffing_quantile=q, dealer_kw=PIT))
    req = forecast_requirements(traffic, quantile=q, dealer_kw=PIT)
    planned = scn.with_requirements(req)
    roster = solve_roster(planned, time_limit=time_limit)
    return PlanResult(planned, req, bt, roster, q)
