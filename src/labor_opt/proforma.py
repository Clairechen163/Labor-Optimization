"""Pro-forma analysis of staffing decisions: projected P&L, break-even and uncertainty.

The basic comparison (baseline vs scenario profit) answers "is it worth it if the uplift happens".
Two additions make it decision-grade:

* margin: sales are not profit. Retail uplift is worth sales x gross margin; casino win is taxed
  (gross gaming revenue tax). Ignoring this overstates the case for extra staff.
* uncertainty: the uplift is an estimate. Monte Carlo over its distribution gives the probability
  the decision loses money, and the break-even uplift says how wrong the estimate can be.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StaffingCase:
    name: str
    revenue: float            # baseline revenue per day
    staff: int                # baseline staff
    cost_per_staff: float     # per day
    extra_staff: int
    uplift: float             # expected revenue uplift, e.g. 0.30 = +30%
    margin: float = 1.0       # share of incremental revenue that is contribution (gross margin, net of tax)
    other_costs: float = 0.0  # fixed costs per day, unchanged by the decision
    uplift_sd: float = 0.0    # uncertainty in the uplift estimate

    def baseline_profit(self) -> float:
        return self.revenue * self.margin - self.staff * self.cost_per_staff - self.other_costs

    def scenario_profit(self, uplift: float | None = None) -> float:
        u = self.uplift if uplift is None else uplift
        return self.revenue * (1 + u) * self.margin - (self.staff + self.extra_staff) * self.cost_per_staff - self.other_costs

    def incremental(self, uplift: float | None = None) -> float:
        return self.scenario_profit(uplift) - self.baseline_profit()

    def break_even_uplift(self) -> float:
        return self.extra_staff * self.cost_per_staff / (self.revenue * self.margin)

    def simulate(self, n: int = 20000, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        u = rng.normal(self.uplift, self.uplift_sd, n) if self.uplift_sd > 0 else np.full(n, self.uplift)
        inc = self.revenue * u * self.margin - self.extra_staff * self.cost_per_staff
        return {"expected_incremental": float(inc.mean()), "p_loss": float((inc < 0).mean()),
                "p05": float(np.quantile(inc, 0.05)), "p95": float(np.quantile(inc, 0.95))}

    def summary(self) -> dict:
        sim = self.simulate()
        return {"case": self.name, "baseline_profit": self.baseline_profit(), "scenario_profit": self.scenario_profit(),
                "incremental_profit": self.incremental(), "break_even_uplift": self.break_even_uplift(),
                "expected_uplift": self.uplift, **{f"mc_{k}": v for k, v in sim.items()}}


def sensitivity(case: StaffingCase, uplifts=np.linspace(0, 0.4, 9), extra_staff=(0, 5, 10, 15)) -> pd.DataFrame:
    rows = []
    for e in extra_staff:
        c = StaffingCase(**{**case.__dict__, "extra_staff": e})
        rows.append({"extra_staff": e, **{f"{u:.0%}": c.incremental(u) for u in uplifts}})
    return pd.DataFrame(rows).set_index("extra_staff")


def productivity_gain_case(units_per_day: float, price: float, workers: int, wage_per_day: float,
                           productivity_gain: float, margin: float = 1.0, training_cost_per_day: float = 0.0) -> dict:
    """Textile: same headcount, higher output after training."""
    base_rev = units_per_day * price
    new_rev = units_per_day * (1 + productivity_gain) * price
    labor = workers * wage_per_day
    base = base_rev * margin - labor
    new = new_rev * margin - labor - training_cost_per_day
    return {"baseline_profit": base, "scenario_profit": new, "incremental_profit": new - base,
            "labor_cost_per_unit_before": labor / units_per_day,
            "labor_cost_per_unit_after": labor / (units_per_day * (1 + productivity_gain))}


def case_studies() -> dict:
    """The three original pro-forma examples, as stated (margin = 100%) and with realistic margins.

    The margins are illustrative parameters, not company figures: 65% gross margin for luxury retail,
    and gaming revenue net of a ~40% gross gaming revenue tax for a Macau-style casino.
    """
    textile = productivity_gain_case(1000, 10, 50, 100, 0.10)
    textile_m = productivity_gain_case(1000, 10, 50, 100, 0.10, margin=0.55)
    retail = StaffingCase("Retail promotion (as stated)", 50000, 10, 200, 5, 0.30, uplift_sd=0.10)
    retail_m = StaffingCase("Retail promotion (65% gross margin)", 50000, 10, 200, 5, 0.30, margin=0.65, uplift_sd=0.10)
    casino = StaffingCase("Casino event (as stated)", 100000, 50, 300, 10, 0.20, uplift_sd=0.08)
    casino_m = StaffingCase("Casino event (net of 40% gaming tax)", 100000, 50, 300, 10, 0.20, margin=0.60, uplift_sd=0.08)
    return {"textile": textile, "textile_margin": textile_m,
            "cases": pd.DataFrame([c.summary() for c in (retail, retail_m, casino, casino_m)])}
