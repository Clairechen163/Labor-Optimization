"""Aggregate shift-covering model (the classic labor-scheduling LP).

x[r, d, k] = number of role-r staff starting template k on day d (continuous for the LP relaxation,
integer for the staffing plan).

    min  sum_{r,d,k} wage_r * len_k * premium_k * x[r,d,k]  +  sum_{r,t} call_in * wage_r * short[r,t]
    s.t. sum_{(d,k) covering t} x[r,d,k] + short[r,t] >= required[r,t]   for every role r and week-hour t
         sum_{(d,k) covering t} x[r,d,k] <= max_on_duty[r]     (floor capacity, optional)
         x >= 0

It ignores who works which shift (availability, contract hours, rest rules), so its optimum is a
lower bound on the cost of any roster built by `roster.solve_roster` with the same call-in pricing. The dual
value of each coverage constraint is the marginal cost of requiring one more person on duty in that
hour: the number a manager needs when asked "what does it cost to add one dealer at 9pm on Friday?".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp

from .config import Scenario


@dataclass
class CoveringResult:
    cost: float
    plan: pd.DataFrame          # role, day, template, start, length, staff
    marginal_cost: pd.DataFrame  # role, t, day, hour, required, marginal_cost (LP only)
    integer: bool


def solve_covering(scn: Scenario, templates: list[str] | None = None, integer: bool = False,
                   hard_coverage: bool = False, solver=None) -> CoveringResult:
    tpls = [t for t in scn.templates if templates is None or t.name in templates]
    req = scn.requirements.set_index(["role", "t"])["required"].to_dict()
    cat = "Integer" if integer else "Continuous"

    prob = pulp.LpProblem("shift_covering", pulp.LpMinimize)
    x = {(r, d, k.name): pulp.LpVariable(f"x_{r}_{d}_{k.name}", lowBound=0, cat=cat)
         for r in scn.roles for d in range(len(scn.days)) for k in tpls}

    call_in = float(scn.rules.get("call_in_multiplier", 3.0))
    short = {} if hard_coverage else {
        (r, t): pulp.LpVariable(f"short_{r}_{t}", lowBound=0) for (r, t), q in req.items() if q > 0}
    prob += (pulp.lpSum(scn.roles[r].wage * k.length * k.premium * x[(r, d, k.name)]
                        for r in scn.roles for d in range(len(scn.days)) for k in tpls)
             + pulp.lpSum(call_in * scn.roles[r].wage * v for (r, t), v in short.items()))

    covering: dict[tuple[str, int], list] = {}
    for (r, d, kn), v in x.items():
        k = next(t for t in tpls if t.name == kn)
        for t in k.covered_hours(d):
            covering.setdefault((r, t), []).append(v)

    for (r, t), q in req.items():
        if q > 0:
            prob += (pulp.lpSum(covering.get((r, t), [])) + short.get((r, t), 0) >= q, f"cover_{r}_{t}")
        cap = scn.roles[r].max_on_duty
        if cap is not None and covering.get((r, t)):
            prob += (pulp.lpSum(covering[(r, t)]) <= cap, f"cap_{r}_{t}")

    prob.solve(solver or pulp.PULP_CBC_CMD(msg=False))
    if prob.status != 1:
        raise RuntimeError(f"covering model status: {pulp.LpStatus[prob.status]}")

    plan = []
    for (r, d, kn), v in x.items():
        val = v.value() or 0.0
        if val > 1e-6:
            k = next(t for t in tpls if t.name == kn)
            plan.append({"role": r, "day": d, "day_name": scn.days[d], "template": kn,
                         "start": k.start, "length": k.length, "staff": round(val, 3)})
    plan_df = pd.DataFrame(plan)

    mc = []
    if not integer:
        for (r, t), q in req.items():
            c = prob.constraints.get(f"cover_{r}_{t}")
            mc.append({"role": r, "t": t, "day": t // 24, "hour": t % 24, "required": q,
                       "marginal_cost": float(c.pi) if c is not None and c.pi is not None else 0.0})
    return CoveringResult(float(pulp.value(prob.objective)), plan_df, pd.DataFrame(mc), integer)


def block_marginal_costs(scn: Scenario, role: str, extra: int = 1, templates: list[str] | None = None
                         ) -> pd.DataFrame:
    """Cost of requiring `extra` more staff of `role` throughout one demand block, by re-solving.

    LP shadow prices are degenerate for covering problems (a whole shift's cost lands on one hour),
    so the planning question "what does one more dealer on Friday evening cost?" is answered here by
    re-solving the integer covering model with the block requirement raised.
    """
    base = solve_covering(scn, templates=templates, integer=True).cost
    blocks = scn.raw["blocks"]
    cap = scn.roles[role].max_on_duty
    rows = []
    for d, day in enumerate(scn.days):
        if day not in scn.operating_days:
            continue
        for block, (b0, b1) in blocks.items():
            req = scn.requirements.copy()
            mask = (req.role == role) & (req.day == d) & (req.hour >= b0) & (req.hour < b1) & (req.required > 0)
            if not mask.any():
                continue
            at_cap = cap is not None and int(req.loc[mask, "required"].max()) + extra > cap
            req.loc[mask, "required"] += extra
            alt = scn.with_requirements(req[req.role == role])
            cost = solve_covering(alt, templates=templates, integer=True).cost
            rows.append({"day": day, "block": block, "hours": f"{b0:02d}-{b1:02d}",
                         "marginal_cost": cost - base, "at_floor_capacity": at_cap})
    return pd.DataFrame(rows)


def marginal_cost_table(res: CoveringResult, scn: Scenario, role: str) -> pd.DataFrame:
    """Day x hour table of the marginal cost of one more required staff member (shadow prices)."""
    df = res.marginal_cost[res.marginal_cost.role == role]
    tab = df.pivot(index="day", columns="hour", values="marginal_cost")
    tab.index = [scn.days[i] for i in tab.index]
    return tab
