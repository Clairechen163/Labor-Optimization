"""Employee-level weekly roster optimization (mixed-integer linear program).

Decision variable y[e, d, k, r] = 1 if employee e works shift template k starting on day d in role r.

Minimize   sum wage_e * len_k * premium_k * y            (regular pay, incl. night premiums)
         + sum wage_e * (ot_premium - 1) * ot_e          (overtime top-up)
         + call_in * wage_r * sum short[r, t]           (uncovered hours filled by call-in staff)

Subject to (see docs/FORMULATION.md for the full statement)
  coverage        sum_{shifts covering t} y + short[r,t] >= required[r,t]      (inequality)
  floor capacity  sum_{shifts covering t} y <= max_on_duty[r]                  (inequality)
  one shift/day   sum_{k,r} y[e,d,k,r] <= 1
  contract hours  min_hours_e <= hours_e <= max_hours_e   (equality when min == max, e.g. 40h guaranteed)
  max days        sum_{d,k,r} y <= max_days_e
  minimum rest    y[e,d,k] + y[e,d+1,k'] <= 1  when the gap between the two shifts < min_rest
  weekend limit   weekend days worked <= max_weekend_days
  overtime        ot_e >= hours_e - threshold,  0 <= ot_e <= max_overtime
  availability    y is only created where the employee is available and qualified
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import pulp

from .config import HOURS_PER_WEEK, WEEKEND, Scenario, ShiftTemplate


@dataclass
class RosterResult:
    status: str
    objective: float
    assignments: pd.DataFrame     # employee, role, day, template, start, length, hours, cost
    coverage: pd.DataFrame        # role, t, day, hour, required, scheduled, short, surplus
    employees: pd.DataFrame       # employee, primary_role, contract, hours, days, overtime
    summary: dict


def default_solver(time_limit: int = 60, gap: float = 0.005, msg: bool = False):
    try:
        return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, gapRel=gap)
    except Exception:  # pragma: no cover - fall back to HiGHS if CBC is unavailable
        return pulp.HiGHS(msg=msg, timeLimit=time_limit, mip_rel_gap=gap)


def _rest_conflicts(templates: list[ShiftTemplate], min_rest: float) -> list[tuple[str, str]]:
    """Template pairs (k on day d, k2 on day d+1) whose rest gap is shorter than min_rest."""
    out = []
    for a in templates:
        end_a = a.start + a.length
        for b in templates:
            gap = 24 + b.start - end_a
            if gap < min_rest:
                out.append((a.name, b.name))
    return out


def solve_roster(
    scn: Scenario,
    templates: list[str] | None = None,
    cross_training: bool = True,
    hard_coverage: bool = False,
    time_limit: int = 60,
    gap: float = 0.005,
    solver=None,
) -> RosterResult:
    rules = scn.rules
    tpls = [t for t in scn.templates if templates is None or t.name in templates]
    n_days = len(scn.days)
    weekend_idx = [i for i, d in enumerate(scn.days) if d in WEEKEND]
    req = scn.requirements.set_index(["role", "t"])["required"].to_dict()
    roles = list(scn.roles)

    def needed(role: str, d: int, k: ShiftTemplate) -> bool:
        return any(req.get((role, t), 0) > 0 for t in k.covered_hours(d))

    prob = pulp.LpProblem("weekly_roster", pulp.LpMinimize)

    # ---- decision variables (only where the employee is qualified and available) ----
    y: dict[tuple, pulp.LpVariable] = {}
    for e in scn.employees:
        skills = e.skills if cross_training else [e.primary_role]
        for d in range(n_days):
            for k in tpls:
                if not e.can_work(d, k):
                    continue
                for r in skills:
                    if needed(r, d, k):
                        y[(e.id, d, k.name, r)] = pulp.LpVariable(f"y_{e.id}_{d}_{k.name}_{r}", cat="Binary")

    covers: dict[tuple[str, int], list] = {}
    tpl_by_name = {k.name: k for k in tpls}
    for (eid, d, kn, r), var in y.items():
        for t in tpl_by_name[kn].covered_hours(d):
            covers.setdefault((r, t), []).append(var)

    short = {}
    for (r, t), q in req.items():
        if q > 0 and not hard_coverage:
            short[(r, t)] = pulp.LpVariable(f"short_{r}_{t}", lowBound=0)

    emp = {e.id: e for e in scn.employees}
    hours_expr, days_expr, ot = {}, {}, {}
    for e in scn.employees:
        mine = [(key, v) for key, v in y.items() if key[0] == e.id]
        hours_expr[e.id] = pulp.lpSum(tpl_by_name[key[2]].length * v for key, v in mine)
        days_expr[e.id] = pulp.lpSum(v for _, v in mine)
        ot[e.id] = pulp.LpVariable(f"ot_{e.id}", lowBound=0, upBound=float(rules.get("max_overtime_hours", 0)))

    # ---- objective ----
    ot_prem = float(rules.get("overtime_premium", 1.5))
    call_in = float(rules.get("call_in_multiplier", 3.0))
    pay = pulp.lpSum(emp[eid].wage * tpl_by_name[kn].length * tpl_by_name[kn].premium * v
                     for (eid, d, kn, r), v in y.items())
    ot_cost = pulp.lpSum(emp[eid].wage * (ot_prem - 1.0) * v for eid, v in ot.items())
    gap_cost = pulp.lpSum(call_in * scn.roles[r].wage * v for (r, t), v in short.items())
    prob += pay + ot_cost + gap_cost

    # ---- coverage and floor capacity ----
    for (r, t), q in req.items():
        on_duty = pulp.lpSum(covers.get((r, t), []))
        if q > 0:
            prob += (on_duty + short.get((r, t), 0) >= q, f"cover_{r}_{t}")
        cap = scn.roles[r].max_on_duty
        if cap is not None and covers.get((r, t)):
            prob += (on_duty <= cap, f"cap_{r}_{t}")

    # ---- per-employee rules ----
    min_rest = float(rules.get("min_rest_hours", 0))
    conflicts = _rest_conflicts(tpls, min_rest) if min_rest > 0 else []
    threshold = float(rules.get("overtime_threshold", 40))
    max_weekend = rules.get("max_weekend_days")

    by_emp_day: dict[tuple[str, int], list] = {}
    by_emp_day_tpl: dict[tuple[str, int, str], list] = {}
    for (eid, d, kn, r), v in y.items():
        by_emp_day.setdefault((eid, d), []).append(v)
        by_emp_day_tpl.setdefault((eid, d, kn), []).append(v)

    for e in scn.employees:
        for d in range(n_days):
            if len(by_emp_day.get((e.id, d), [])) > 1:
                prob += (pulp.lpSum(by_emp_day[(e.id, d)]) <= 1, f"oneshift_{e.id}_{d}")
        h = hours_expr[e.id]
        if e.min_hours == e.max_hours:
            prob += (h == e.min_hours, f"hours_eq_{e.id}")                 # equality constraint
        else:
            if e.min_hours > 0:
                prob += (h >= e.min_hours, f"hours_min_{e.id}")
            prob += (h <= e.max_hours, f"hours_max_{e.id}")
        prob += (days_expr[e.id] <= e.max_days, f"days_{e.id}")
        prob += (ot[e.id] >= h - threshold, f"ot_{e.id}")
        if max_weekend is not None and weekend_idx:
            wk = [v for d in weekend_idx for v in by_emp_day.get((e.id, d), [])]
            if wk and max_weekend < len(weekend_idx):
                prob += (pulp.lpSum(wk) <= max_weekend, f"weekend_{e.id}")
        for d in range(n_days):
            d2 = (d + 1) % n_days
            for a, b in conflicts:
                va = by_emp_day_tpl.get((e.id, d, a), [])
                vb = by_emp_day_tpl.get((e.id, d2, b), [])
                if va and vb:
                    prob += (pulp.lpSum(va) + pulp.lpSum(vb) <= 1, f"rest_{e.id}_{d}_{a}_{b}")

    # ---- solve ----
    t0 = time.perf_counter()
    prob.solve(solver or default_solver(time_limit, gap))
    runtime = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]
    sol_status = {1: "optimal", 2: "feasible (time limit)", 0: "not solved",
                  -1: "infeasible", -2: "unbounded", -3: "undefined"}.get(prob.sol_status, str(prob.sol_status))

    if prob.status != 1:
        empty = pd.DataFrame()
        return RosterResult(status, float("nan"), empty, empty, empty,
                            {"status": status, "solution": sol_status, "runtime_s": runtime})

    # ---- results ----
    rows = []
    for (eid, d, kn, r), v in y.items():
        if v.value() and v.value() > 0.5:
            k = tpl_by_name[kn]
            rows.append({"employee": eid, "role": r, "primary_role": emp[eid].primary_role,
                         "contract": emp[eid].contract, "day": d, "day_name": scn.days[d],
                         "template": kn, "start": k.start, "length": k.length,
                         "cost": emp[eid].wage * k.length * k.premium})
    assign = pd.DataFrame(rows).sort_values(["day", "start", "role", "employee"]).reset_index(drop=True)

    cov_rows = []
    for r in roles:
        for t in range(HOURS_PER_WEEK):
            q = req.get((r, t), 0)
            sched = sum(v.value() > 0.5 for v in covers.get((r, t), []))
            s = short[(r, t)].value() if (r, t) in short else 0.0
            cov_rows.append({"role": r, "t": t, "day": t // 24, "hour": t % 24, "required": q,
                             "scheduled": int(sched), "short": max(0.0, q - sched),
                             "surplus": max(0, sched - q)})
    cov = pd.DataFrame(cov_rows)

    emp_rows = []
    for e in scn.employees:
        mine = assign[assign.employee == e.id]
        hrs = float(mine.length.sum()) if len(mine) else 0.0
        emp_rows.append({"employee": e.id, "primary_role": e.primary_role, "contract": e.contract,
                         "hours": hrs, "days": len(mine), "overtime": max(0.0, hrs - threshold),
                         "cross_role_shifts": int((mine.role != e.primary_role).sum()) if len(mine) else 0})
    emp_df = pd.DataFrame(emp_rows)

    regular = float(assign.cost.sum())
    ot_total = float(sum(emp[eid].wage * (ot_prem - 1.0) * (v.value() or 0) for eid, v in ot.items()))
    req_hours = float(cov.required.sum())
    staffed_hours = float(cov.scheduled.sum())
    short_hours = float(cov.short.sum())
    # Uncovered hours are still staffed, by call-in staff at call_in_multiplier x wage. The same price
    # is used in the objective, so schedules with gaps are never made to look cheap.
    gap_fill = float(sum(row.short * scn.roles[row.role].wage * call_in for row in cov.itertuples() if row.short))
    summary = {
        "status": status,
        "solution": sol_status,
        "labor_cost": regular + ot_total,
        "call_in_cost": gap_fill,
        "total_cost": regular + ot_total + gap_fill,
        "overtime_cost": ot_total,
        "paid_hours": float(assign.length.sum()),
        "required_staff_hours": req_hours,
        "shortfall_staff_hours": short_hours,
        "surplus_staff_hours": float(cov.surplus.sum()),
        "coverage_rate": 1.0 - short_hours / req_hours if req_hours else 1.0,
        "schedule_efficiency": (req_hours - short_hours) / staffed_hours if staffed_hours else 0.0,
        "cross_role_shifts": int((assign.role != assign.primary_role).sum()),
        "n_variables": len(prob.variables()),
        "n_constraints": len(prob.constraints),
        "runtime_s": runtime,
        "templates": [k.name for k in tpls],
    }
    return RosterResult(status, float(pulp.value(prob.objective)), assign, cov, emp_df, summary)


def compare_policies(scn: Scenario, time_limit: int = 60) -> pd.DataFrame:
    """Value of flexibility: fixed three-shift baseline vs flexible starts, with and without cross-training."""
    runs = {
        "Fixed shifts, no cross-training": dict(templates=scn.baseline_templates, cross_training=False),
        "Fixed shifts + cross-training": dict(templates=scn.baseline_templates, cross_training=True),
        "Flexible starts, no cross-training": dict(templates=None, cross_training=False),
        "Flexible starts + cross-training (optimized)": dict(templates=None, cross_training=True),
    }
    rows = []
    for label, kw in runs.items():
        res = solve_roster(scn, time_limit=time_limit, **kw)
        s = res.summary
        rows.append({"policy": label, "labor_cost": s.get("labor_cost"), "call_in_cost": s.get("call_in_cost"),
                     "total_cost": s.get("total_cost"), "paid_hours": s.get("paid_hours"),
                     "shortfall_staff_hours": s.get("shortfall_staff_hours"),
                     "surplus_staff_hours": s.get("surplus_staff_hours"),
                     "coverage_rate": s.get("coverage_rate"), "schedule_efficiency": s.get("schedule_efficiency"),
                     "solution": s.get("solution"), "runtime_s": s.get("runtime_s")})
    return pd.DataFrame(rows)
