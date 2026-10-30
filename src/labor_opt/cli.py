"""Command-line interface: `labor-opt <command>` (or `python -m labor_opt <command>`)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_scenario, scenario_path


def _fmt(summary: dict) -> str:
    keys = ["solution", "labor_cost", "call_in_cost", "total_cost", "paid_hours", "required_staff_hours",
            "shortfall_staff_hours", "coverage_rate", "schedule_efficiency", "cross_role_shifts", "runtime_s"]
    out = []
    for k in keys:
        v = summary.get(k)
        if isinstance(v, float):
            v = f"{v:.1%}" if k in ("coverage_rate", "schedule_efficiency") else f"{v:,.1f}"
        out.append(f"  {k:<24}{v}")
    return "\n".join(out)


def cmd_schedule(a):
    from .roster import solve_roster
    scn = load_scenario(scenario_path(a.scenario))
    res = solve_roster(scn, templates=scn.baseline_templates if a.fixed else None,
                       cross_training=not a.no_cross_training, time_limit=a.time_limit)
    print(f"{scn.name}: {len(scn.employees)} employees, {len(scn.templates)} shift templates")
    print(_fmt(res.summary))
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        res.assignments.to_csv(out / "assignments.csv", index=False)
        res.coverage.to_csv(out / "coverage.csv", index=False)
        res.employees.to_csv(out / "employee_hours.csv", index=False)
        print(f"wrote assignments.csv, coverage.csv, employee_hours.csv to {out}")


def cmd_compare(a):
    from .roster import compare_policies
    scn = load_scenario(scenario_path(a.scenario))
    tab = compare_policies(scn, time_limit=a.time_limit)
    with pd.option_context("display.width", 160, "display.float_format", "{:,.1f}".format):
        print(tab[["policy", "labor_cost", "call_in_cost", "total_cost", "shortfall_staff_hours", "schedule_efficiency"]])


def cmd_forecast(a):
    from .pipeline import forecast_driven_plan
    plan = forecast_driven_plan(a.scenario, time_limit=a.time_limit)
    with pd.option_context("display.width", 160, "display.float_format", "{:,.3f}".format):
        print(plan.backtest)
    print(f"\nStaffing quantile (newsvendor): {plan.quantile:.3f}")
    print(_fmt(plan.roster.summary))


def cmd_casino_math(a):
    from .casino_math import game_summary
    with pd.option_context("display.width", 160, "display.float_format", "{:.5f}".format):
        print(game_summary())


def cmd_etl(a):
    from .etl import run_etl
    from .roster import solve_roster
    scn = load_scenario(scenario_path(a.scenario))
    roster = solve_roster(scn, time_limit=a.time_limit)
    rep = run_etl(scn, roster, a.db, a.out, weeks=a.weeks)
    print(f"loaded {rep.db_path}: {rep.rows_loaded}")
    print(f"data quality: {rep.quality}")
    print(f"KPI extracts written to {a.out}")


def cmd_report(a):
    from .report import build_report
    build_report(a.out, time_limit=a.time_limit)
    print(f"wrote {Path(a.out) / 'RESULTS.md'} and figures")


def main(argv=None):
    p = argparse.ArgumentParser(prog="labor-opt", description="Labor scheduling and workforce analytics")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("schedule", help="optimize a weekly roster")
    s.add_argument("scenario", help="casino | textile | retail | path/to/config.yaml")
    s.add_argument("--fixed", action="store_true", help="use only the fixed baseline shifts")
    s.add_argument("--no-cross-training", action="store_true")
    s.add_argument("--out", help="folder for CSV output")
    s.set_defaults(func=cmd_schedule)

    c = sub.add_parser("compare", help="fixed vs flexible shifts, with and without cross-training")
    c.add_argument("scenario")
    c.set_defaults(func=cmd_compare)

    f = sub.add_parser("forecast", help="forecast pit traffic, derive dealer requirement, roster it")
    f.add_argument("scenario", nargs="?", default="casino")
    f.set_defaults(func=cmd_forecast)

    m = sub.add_parser("casino-math", help="house edge and payback table")
    m.set_defaults(func=cmd_casino_math)

    e = sub.add_parser("etl", help="build the SQLite labor warehouse and KPI extracts")
    e.add_argument("--scenario", default="casino")
    e.add_argument("--db", default="data/warehouse.db")
    e.add_argument("--out", default="reports/etl")
    e.add_argument("--weeks", type=int, default=4)
    e.set_defaults(func=cmd_etl)

    r = sub.add_parser("report", help="run all case studies, write reports/RESULTS.md and figures")
    r.add_argument("--out", default="reports")
    r.set_defaults(func=cmd_report)

    for sp in (s, c, f, e, r):
        sp.add_argument("--time-limit", type=int, default=60, help="solver time limit in seconds")

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
