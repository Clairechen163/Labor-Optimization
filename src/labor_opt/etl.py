"""Scheduled ETL: optimized roster + time-clock feed -> SQLite labor warehouse -> KPI extracts.

Extract   the weekly roster from the optimizer and a simulated time-clock feed (with the usual
          problems: duplicate swipes, missing clock-outs, swipes for the wrong shift)
Transform de-duplicate, validate, impute missing clock-outs, compute worked hours and weekly payroll
Load      into the schema in sql/schema.sql, then read the reporting views

Run it with `labor-opt etl --scenario casino`; .github/workflows/scheduled-etl.yml runs it weekly.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Scenario
from .roster import RosterResult

FIRST = ["Ada", "Ben", "Chloe", "Dev", "Elena", "Felix", "Grace", "Hugo", "Ivy", "Jun", "Kai", "Lena", "Marco",
         "Nina", "Omar", "Pia", "Quinn", "Ravi", "Sara", "Tom", "Uma", "Vic", "Wen", "Xin", "Yara", "Zed"]
LAST = ["Lam", "Wong", "Silva", "Chan", "Costa", "Ho", "Leung", "Santos", "Ng", "Lei", "Cheung", "Sousa", "Tam"]


@dataclass
class EtlReport:
    db_path: Path
    rows_loaded: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    kpis: dict = field(default_factory=dict)


# ------------------------------------------------------------------ extract
def extract(scn: Scenario, roster: RosterResult, weeks: int = 4, start: str = "2026-01-05", seed: int = 1,
            absence_rate: float = 0.03) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    roles = pd.DataFrame([{"role_id": r, "role_name": r.replace("_", " ").title(), "base_wage": v.wage,
                           "supervisor_role_id": None} for r, v in scn.roles.items()])
    emps = pd.DataFrame([{
        "employee_id": e.id, "first_name": FIRST[i % len(FIRST)], "last_name": LAST[(i * 7) % len(LAST)],
        "role_id": e.primary_role, "contract_type": e.contract, "department": scn.industry,
        "hire_date": (start_ts - pd.Timedelta(days=int(rng.integers(60, 2500)))).date().isoformat(),
    } for i, e in enumerate(scn.employees)])

    tpl = {t.name: t for t in scn.templates}
    sched_rows, sid = [], 0
    for w in range(weeks):
        for a in roster.assignments.itertuples():
            day = start_ts + pd.Timedelta(days=7 * w + a.day)
            s0 = day + pd.Timedelta(hours=a.start)
            sid += 1
            sched_rows.append({"schedule_id": sid, "employee_id": a.employee, "role_id": a.role,
                               "work_date": day.date().isoformat(), "template": a.template,
                               "shift_start": s0, "shift_end": s0 + pd.Timedelta(hours=a.length),
                               "premium": tpl[a.template].premium})
    sched = pd.DataFrame(sched_rows)

    # Raw time-clock swipes: one row per in/out event, with realistic data problems.
    swipes = []
    for s in sched.itertuples():
        if rng.random() < absence_rate:
            continue
        cin = s.shift_start + pd.Timedelta(minutes=float(rng.normal(-4, 5) + (rng.exponential(12) if rng.random() < 0.08 else 0)))
        cout = s.shift_end + pd.Timedelta(minutes=float(rng.normal(3, 6)))
        swipes.append({"employee_id": s.employee_id, "event": "in", "ts": cin})
        if rng.random() < 0.01:
            swipes.append({"employee_id": s.employee_id, "event": "in", "ts": cin + pd.Timedelta(seconds=40)})  # double swipe
        if rng.random() > 0.015:
            swipes.append({"employee_id": s.employee_id, "event": "out", "ts": cout})                          # else forgot to clock out
    raw = pd.DataFrame(swipes).sample(frac=1, random_state=seed).reset_index(drop=True)

    perf = pd.DataFrame([{"performance_id": i + 1, "employee_id": e.id,
                          "evaluation_date": (start_ts - pd.Timedelta(days=30)).date().isoformat(),
                          "performance_score": round(float(np.clip(rng.normal(3.6, 0.6), 1, 5)), 1), "comments": None}
                         for i, e in enumerate(scn.employees)])
    train = pd.DataFrame([{"training_id": i + 1, "employee_id": e.id,
                           "training_date": (start_ts - pd.Timedelta(days=int(rng.integers(10, 300)))).date().isoformat(),
                           "training_type": "cross-training" if len(e.skills) > 1 else "role certification",
                           "certification": "+".join(e.skills)}
                          for i, e in enumerate(scn.employees)])
    leave = pd.DataFrame([{"leave_id": i + 1, "employee_id": e.id,
                           "leave_start": (start_ts + pd.Timedelta(days=7 * weeks + 3)).date().isoformat(),
                           "leave_end": (start_ts + pd.Timedelta(days=7 * weeks + 5)).date().isoformat(),
                           "leave_type": "annual", "approval_status": "pending"}
                          for i, e in enumerate(scn.employees) if e.days_off])
    r = scn.rules
    rules = pd.DataFrame([{"rule_id": "R1", "regulation_type": "working time",
                           "max_hours_per_week": float(r.get("overtime_threshold", 40)) + float(r.get("max_overtime_hours", 0)),
                           "overtime_threshold": float(r.get("overtime_threshold", 40)),
                           "overtime_rate": float(r.get("overtime_premium", 1.5)),
                           "min_rest_hours": float(r.get("min_rest_hours", 11))}])
    return {"job_role": roles, "employee": emps, "shift_schedule": sched, "swipes": raw,
            "performance": perf, "training": train, "leave_request": leave, "labor_rule": rules}


# ------------------------------------------------------------------ transform
def transform(raw: dict[str, pd.DataFrame], match_window_hours: float = 3.0) -> tuple[dict[str, pd.DataFrame], dict]:
    sched = raw["shift_schedule"].copy()
    for c in ("shift_start", "shift_end"):
        sched[c] = pd.to_datetime(sched[c]).astype("datetime64[ns]")
    sw = raw["swipes"].copy()
    sw["ts"] = pd.to_datetime(sw.ts).astype("datetime64[ns]")
    sw = sw.sort_values(["employee_id", "ts"])
    quality = {"raw_swipes": len(sw)}

    # 1. duplicate swipes: same employee + event within 2 minutes
    sw["gap"] = sw.groupby(["employee_id", "event"]).ts.diff().dt.total_seconds()
    dup = sw.gap.notna() & (sw.gap < 120)
    quality["duplicate_swipes_removed"] = int(dup.sum())
    sw = sw[~dup].drop(columns="gap")

    # 2. match swipes to scheduled shifts (nearest scheduled start/end within the window)
    win = pd.Timedelta(hours=match_window_hours)
    ins = sw[sw.event == "in"].rename(columns={"ts": "clock_in"})
    outs = sw[sw.event == "out"].rename(columns={"ts": "clock_out"})
    s_sorted = sched.sort_values("shift_start")
    m_in = pd.merge_asof(ins.sort_values("clock_in"), s_sorted[["schedule_id", "employee_id", "shift_start"]],
                         left_on="clock_in", right_on="shift_start", by="employee_id", direction="nearest", tolerance=win)
    m_out = pd.merge_asof(outs.sort_values("clock_out"), sched.sort_values("shift_end")[["schedule_id", "employee_id", "shift_end"]],
                          left_on="clock_out", right_on="shift_end", by="employee_id", direction="nearest", tolerance=win)
    quality["unmatched_swipes"] = int(m_in.schedule_id.isna().sum() + m_out.schedule_id.isna().sum())

    att = sched[["schedule_id", "employee_id", "work_date", "shift_start", "shift_end"]].merge(
        m_in.dropna(subset=["schedule_id"])[["schedule_id", "clock_in"]], on="schedule_id", how="left").merge(
        m_out.dropna(subset=["schedule_id"])[["schedule_id", "clock_out"]], on="schedule_id", how="left")

    # 3. statuses and imputation
    absent = att.clock_in.isna()
    missing_out = att.clock_in.notna() & att.clock_out.isna()
    att.loc[missing_out, "clock_out"] = att.loc[missing_out, "shift_end"]
    att["status"] = np.where(absent, "absent", np.where(missing_out, "imputed_clock_out", "present"))
    quality["absences"] = int(absent.sum())
    quality["imputed_clock_outs"] = int(missing_out.sum())

    # 4. validation: clock-out must follow clock-in and shifts cannot exceed 16 hours
    hours = (att.clock_out - att.clock_in).dt.total_seconds() / 3600
    bad = hours.notna() & ((hours <= 0) | (hours > 16))
    quality["invalid_intervals"] = int(bad.sum())
    att["worked_hours"] = hours.where(~bad, 0).fillna(0).round(3)
    att.insert(0, "attendance_id", range(1, len(att) + 1))

    # 5. weekly payroll from worked hours
    emp = raw["employee"].set_index("employee_id")
    wage = raw["job_role"].set_index("role_id").base_wage
    rule = raw["labor_rule"].iloc[0]
    a = att.merge(sched[["schedule_id", "premium"]], on="schedule_id")
    a["week_start"] = pd.to_datetime(a.work_date) - pd.to_timedelta(pd.to_datetime(a.work_date).dt.dayofweek, unit="D")
    a["w"] = a.employee_id.map(emp.role_id).map(wage)
    a["premium_pay"] = a.worked_hours * a.w * (a.premium - 1)
    g = a.groupby(["employee_id", "week_start"]).agg(hours=("worked_hours", "sum"), premium_pay=("premium_pay", "sum"),
                                                     wage=("w", "first")).reset_index()
    g["overtime_hours"] = (g.hours - rule.overtime_threshold).clip(lower=0)
    g["regular_hours"] = g.hours - g.overtime_hours
    g["base_pay"] = g.regular_hours * g.wage
    g["overtime_pay"] = g.overtime_hours * g.wage * rule.overtime_rate
    g["gross_pay"] = g.base_pay + g.premium_pay + g.overtime_pay
    payroll = pd.DataFrame({
        "payroll_id": range(1, len(g) + 1), "employee_id": g.employee_id,
        "period_start": g.week_start.dt.date.astype(str), "period_end": (g.week_start + pd.Timedelta(days=6)).dt.date.astype(str),
        "regular_hours": g.regular_hours.round(2), "overtime_hours": g.overtime_hours.round(2),
        "base_pay": g.base_pay.round(2), "premium_pay": g.premium_pay.round(2),
        "overtime_pay": g.overtime_pay.round(2), "gross_pay": g.gross_pay.round(2)})

    for c in ("shift_start", "shift_end"):
        sched[c] = sched[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in ("clock_in", "clock_out"):
        att[c] = att[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    att = att[["attendance_id", "schedule_id", "employee_id", "work_date", "clock_in", "clock_out", "status", "worked_hours"]]
    clean = {k: v for k, v in raw.items() if k != "swipes"}
    clean.update({"shift_schedule": sched, "attendance": att, "payroll": payroll})
    return clean, quality


# ------------------------------------------------------------------ load
LOAD_ORDER = ["job_role", "labor_rule", "employee", "shift_schedule", "attendance", "payroll",
              "performance", "training", "leave_request"]


def load(tables: dict[str, pd.DataFrame], db_path: str | Path) -> dict:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    schema = resources.files("labor_opt").joinpath("sql/schema.sql").read_text()
    counts = {}
    with sqlite3.connect(db_path) as con:
        con.executescript(schema)
        for name in LOAD_ORDER:
            df = tables[name]
            df.to_sql(name, con, if_exists="append", index=False)
            counts[name] = len(df)
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise ValueError(f"foreign key violations: {fk[:5]}")
    return counts


def query(db_path: str | Path, sql: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(sql, con)


def kpi_extracts(db_path: str | Path) -> dict[str, pd.DataFrame]:
    return {
        "labor_cost_daily": query(db_path, "SELECT * FROM v_labor_cost_daily ORDER BY work_date, role_id"),
        "weekly_hours": query(db_path, "SELECT * FROM v_weekly_hours ORDER BY week_start, employee_id"),
        "rest_violations": query(db_path, "SELECT * FROM v_rest_violations"),
        "absenteeism": query(db_path, "SELECT * FROM v_absenteeism"),
        "late_arrivals": query(db_path, """SELECT role_id, COUNT(*) AS shifts,
                                       ROUND(AVG(CASE WHEN late_minutes > 5 THEN 1.0 ELSE 0 END), 4) AS late_rate,
                                       ROUND(AVG(variance_hours), 3) AS avg_variance_hours
                                       FROM v_shift_variance WHERE status != 'absent' GROUP BY role_id"""),
    }


def run_etl(scn: Scenario, roster: RosterResult, db_path: str | Path, out_dir: str | Path | None = None,
            weeks: int = 4, seed: int = 1) -> EtlReport:
    raw = extract(scn, roster, weeks=weeks, seed=seed)
    clean, quality = transform(raw)
    counts = load(clean, db_path)
    kpis = kpi_extracts(db_path)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for k, v in kpis.items():
            v.to_csv(out / f"{k}.csv", index=False)
    return EtlReport(Path(db_path), counts, quality, kpis)
