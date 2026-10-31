import sqlite3
from contextlib import closing

from labor_opt.etl import run_etl
from labor_opt.roster import solve_roster


def test_etl_loads_and_reports(retail, tmp_path):
    roster = solve_roster(retail, time_limit=30)
    db = tmp_path / "wh.db"
    rep = run_etl(retail, roster, db, tmp_path / "out", weeks=2, seed=3)

    n_shifts = 2 * len(roster.assignments)
    assert rep.rows_loaded["shift_schedule"] == n_shifts
    assert rep.rows_loaded["attendance"] == n_shifts            # one attendance row per scheduled shift
    assert rep.quality["invalid_intervals"] == 0

    with closing(sqlite3.connect(db)) as con:
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        absent = con.execute("SELECT COUNT(*) FROM attendance WHERE status='absent'").fetchone()[0]
        assert absent == rep.quality["absences"]
        bad = con.execute("SELECT COUNT(*) FROM payroll WHERE gross_pay < base_pay").fetchone()[0]
        assert bad == 0

    k = rep.kpis
    assert set(k) == {"labor_cost_daily", "weekly_hours", "rest_violations", "absenteeism", "late_arrivals"}
    assert len(k["labor_cost_daily"]) == 14 * len(retail.roles)
    assert k["rest_violations"].empty                            # the optimizer enforces 11h rest
    assert (tmp_path / "out" / "weekly_hours.csv").exists()
