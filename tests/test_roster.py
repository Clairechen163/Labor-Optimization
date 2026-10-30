import pytest

from labor_opt.roster import solve_roster
from labor_opt.staffing_lp import block_marginal_costs, solve_covering


def test_tiny_problem_covers_exactly(make_scenario):
    scn = make_scenario()
    res = solve_roster(scn)
    s = res.summary
    assert s["solution"] == "optimal"
    assert s["shortfall_staff_hours"] == 0
    assert s["labor_cost"] == pytest.approx(24 * 100)          # three 8h shifts at 100/h
    mon = res.coverage[res.coverage.day == 0]
    assert (mon.scheduled == 1).all()


def test_one_shift_per_day_and_call_in_pricing(make_scenario):
    # One employee can work at most one shift a day, so 16 of the 24 required hours are call-ins.
    scn = make_scenario(workforce={"staff": [{"count": 1, "contract": "FT"}]})
    s = solve_roster(scn).summary
    assert s["paid_hours"] == 8
    assert s["shortfall_staff_hours"] == 16
    assert s["call_in_cost"] == pytest.approx(16 * 100 * 3.0)


def test_hard_coverage_infeasible_when_understaffed(make_scenario):
    scn = make_scenario(workforce={"staff": [{"count": 1, "contract": "FT"}]})
    assert solve_roster(scn, hard_coverage=True).status == "Infeasible"


def test_guaranteed_hours_equality(casino):
    res = solve_roster(casino)
    ft = res.employees[res.employees.contract == "FT"]
    assert (ft.hours == 40).all()                       # min_hours == max_hours == 40 is an equality constraint
    pt = res.employees[res.employees.contract == "PT"]
    assert (pt.hours <= 24).all()


def test_availability_rest_and_capacity_respected(casino):
    res = solve_roster(casino)
    emp = {e.id: e for e in casino.employees}
    tpl = {t.name: t for t in casino.templates}
    for a in res.assignments.itertuples():
        e = emp[a.employee]
        assert a.day not in e.days_off
        assert not (e.no_nights and tpl[a.template].crosses_midnight)
        assert a.role in e.skills
    # at most one shift per employee per day
    assert not res.assignments.duplicated(["employee", "day"]).any()
    # 11h minimum rest between consecutive days
    for eid, g in res.assignments.groupby("employee"):
        starts = {r.day: (r.start, r.length) for r in g.itertuples()}
        for d, (s0, ln) in starts.items():
            nxt = starts.get((d + 1) % 7)
            if nxt:
                assert 24 + nxt[0] - (s0 + ln) >= 11
    cap = casino.roles["dealer"].max_on_duty
    assert res.coverage[res.coverage.role == "dealer"].scheduled.max() <= cap


def test_cross_training_switch(casino):
    res = solve_roster(casino, cross_training=False)
    assert (res.assignments.role == res.assignments.primary_role).all()


def test_flexible_never_worse_than_fixed(casino):
    fixed = solve_roster(casino, templates=casino.baseline_templates).summary["total_cost"]
    flex = solve_roster(casino).summary["total_cost"]
    assert flex <= fixed * 1.005    # baseline templates are a subset; allow the solver's relative gap


def test_lp_is_lower_bound(casino):
    lp = solve_covering(casino).cost
    ip = solve_covering(casino, integer=True).cost
    roster = solve_roster(casino).summary["total_cost"]
    assert lp <= ip + 1e-6
    assert ip <= roster * 1.005


def test_block_marginal_costs_nonnegative(casino):
    mc = block_marginal_costs(casino, "dealer")
    assert len(mc) == 28
    assert (mc.marginal_cost >= -1e-6).all()
