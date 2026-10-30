def _req(scn, role, day, hour):
    r = scn.requirements
    return int(r[(r.role == role) & (r.day == day) & (r.hour == hour)].required.iat[0])


def test_casino_requirements(casino):
    assert _req(casino, "dealer", 0, 20) == 6          # Monday evening, as specified
    assert _req(casino, "dealer", 4, 20) == 9          # Friday evening, x1.5 peak multiplier
    assert _req(casino, "supervisor", 2, 3) == 1       # night block
    assert len(casino.requirements) == 3 * 168


def test_retail_weekend_override_and_closed_hours(retail):
    assert _req(retail, "sales_associate", 1, 15) == 5
    assert _req(retail, "sales_associate", 5, 15) == 6
    assert _req(retail, "manager", 6, 11) == 2
    assert _req(retail, "csr", 1, 8) == 0              # store closed before 10:00


def test_workforce_and_cross_training(casino):
    assert len(casino.employees) == 50
    sup = [e for e in casino.employees if e.primary_role == "supervisor"]
    assert sum("dealer" in e.skills for e in sup) == 5     # 50% of 10, full-timers first
    assert all(e.contract == "FT" for e in sup if "dealer" in e.skills)


def test_template_wraps_week():
    from labor_opt.config import ShiftTemplate
    t = ShiftTemplate("grave", 22, 8)
    assert t.covered_hours(6) == [166, 167, 0, 1, 2, 3, 4, 5]
    assert t.crosses_midnight
