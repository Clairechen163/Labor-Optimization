import numpy as np
import pandas as pd
import pytest

from labor_opt import kpi, proforma as pf
from labor_opt import synthetic as syn


def test_consecutive_rule():
    df = pd.DataFrame({"unit": ["A"] * 5 + ["B"] * 5, "t": list(range(5)) * 2,
                       "x": [70, 58, 70, 58, 57, 58, 70, 70, 70, 70]})
    f = kpi.flag(df, [{"metric": "x", "op": "<", "threshold": 60, "consecutive": 2}], "unit", "t")
    assert f.unit.tolist() == ["A"]
    assert f.first_flagged.iat[0] == 4


def test_p_chart_catches_drift_after_it_starts():
    tx = kpi.textile_kpis(syn.textile_daily_kpis(shift_day=60))
    det = kpi.compare_detection(tx, "line", "L4", "date", "defects", "units", threshold=0.05)
    sig = det["spc_first_signal"]
    assert sig is not None and sig > pd.Timestamp("2026-01-01") + pd.Timedelta(days=60)
    # stable lines should rarely signal
    healthy = kpi.compare_detection(tx, "line", "L1", "date", "defects", "units", threshold=0.05)
    assert healthy["chart"].signal.mean() < 0.05


def test_xmr_detects_step():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(25, 1, 40), rng.normal(21, 1, 20)]
    ch = kpi.xmr_chart(x, baseline_periods=40)
    after = np.flatnonzero(ch.signal.to_numpy()[40:])
    assert len(after) and after[0] <= 2                 # caught within three points of the step
    assert ch.signal[:40].sum() <= 2                    # occasional false alarms only in the stable period


def test_proforma_reproduces_case_studies():
    r = pf.case_studies()
    assert r["textile"]["incremental_profit"] == pytest.approx(1000)
    cases = r["cases"].set_index("case")
    assert cases.loc["Retail promotion (as stated)", "incremental_profit"] == pytest.approx(14000)
    assert cases.loc["Casino event (as stated)", "incremental_profit"] == pytest.approx(17000)
    assert cases.loc["Casino event (net of 40% gaming tax)", "incremental_profit"] == pytest.approx(9000)


def test_break_even_and_certainty():
    c = pf.StaffingCase("x", 100_000, 50, 300, 10, 0.2, margin=0.6)
    assert c.incremental(c.break_even_uplift()) == pytest.approx(0, abs=1e-6)
    assert c.simulate()["p_loss"] == 0.0
