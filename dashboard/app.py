"""Streamlit dashboard: `streamlit run dashboard/app.py` from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labor_opt import casino_math as cm  # noqa: E402
from labor_opt import kpi  # noqa: E402
from labor_opt import synthetic as syn  # noqa: E402
from labor_opt.config import CONFIG_DIR, load_scenario  # noqa: E402
from labor_opt.demand import erlang_staff, table_game_dealers, workload_staff  # noqa: E402
from labor_opt.forecast import backtest, summarize_backtest  # noqa: E402
from labor_opt.pipeline import PIT, newsvendor_quantile  # noqa: E402
from labor_opt.proforma import StaffingCase  # noqa: E402
from labor_opt.roster import compare_policies, solve_roster  # noqa: E402

st.set_page_config(page_title="Labor Optimization", layout="wide")
st.title("Labor Optimization")
st.caption("Forecasting, staffing models, roster optimization and evaluation. All data is synthetic.")

SCENARIOS = {"Casino gaming pit": "casino", "Textile factory": "textile", "Luxury retail boutique": "retail"}

# Same palette as the report figures: required = grey, scheduled / measured = blue, alerts = orange.
# GREY also marks annotations: it stays readable on both the light and the dark Streamlit theme.
BLUE, GREY, ORANGE = "#2a78d6", "#8a8984", "#eb6834"
ROLE_NAMES = {"csr": "Customer service rep"}


def role_label(role: str) -> str:
    return ROLE_NAMES.get(role, role.replace("_", " ").capitalize())


def coverage_chart(cov: pd.DataFrame, days: list[str], role: str) -> alt.LayerChart:
    """Required vs scheduled staff for one role across the week, as steps, with gaps shaded."""
    d = cov[["t", "day", "hour", "required", "scheduled"]].sort_values("t").reset_index(drop=True)
    d["top"] = d[["required", "scheduled"]].max(axis=1)
    d["t1"] = d.t + 1
    d["mid"] = d.t + 0.5
    d["day_name"] = [days[i] for i in d.day]
    d["time"] = [f"{h:02d}:00-{(h + 1) % 24:02d}:00" for h in d.hour]
    steps = pd.concat([d, d.iloc[[-1]].assign(t=168)], ignore_index=True)    # close the last hour
    long = steps.melt(id_vars="t", value_vars=["scheduled", "required"], var_name="series", value_name="staff")
    long["series"] = long.series.map({"scheduled": "Scheduled", "required": "Required"})

    day_names = "[" + ",".join(f"'{x}'" for x in days) + "]"
    x = alt.X("t:Q", scale=alt.Scale(domain=[0, 168], nice=False), title=None,
              axis=alt.Axis(values=list(range(0, 168, 24)), labelExpr=f"{day_names}[floor(datum.value / 24)]",
                            labelAlign="left", labelPadding=4))
    y_max = int(d.top.max()) + 1
    y = alt.Y("staff:Q", title=f"{role_label(role).lower()}s on duty",
              scale=alt.Scale(domain=[0, y_max]), axis=alt.Axis(tickMinStep=1, format="d"))
    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["Scheduled", "Required"], range=[BLUE, GREY]))

    gap = alt.Chart(steps).mark_area(interpolate="step-after", color=ORANGE, opacity=0.35).encode(
        x=x, y=alt.Y("scheduled:Q", scale=alt.Scale(domain=[0, y_max])), y2="top:Q")
    sched = alt.Chart(long).transform_filter("datum.series == 'Scheduled'").mark_line(
        interpolate="step-after", strokeWidth=2.5).encode(x=x, y=y, color=color)
    req = alt.Chart(long).transform_filter("datum.series == 'Required'").mark_line(
        interpolate="step-after", strokeWidth=1.6, strokeDash=[5, 3]).encode(x=x, y=y, color=color)
    hover = alt.selection_point(on="pointerover", fields=["t"], empty=False, clear="pointerout")
    hit = alt.Chart(d).mark_rect(color="white", opacity=0.001).encode(
        x=x, x2="t1:Q",
        tooltip=[alt.Tooltip("day_name:N", title="Day"), alt.Tooltip("time:N", title="Hour"),
                 alt.Tooltip("required:Q", title="Required"), alt.Tooltip("scheduled:Q", title="Scheduled")],
    ).add_params(hover)
    rule = alt.Chart(d).mark_rule(color=GREY, strokeWidth=1).encode(x="mid:Q").transform_filter(hover)
    return alt.layer(gap, sched, req, rule, hit).properties(height=300)


def control_chart(det: dict, threshold: float) -> alt.LayerChart:
    """Daily defect rate with its 3-sigma control limit, the fixed threshold and the first signal."""
    ch = det["chart"].copy()
    ch["rate"] = ch.p * 100
    ch["limit"] = ch.ucl * 100
    ch["fixed"] = threshold * 100
    names = {"rate": "Defect rate", "limit": "Control limit (3 sigma)", "fixed": f"Fixed threshold ({threshold:.0%})"}
    long = ch.melt(id_vars="date", value_vars=list(names), var_name="series", value_name="pct")
    long["series"] = long.series.map(names)
    domain = list(names.values())
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%b %d", labelAngle=0, tickCount=8))
    y_max = max(threshold * 100, float(ch.rate.max())) * 1.1
    yscale = alt.Scale(domain=[0, y_max])
    lines = alt.Chart(long).mark_line().encode(
        x=x, y=alt.Y("pct:Q", title="defect rate (%)", scale=yscale),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=domain, range=[BLUE, GREY, ORANGE])),
        strokeDash=alt.StrokeDash("series:N", title=None, legend=alt.Legend(orient="top"),
                                  scale=alt.Scale(domain=domain, range=[[1, 0], [4, 3], [1, 0]])),
        strokeWidth=alt.StrokeWidth("series:N", legend=None, scale=alt.Scale(domain=domain, range=[2, 1.4, 2])))
    pts = alt.Chart(ch).mark_circle(size=28, color=BLUE, opacity=1).encode(
        x=x, y=alt.Y("rate:Q", scale=yscale),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%b %d"),
                 alt.Tooltip("rate:Q", format=".2f", title="Defect rate (%)"),
                 alt.Tooltip("limit:Q", format=".2f", title="Control limit (%)")])
    layers = [lines, pts]
    if det["spc_first_signal"] is not None:
        sig = ch[ch.date == det["spc_first_signal"]]
        ring = alt.Chart(sig).mark_point(size=240, shape="circle", filled=False, color=GREY, strokeWidth=2.5)
        label = alt.Chart(sig).mark_text(text="first signal", color=GREY, align="right", dx=-12, dy=-12, fontSize=12)
        layers += [ring.encode(x=x, y=alt.Y("rate:Q", scale=yscale)), label.encode(x=x, y=alt.Y("rate:Q", scale=yscale))]
    return alt.layer(*layers).properties(height=280)

tab_roster, tab_demand, tab_kpi, tab_casino, tab_pf = st.tabs(
    ["Roster optimizer", "Demand & forecast", "KPI monitor", "Casino math", "Pro-forma"])


@st.cache_data(show_spinner=False)
def _solve(key: str, mult: float, flexible: bool, cross: bool):
    scn = load_scenario(CONFIG_DIR / f"{key}.yaml")
    if mult != 1.0:
        req = scn.requirements.copy()
        req["required"] = np.floor(req.required * mult + 0.5).astype(int)
        scn = scn.with_requirements(req)
    res = solve_roster(scn, templates=None if flexible else scn.baseline_templates, cross_training=cross, time_limit=30)
    return scn, res


with tab_roster:
    c1, c2, c3, c4 = st.columns(4)
    label = c1.selectbox("Operation", list(SCENARIOS))
    mult = c2.slider("Demand multiplier", 0.7, 1.3, 1.0, 0.05)
    flexible = c3.toggle("Flexible shift starts", True)
    cross = c4.toggle("Use cross-trained staff", True)
    scn, res = _solve(SCENARIOS[label], mult, flexible, cross)
    s = res.summary
    if res.status != "Optimal":
        st.error(f"Solver status: {s.get('solution')}")
    else:
        m = st.columns(5)
        m[0].metric("Total weekly cost", f"{scn.currency} {s['total_cost']:,.0f}")
        m[1].metric("Call-in cost", f"{s['call_in_cost']:,.0f}")
        m[2].metric("Coverage", f"{s['coverage_rate']:.1%}")
        m[3].metric("Schedule efficiency", f"{s['schedule_efficiency']:.1%}")
        m[4].metric("Solve time", f"{s['runtime_s']:.1f}s")
        role = st.radio("Role", list(scn.roles), horizontal=True, format_func=role_label)
        cov = res.coverage[res.coverage.role == role]
        st.altair_chart(coverage_chart(cov, scn.days, role))
        name = role_label(role).lower()
        st.caption(f"Orange shading marks hours with too few staff: {cov.short.sum():.0f} uncovered and "
                   f"{cov.surplus.sum():.0f} surplus {name}-hours this week. Hover over the chart for hourly detail.")
        with st.expander("Shift assignments"):
            st.dataframe(res.assignments[["day_name", "template", "start", "length", "employee", "role", "primary_role", "cost"]],
                         width="stretch", hide_index=True)
        with st.expander("Hours per employee"):
            st.dataframe(res.employees, width="stretch", hide_index=True)
        if st.button("Compare all four policies"):
            tab = compare_policies(scn, time_limit=30)
            st.dataframe(tab.style.format({c: "{:,.0f}" for c in ["labor_cost", "call_in_cost", "total_cost", "paid_hours",
                                                                   "shortfall_staff_hours", "surplus_staff_hours"]}
                                          | {"coverage_rate": "{:.1%}", "schedule_efficiency": "{:.1%}", "runtime_s": "{:.1f}"}),
                         width="stretch", hide_index=True)

with tab_demand:
    st.subheader("Staffing calculators")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Workload model (manufacturing)**")
        t = st.number_input("Units per month", value=100_000, step=5_000)
        p = st.number_input("Units per worker-hour", value=25.0)
        h = st.number_input("Hours per worker per month", value=160)
        e = st.slider("Efficiency", 0.5, 1.0, 0.9)
        st.metric("Workers", workload_staff(t, p, h, e)["workers"])
    with b:
        st.markdown("**Erlang C (service desk / sales floor)**")
        lam = st.number_input("Customers per hour", value=50.0)
        aht = st.number_input("Minutes per customer", value=10.0)
        wait = st.number_input("Target wait (minutes)", value=2.0)
        sl = st.slider("Service level", 0.5, 0.99, 0.8)
        shrink = st.slider("Shrinkage", 0.0, 0.4, 0.2)
        r = erlang_staff(lam, aht, wait, sl, shrink)
        st.metric("On duty", r["servers"], help=f"P(wait)={r['prob_wait']:.2f}, occupancy={r['occupancy']:.0%}")
        st.metric("Scheduled (after shrinkage)", r["scheduled"])
    with c:
        st.markdown("**Table games (casino)**")
        guests = st.number_input("Guests on floor", value=300)
        share = st.slider("Share playing this game", 0.0, 1.0, 0.4)
        occ = st.slider("Target seat occupancy", 0.5, 1.0, 0.8)
        relief = st.slider("Relief ratio (dealers per table)", 1.0, 1.5, 1.25)
        r = table_game_dealers(guests * share, target_occupancy=occ, relief_ratio=relief)
        st.metric("Tables", r["tables"])
        st.metric("Dealers", r["dealers"])

    st.subheader("Pit traffic forecast backtest")
    call_in = st.slider("Call-in cost multiple of wage", 1.5, 5.0, 3.0, 0.5)
    q = newsvendor_quantile(call_in - 1, 1)
    st.caption(f"Newsvendor staffing quantile = {q:.2f}")

    @st.cache_data(show_spinner=False)
    def _bt(q):
        return summarize_backtest(backtest(syn.pit_traffic(), staffing_quantile=q, dealer_kw=PIT))
    st.dataframe(_bt(round(q, 3)), width="stretch", hide_index=True)

with tab_kpi:
    th = kpi.load_thresholds(CONFIG_DIR / "kpi_thresholds.yaml")
    tx = kpi.textile_kpis(syn.textile_daily_kpis())
    rt = kpi.retail_kpis(syn.retail_daily_kpis())
    cs = syn.casino_shift_kpis()
    st.subheader("Flagged operations")
    flags = kpi.flag_table([("Textile", tx, th["textile"], "line", "date"),
                            ("Retail", rt, th["retail"], "store", "date"),
                            ("Casino", cs, th["casino"], "dealer", "shift")])
    flags.columns = ["Operation", "Unit", "Rule", "First flagged", "Value", "Target", "Action"]
    st.dataframe(flags.style.format({"Value": "{:.3g}", "Target": "{:.3g}"}), width="stretch", hide_index=True)
    st.subheader("Defect-rate control chart")
    line = st.selectbox("Production line", sorted(tx.line.unique()), index=3,
                        help="Line L4 develops tool wear from March 2; the other lines are stable.")
    det = kpi.compare_detection(tx, "line", line, "date", "defects", "units", 0.05)
    st.altair_chart(control_chart(det, 0.05))

    def _day(v):
        return pd.Timestamp(v).strftime("%b %d")
    sig, thr = det["spc_first_signal"], det["threshold_first_flag"]
    spc_text = f"The control chart first signals on {_day(sig)} (ringed)." if sig is not None else \
        "The control chart raises no signal on this line."
    thr_text = f"The fixed 5% threshold first fires on {_day(thr)}." if thr is not None else \
        "The fixed 5% threshold never fires."
    st.caption(f"{spc_text} {thr_text}")

with tab_casino:
    st.subheader("House edge and payback")
    gs = cm.game_summary()
    st.dataframe(gs.style.format({"p_win": "{:.4f}", "house_edge": "{:.3%}", "payback": "{:.3%}"}),
                 width="stretch", hide_index=True)
    a, b = st.columns(2)
    with a:
        st.markdown("**Slot session**")
        rtp = st.slider("RTP", 0.85, 0.99, 0.96, 0.005)
        spins = st.number_input("Spins", value=1000, step=100)
        sess = cm.example_paytable(rtp).session(int(spins))
        st.metric("Expected loss (units)", f"{sess['expected_loss']:,.1f}")
        st.caption(f"90% of sessions end between {sess['p05_net']:,.0f} and {sess['p95_net']:,.0f} units")
    with b:
        st.markdown("**Poker call decision**")
        pot = st.number_input("Pot", value=100.0)
        call = st.number_input("Amount to call", value=20.0)
        outs = st.number_input("Outs", value=9)
        unseen = st.number_input("Unseen cards", value=46)
        eq = outs / unseen
        st.metric("Required equity", f"{cm.required_equity(pot, call):.1%}", help=f"Your equity: {eq:.1%}")
        st.metric("EV of calling", f"{cm.call_ev(pot, call, eq):+.2f}")

with tab_pf:
    st.subheader("Should we add staff for the event?")
    a, b = st.columns(2)
    rev = a.number_input("Baseline revenue per day", value=100_000)
    staff = a.number_input("Baseline staff", value=50)
    cost = a.number_input("Cost per staff per day", value=300)
    extra = b.number_input("Extra staff", value=10)
    up = b.slider("Expected revenue uplift", 0.0, 0.5, 0.2)
    sd = b.slider("Uncertainty in uplift (SD)", 0.0, 0.2, 0.08)
    margin = b.slider("Contribution margin (after tax / COGS)", 0.2, 1.0, 0.6)
    case = StaffingCase("custom", rev, staff, cost, extra, up, margin=margin, uplift_sd=sd)
    s = case.summary()
    m = st.columns(4)
    m[0].metric("Incremental profit/day", f"{s['incremental_profit']:,.0f}")
    m[1].metric("Break-even uplift", f"{s['break_even_uplift']:.1%}")
    m[2].metric("P(loss)", f"{s['mc_p_loss']:.1%}")
    m[3].metric("90% range", f"{s['mc_p05']:,.0f} to {s['mc_p95']:,.0f}")
