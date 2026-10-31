"""Run every case study and write reports/RESULTS.md plus figures. `labor-opt report` calls this."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import casino_math as cm  # noqa: E402
from . import experiments as ex  # noqa: E402
from . import kpi  # noqa: E402
from . import synthetic as syn  # noqa: E402
from .config import CONFIG_DIR, load_scenario  # noqa: E402
from .demand import erlang_staff, table_game_dealers, workload_staff  # noqa: E402
from .pipeline import forecast_driven_plan  # noqa: E402
from .proforma import case_studies  # noqa: E402
from .roster import compare_policies, solve_roster  # noqa: E402
from .staffing_lp import block_marginal_costs, solve_covering  # noqa: E402

# Validated reference palette (light mode): categorical slots 1-3, ink and surface tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2,
})


def _money(x: float) -> str:
    return f"{x:,.0f}"


def _md_table(df: pd.DataFrame, floatfmt: dict | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            f = floatfmt.get(c)
            if f and isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v):
                cells.append(format(v, f))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ figures
def fig_coverage(scn, fixed, flex, path: Path, role: str = "dealer"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, sharey=True)
    for ax, (label, res) in zip(axes, [("Fixed three-shift pattern", fixed), ("Optimized flexible starts", flex)]):
        c = res.coverage[res.coverage.role == role].sort_values("t")
        ax.step(c.t, c.required, where="post", color=MUTED, lw=1.6, label="Required")
        ax.step(c.t, c.scheduled, where="post", color=BLUE, lw=2, label="Scheduled")
        gap = c.scheduled < c.required
        ax.fill_between(c.t, c.scheduled, c.required, where=gap, step="post", color=ORANGE, alpha=0.35,
                        lw=0, label="Uncovered")
        ax.set_title(f"{label}: {c.short.sum():.0f} uncovered and {c.surplus.sum():.0f} surplus {role}-hours",
                     fontsize=10)
        ax.set_ylabel(f"{role}s on duty")
    axes[1].set_xticks(np.arange(0, 169, 24), labels=scn.days + [""])
    axes[1].set_xlim(0, 168)
    axes[0].legend(loc="upper left", ncol=3, fontsize=9)
    fig.suptitle(f"{scn.name}: dealer coverage across the week", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_policy(tables: dict[str, pd.DataFrame], path: Path):
    series = [("Fixed shifts, no cross-training", "Fixed shifts (baseline)", MUTED),
              ("Fixed shifts + cross-training", "Fixed shifts + cross-training", ORANGE),
              ("Flexible starts, no cross-training", "Flexible starts", AQUA),
              ("Flexible starts + cross-training (optimized)", "Flexible starts + cross-training", BLUE)]
    names = list(tables)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    h = 0.19
    for i, (pol, label, color) in enumerate(series):
        vals = []
        for n in names:
            t = tables[n].set_index("policy")
            vals.append(100 * t.loc[pol, "total_cost"] / t.loc[series[0][0], "total_cost"])
        y = np.arange(len(names)) + (i - 1.5) * (h + 0.02)
        ax.barh(y, vals, height=h, color=color, label=label)
        for yy, v in zip(y, vals):
            ax.text(v + 1.0, yy, f"{v:.1f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(np.arange(len(names)), labels=names)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("Total weekly cost incl. call-ins (baseline = 100)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8.5, handlelength=1.2, columnspacing=1.2)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_forecast(traffic: pd.DataFrame, path: Path, q: float):
    from .forecast import CountForecaster
    w = int(traffic.week.max())
    train, test = traffic[traffic.week < w], traffic[traffic.week == w]
    m = CountForecaster().fit(train)
    t = np.arange(168)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(t, test.players, color=MUTED, lw=1.4, label="Actual players")
    ax.plot(t, m.predict(test), color=BLUE, label="Forecast (mean)")
    ax.plot(t, m.quantile(test, q), color=ORANGE, lw=1.6, ls="--", label=f"Staffing plan (P{q * 100:.0f})")
    ax.set_xticks(np.arange(0, 169, 24), labels=syn.DAYS + [""])
    ax.set_xlim(0, 168)
    ax.set_ylabel("players in pit")
    ax.set_title(f"Hold-out week {w}: negative-binomial forecast of hourly pit traffic")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_marginal(mc: pd.DataFrame, days: list[str], path: Path):
    order = [b for b in ["night", "morning", "afternoon", "evening"] if b in mc.block.unique()]
    tab = mc.pivot(index="day", columns="block", values="marginal_cost").loc[[d for d in days if d in mc.day.unique()], order]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    im = ax.imshow(tab.to_numpy(), cmap=cmap, aspect="auto")
    vmax = np.nanmax(tab.to_numpy())
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.iat[i, j]
            ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.55 * vmax else INK)
    ax.set_xticks(range(len(order)), labels=[f"{b}\n{mc[mc.block == b].hours.iat[0]}" for b in order])
    ax.set_yticks(range(len(tab.index)), labels=tab.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Cost of requiring one more dealer through a block (HKD)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_spc(det: dict, threshold: float, wear_start, path: Path):
    ch = det["chart"]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(ch.date, ch.p * 100, color=BLUE, lw=1.6, marker="o", ms=3, label="Daily defect rate, line L4")
    ax.plot(ch.date, ch.ucl * 100, color=INK2, lw=1, ls="--", label="Control limit (3σ)")
    ax.axhline(threshold * 100, color=ORANGE, lw=1.6, label=f"Fixed threshold ({threshold:.0%})")
    ax.axvline(wear_start, color=MUTED, lw=1, ls=":")
    ax.text(wear_start, ax.get_ylim()[1] * 0.97, " tool wear begins", color=INK2, fontsize=8.5, va="top")
    if det["spc_first_signal"] is not None:
        sig = det["spc_first_signal"]
        y = float(ch.loc[ch.date == sig, "p"].iat[0] * 100)
        ax.scatter([sig], [y], s=70, facecolor=SURFACE, edgecolor=INK, zorder=5, lw=1.5)
        ax.annotate(f"first control-chart signal ({sig:%b %d})", (sig, y), xytext=(28, -42), textcoords="offset points",
                    fontsize=8.5, color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 1})
    ax.set_ylabel("defect rate (%)")
    ax.set_ylim(top=max(threshold * 100, float(ch.p.max() * 100)) * 1.08)
    caught = "never flags" if det["threshold_first_flag"] is None else "flags late"
    ax.set_title(f"Control chart catches the drift the fixed threshold {caught}", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_did(panel: pd.DataFrame, path: Path):
    g = panel.groupby(["week", "treated_line"]).units_per_labor_hour.mean().unstack()
    switch = panel.loc[panel.post == 1, "week"].min()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(g.index, g[0], color=ORANGE, marker="o", ms=4, label="Control lines (old workflow)")
    ax.plot(g.index, g[1], color=BLUE, marker="o", ms=4, label="Treated lines (new workflow from week %d)" % switch)
    ax.axvline(switch - 0.5, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("week")
    ax.set_ylabel("units per labor hour")
    ax.set_title("Both groups trend up; only the gap between them is the workflow effect")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ report
def build_report(out_dir: str | Path = "reports", time_limit: int = 60) -> dict:
    out = Path(out_dir)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    md: list[str] = ["# Results", "",
                     "Generated by `labor-opt report` from the configs in `configs/` and seeded synthetic data. "
                     "Every number below is reproducible. Currency is HKD unless stated.", ""]

    # 1. scheduling
    md += ["## 1. Weekly roster optimization", "",
           "Total cost = scheduled pay (incl. night premiums and overtime) + call-in cost for any uncovered "
           "staff-hours (3x base wage). The covering LP is the same problem without individual employees, "
           "so its optimum is a lower bound on any roster.", ""]
    policy_tables = {}
    for name in ("casino", "textile", "retail"):
        scn = load_scenario(CONFIG_DIR / f"{name}.yaml")
        tab = compare_policies(scn, time_limit=time_limit)
        lb = solve_covering(scn).cost
        policy_tables[scn.name] = tab
        base = tab.iloc[0].total_cost
        best = tab.iloc[-1]
        results[name] = {"policies": tab.to_dict("records"), "lp_lower_bound": lb,
                         "saving_vs_fixed": 1 - best.total_cost / base}
        show = tab[["policy", "labor_cost", "call_in_cost", "total_cost", "shortfall_staff_hours", "schedule_efficiency"]].copy()
        show.columns = ["Policy", "Scheduled pay", "Call-in cost", "Total cost", "Uncovered staff-hours", "Efficiency"]
        md += [f"### {scn.name}", "",
               f"{len(scn.employees)} employees, {int(scn.requirements.required.sum())} required staff-hours per week. "
               f"Optimized total cost is **{1 - best.total_cost / base:.1%} lower** than the fixed-shift baseline "
               f"({', '.join(scn.baseline_templates)}). The covering-LP lower bound is {_money(lb)}; the optimized roster "
               f"is {best.total_cost / lb - 1:.1%} above it, which is the price of individual rules "
               f"(guaranteed hours, availability, rest).", "",
               _md_table(show, {"Scheduled pay": ",.0f", "Call-in cost": ",.0f", "Total cost": ",.0f",
                                "Uncovered staff-hours": ".0f", "Efficiency": ".1%"}), ""]
        if name == "casino":
            fixed = solve_roster(scn, templates=scn.baseline_templates, cross_training=False, time_limit=time_limit)
            flex = solve_roster(scn, time_limit=time_limit)
            fig_coverage(scn, fixed, flex, figs / "coverage_casino.png")
            mc = block_marginal_costs(scn, "dealer")
            fig_marginal(mc, scn.days, figs / "marginal_cost_dealers.png")
            results["casino"]["marginal_cost_dealer"] = mc.to_dict("records")
            cheap = mc.loc[mc.marginal_cost.idxmin()]
            dear = mc.loc[mc.marginal_cost.idxmax()]
            capped = mc[mc.at_floor_capacity]
            cap_note = (f" {', '.join(capped.day + ' ' + capped.block)} are at floor capacity, so the extra dealer "
                        "there can only come from call-ins." if len(capped) else "")
            md += ["![Dealer coverage](figures/coverage_casino.png)", "",
                   "**Marginal cost of demand.** Re-solving with one more dealer required through each 6-hour block "
                   "prices the requirement. A block that a lightly used shift can absorb is cheap "
                   f"({cheap.day} {cheap.block}: {_money(cheap.marginal_cost)}); one that needs a whole new 8-hour "
                   f"shift is not ({dear.day} {dear.block}: {_money(dear.marginal_cost)}).{cap_note}", "",
                   "![Marginal cost](figures/marginal_cost_dealers.png)", ""]
    fig_policy(policy_tables, figs / "policy_comparison.png")
    md += ["### Value of flexibility, all three operations", "", "![Policy comparison](figures/policy_comparison.png)", ""]

    # 2. forecast-driven plan
    plan = forecast_driven_plan(time_limit=time_limit)
    traffic = syn.pit_traffic()
    fig_forecast(traffic, figs / "forecast_week.png", plan.quantile)
    bt = plan.backtest.copy()
    results["forecast"] = {"backtest": bt.to_dict("records"), "quantile": plan.quantile,
                           "roster": {k: v for k, v in plan.roster.summary.items() if k != "templates"}}
    bt.columns = ["Method", "WAPE", "MAE (players)", "Under-staffed dealer-hours/wk", "Over-staffed dealer-hours/wk"]
    s = plan.roster.summary
    md += ["## 2. Forecast -> requirement -> roster (casino pit)", "",
           "30 weeks of hourly pit traffic, rolling-origin backtest over the last 6 weeks. Players are converted to "
           "dealers with the table-game model (7 seats, 80% target occupancy, 5 dealers per 4 tables). "
           f"The plan staffs to the P{plan.quantile * 100:.0f} forecast: with call-ins at 3x wage, an uncovered hour "
           "costs 2 wages more than a scheduled one and an idle hour wastes 1, so the newsvendor ratio is 2/3.", "",
           _md_table(bt, {"WAPE": ".3f", "MAE (players)": ".2f", "Under-staffed dealer-hours/wk": ".1f",
                          "Over-staffed dealer-hours/wk": ".1f"}), "",
           f"Rostering the forecast week: total cost {_money(s['total_cost'])}, coverage {s['coverage_rate']:.1%}, "
           f"{s['cross_role_shifts']} supervisor shifts redeployed to dealing, solved in {s['runtime_s']:.1f}s "
           f"({s['n_variables']:,} variables).", "", "![Forecast](figures/forecast_week.png)", ""]

    # 3. demand models
    w = workload_staff(100_000, 25, 160, 0.90)
    e80 = erlang_staff(50, 10, 2, 0.80)
    e95 = erlang_staff(50, 10, 2, 0.95, shrinkage=0.2)
    tg = table_game_dealers(120, min_tables=1)
    results["demand"] = {"workload": w, "erlang_80": e80, "erlang_95_shrink": e95, "table_game": tg}
    md += ["## 3. Labor demand models", "",
           "| Model | Inputs | Staff required |", "|---|---|---|",
           f"| Workload (textile) | 100,000 garments/month, 25/hour, 160 h/worker, 90% efficiency | {w['workers']} workers ({w['exact']:.1f}) |",
           f"| Erlang C (retail) | 50 customers/hour, 10 min each, 80% served within 2 min | {e80['servers']} associates on duty |",
           f"| Erlang C (retail) | same, 95% within 2 min, 20% shrinkage | {e95['servers']} on duty, {e95['scheduled']} scheduled |",
           f"| Table game (casino) | 300 guests, 40% at blackjack | {tg['tables']} tables, {tg['dealers']} dealers |", ""]

    # 4. experiments
    dexp = ex.analyze_dealer_experiment(syn.dealer_experiment())
    texp = ex.analyze_textile_workflow(syn.textile_workflow_panel())
    rexp = ex.analyze_retail_staffing(syn.retail_staffing_test())
    fig_did(syn.textile_workflow_panel(), figs / "did_textile.png")
    results["experiments"] = {"dealer": dexp, "textile": texp, "retail": rexp}
    a, n, eco = dexp["adjusted_ols"], dexp["naive_welch"], dexp["economics_per_peak_hour"]
    md += ["## 4. Evaluating optimization changes", "",
           "Each dataset has a known planted effect, so the estimates can be checked against the truth.", "",
           "| Experiment | Planted effect | Method | Estimate (95% CI) |", "|---|---|---|---|",
           f"| Casino: +15% dealers at peak | +6.0 hands/h | Welch t-test | {n['estimate']:.2f} ({n['ci_low']:.2f}, {n['ci_high']:.2f}) |",
           f"| | | OLS + controls, table-clustered SE | {a['estimate']:.2f} ({a['ci_low']:.2f}, {a['ci_high']:.2f}) |",
           f"| Textile: new workflow | +1.5 units/labor-h | Naive before/after (treated lines) | {texp['naive_before_after']['estimate']:.2f} ({texp['naive_before_after']['ci_low']:.2f}, {texp['naive_before_after']['ci_high']:.2f}) |",
           f"| | | Difference-in-differences | {texp['diff_in_diff']['estimate']:.2f} ({texp['diff_in_diff']['ci_low']:.2f}, {texp['diff_in_diff']['ci_high']:.2f}) |",
           f"| Retail: dynamic staffing | log-odds +0.12 | Binomial GLM, store-clustered SE | {rexp['estimate']:.3f} ({rexp['ci_low']:.3f}, {rexp['ci_high']:.3f}) |", "",
           f"The naive before/after test credits the common upward trend to the workflow and overstates the effect by "
           f"{texp['naive_before_after']['estimate'] / 1.5 - 1:.0%}; the pre-period trend difference between groups is "
           f"{texp['pretrend']['estimate']:.3f}/week (p = {texp['pretrend']['p_value']:.2f}), consistent with parallel trends. "
           f"Retail conversion rises from {rexp['baseline_conversion']:.1%} to an implied {rexp['implied_conversion']:.1%}.", "",
           f"**Does the casino change pay?** At {eco['assumptions']['avg_players']} players per table, "
           f"HK${eco['assumptions']['avg_bet']:.0f} average bet and a {eco['assumptions']['house_edge']:.0%} house edge, "
           f"the extra hands are worth {_money(eco['extra_theoretical_win'])} theoretical win per peak hour across "
           f"{eco['assumptions']['tables']} tables, against {_money(eco['extra_labor_cost'])} of extra dealer pay "
           f"(net {_money(eco['net'])}; {_money(eco['net_at_ci_low'])} at the lower confidence bound). "
           f"Detecting a 3 hands/h effect needs {dexp['n_per_arm_for_3_hands']} table-shifts per arm.", "",
           "![Diff-in-diff](figures/did_textile.png)", ""]

    # 5. KPI monitoring
    th = kpi.load_thresholds(CONFIG_DIR / "kpi_thresholds.yaml")
    tx = kpi.textile_kpis(syn.textile_daily_kpis())
    det = kpi.compare_detection(tx, "line", "L4", "date", "defects", "units", threshold=0.05)
    wear = pd.Timestamp("2026-01-01") + pd.Timedelta(days=60)
    fig_spc(det, 0.05, wear, figs / "spc_line_L4.png")
    flags = pd.concat([
        kpi.flag(tx, th["textile"], "line", "date").assign(operation="Textile"),
        kpi.flag(kpi.retail_kpis(syn.retail_daily_kpis()), th["retail"], "store", "date").assign(operation="Retail"),
        kpi.flag(syn.casino_shift_kpis(), th["casino"], "dealer", "shift").assign(operation="Casino"),
    ], ignore_index=True)
    flags["first_flagged"] = [f"shift {v}" if isinstance(v, (int, np.integer)) else str(v)[:10] for v in flags.first_flagged]
    flags["unit"] = flags[["line", "store", "dealer"]].bfill(axis=1).iloc[:, 0]
    results["kpi"] = {"spc_first_signal": str(det["spc_first_signal"]),
                      "threshold_first_flag": str(det["threshold_first_flag"]), "flags": len(flags)}
    sig = det["spc_first_signal"]
    md += ["## 5. Tracking productivity and flagging operations", "",
           f"Tool wear starts on line L4 on {wear.date()}. The 5% defect threshold "
           f"{'never fires' if det['threshold_first_flag'] is None else 'first fires on ' + str(det['threshold_first_flag'].date())}; "
           f"the p-chart signals on {sig.date() if sig is not None else 'n/a'} "
           f"({(sig - wear).days if sig is not None else 'n/a'} days after the drift begins).", "",
           "![SPC](figures/spc_line_L4.png)", "",
           "Threshold rules from `configs/kpi_thresholds.yaml` flag these units:", "",
           _md_table(flags[["operation", "unit", "rule", "first_flagged", "value", "action"]].rename(columns={
               "operation": "Operation", "unit": "Unit", "rule": "Rule", "first_flagged": "First flagged",
               "value": "Value", "action": "Action"}), {"Value": ".3g"}), ""]

    # 6. casino maths
    gs = cm.game_summary()
    results["casino_math"] = gs.to_dict("records")
    md += ["## 6. Casino mathematics", "",
           "Exact values (rational arithmetic, or full enumeration of an 8-deck baccarat shoe), checked against "
           "Monte Carlo in the tests.", "",
           _md_table(gs.rename(columns={"game": "Game", "bet": "Bet", "p_win": "P(win)", "house_edge": "House edge",
                                        "payback": "Payback"}), {"P(win)": ".4f", "House edge": ".3%", "Payback": ".3%"}), "",
           f"Blackjack: P(natural), 6 decks = {cm.blackjack_natural_probability(6):.4f}. Hitting hard 16 busts with "
           f"probability {cm.bust_probability(16, 1000):.3f} (infinite deck) or {cm.bust_probability(16, 1, [10, 6, 7]):.3f} "
           f"(single deck, 10-6 vs dealer 7 removed). Poker: a flush draw on the turn (9 outs of 46, "
           f"{9 / 46:.1%}) facing a 20 bet into 100 needs {cm.required_equity(100, 20):.1%} equity, so calling is "
           f"+EV ({cm.call_ev(100, 20, 9 / 46):+.2f} units).", ""]

    # 7. pro-forma
    pf = case_studies()
    results["proforma"] = {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in pf.items()}
    c = pf["cases"][["case", "incremental_profit", "break_even_uplift", "expected_uplift", "mc_p_loss"]].copy()
    c.columns = ["Case", "Incremental profit/day", "Break-even uplift", "Expected uplift", "P(loss)"]
    md += ["## 7. Pro-forma: is the extra labor worth it?", "",
           f"Textile training (+10% output, same 50 workers): +{_money(pf['textile']['incremental_profit'])}/day as stated, "
           f"+{_money(pf['textile_margin']['incremental_profit'])}/day at a 55% contribution margin; labor cost per unit falls from "
           f"{pf['textile']['labor_cost_per_unit_before']:.2f} to {pf['textile']['labor_cost_per_unit_after']:.2f}.", "",
           _md_table(c, {"Incremental profit/day": ",.0f", "Break-even uplift": ".1%", "Expected uplift": ".0%", "P(loss)": ".1%"}), "",
           f"Treating sales or gaming win as profit overstates the benefit by "
           f"{c.iloc[0, 1] / c.iloc[1, 1] - 1:.0%} (retail) and {c.iloc[2, 1] / c.iloc[3, 1] - 1:.0%} (casino). "
           "P(loss) comes from Monte Carlo over the uncertain uplift (normal, SD 10 points for retail and 8 for the "
           "casino). These examples keep the original case study's currency units.", ""]

    (out / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    return results


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
