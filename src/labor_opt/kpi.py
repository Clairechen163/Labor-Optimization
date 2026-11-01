"""Productivity KPIs, threshold flags and statistical process control.

Fixed thresholds ("flag if defect rate > 5%") are easy to explain but react late and fire on noise.
Control charts set limits from the process's own variation and add run rules, so a drifting process
is caught earlier. Both are provided so they can be compared on the same data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}


def add_productivity(df: pd.DataFrame, output: str, hours: str = "labor_hours", name: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out[name or f"{output}_per_labor_hour"] = out[output] / out[hours]
    return out


def textile_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["units_per_labor_hour"] = out.units / out.labor_hours
    out["defect_rate"] = out.defects / out.units
    return out


def retail_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sales_per_labor_hour"] = out.sales / out.labor_hours
    out["conversion"] = out.transactions / out.visitors
    out["avg_transaction_value"] = out.sales / out.transactions
    return out


def load_thresholds(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def flag(df: pd.DataFrame, rules: list[dict], entity: str, time: str) -> pd.DataFrame:
    """Apply threshold rules. A rule fires when `metric op threshold` holds for `consecutive` periods in a row.

    rule = {"metric": "hands_per_hour", "op": "<", "threshold": 60, "consecutive": 2, "target": 70}
    """
    flags = []
    for rule in rules:
        m, op, thr = rule["metric"], rule["op"], rule["threshold"]
        need = int(rule.get("consecutive", 1))
        for ent, g in df.sort_values(time).groupby(entity):
            breach = OPS[op](g[m].to_numpy(), thr)
            run = 0
            for i, b in enumerate(breach):
                run = run + 1 if b else 0
                if run == need:
                    row = g.iloc[i]
                    flags.append({entity: ent, "metric": m, "first_flagged": row[time], "value": row[m],
                                  "rule": f"{m} {op} {thr}" + (f" for {need} periods" if need > 1 else ""),
                                  "target": rule.get("target"), "action": rule.get("action", "")})
                    break
    return pd.DataFrame(flags)


def flag_table(items: list[tuple[str, pd.DataFrame, list[dict], str, str]]) -> pd.DataFrame:
    """Flags from several operations in one tidy table.

    items: (operation, data, rules, entity column, time column). Time columns may be dates or
    shift numbers; `first_flagged` is returned as text ("2026-03-21" or "shift 21") so the two
    never get mixed into one datetime column.
    """
    rows = []
    for operation, df, rules, entity, time in items:
        f = flag(df, rules, entity, time)
        for r in f.itertuples(index=False):
            when = getattr(r, "first_flagged")
            when = f"shift {when}" if isinstance(when, (int, np.integer)) else str(pd.Timestamp(when).date())
            rows.append({"operation": operation, "unit": getattr(r, entity), "rule": r.rule,
                         "first_flagged": when, "value": r.value, "target": r.target, "action": r.action})
    return pd.DataFrame(rows, columns=["operation", "unit", "rule", "first_flagged", "value", "target", "action"])


# ------------------------------------------------------------------ control charts
def p_chart(defects: np.ndarray, units: np.ndarray, baseline_periods: int | None = None) -> pd.DataFrame:
    """p-chart with limits for varying sample size. Centre line from the first `baseline_periods` points."""
    defects, units = np.asarray(defects, float), np.asarray(units, float)
    k = baseline_periods or len(defects)
    pbar = defects[:k].sum() / units[:k].sum()
    p = defects / units
    se = np.sqrt(pbar * (1 - pbar) / units)
    out = pd.DataFrame({"p": p, "center": pbar, "ucl": pbar + 3 * se, "lcl": np.clip(pbar - 3 * se, 0, None)})
    out["signal_rule1"] = (out.p > out.ucl) | (out.p < out.lcl)
    out["signal_rule2"] = _run_rule(out.p.to_numpy() > pbar, 8)
    out["signal"] = out.signal_rule1 | out.signal_rule2
    return out


def xmr_chart(x: np.ndarray, baseline_periods: int | None = None) -> pd.DataFrame:
    """Individuals (X-mR) chart: sigma estimated from the average moving range / 1.128."""
    x = np.asarray(x, float)
    k = baseline_periods or len(x)
    base = x[:k]
    center = base.mean()
    sigma = np.abs(np.diff(base)).mean() / 1.128
    out = pd.DataFrame({"x": x, "center": center, "ucl": center + 3 * sigma, "lcl": center - 3 * sigma})
    z = (x - center) / sigma
    out["signal_rule1"] = np.abs(z) > 3
    out["signal_rule2"] = _run_rule(x > center, 8) | _run_rule(x < center, 8)
    # Western Electric rule: 2 of 3 consecutive points beyond 2 sigma on the same side
    hi, lo = z > 2, z < -2
    r3 = np.zeros(len(x), bool)
    for i in range(2, len(x)):
        r3[i] = hi[i - 2:i + 1].sum() >= 2 or lo[i - 2:i + 1].sum() >= 2
    out["signal_rule3"] = r3
    out["signal"] = out.signal_rule1 | out.signal_rule2 | out.signal_rule3
    return out


def _run_rule(side: np.ndarray, n: int) -> np.ndarray:
    run, out = 0, np.zeros(len(side), bool)
    for i, s in enumerate(side):
        run = run + 1 if s else 0
        out[i] = run >= n
    return out


def first_signal(chart: pd.DataFrame, index) -> object | None:
    hits = np.flatnonzero(chart.signal.to_numpy())
    return index[hits[0]] if len(hits) else None


def compare_detection(df: pd.DataFrame, entity_col: str, entity: str, time_col: str,
                      defects: str, units: str, threshold: float, baseline_periods: int = 30) -> dict:
    """When does a fixed threshold fire vs a p-chart, for one line?"""
    g = df[df[entity_col] == entity].sort_values(time_col).reset_index(drop=True)
    rate = g[defects] / g[units]
    thr_hits = np.flatnonzero(rate.to_numpy() > threshold)
    chart = p_chart(g[defects], g[units], baseline_periods)
    return {"entity": entity,
            "threshold_first_flag": g[time_col].iloc[thr_hits[0]] if len(thr_hits) else None,
            "spc_first_signal": first_signal(chart, g[time_col]),
            "chart": chart.assign(**{time_col: g[time_col]})}
