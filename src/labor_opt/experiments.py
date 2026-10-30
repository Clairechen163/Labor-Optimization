"""Evaluating optimization changes: experiment design and analysis.

Three designs, matched to how each change can actually be rolled out:

* casino: randomized at the table-day level -> two-sample comparison, then regression with controls
  and standard errors clustered by table (repeated observations of the same table are not independent)
* textile: some lines adopt a new workflow, others do not -> difference-in-differences with line and
  week fixed effects. A naive before/after test on the treated lines alone picks up the common trend.
* retail: dynamic staffing in half the stores from mid-period -> binomial GLM for conversion with
  store-clustered standard errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.power import TTestIndPower

from .demand import theoretical_win_per_hour


def welch_ttest(treated, control, alpha: float = 0.05) -> dict:
    a, b = np.asarray(treated, float), np.asarray(control, float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    q = stats.t.ppf(1 - alpha / 2, df)
    diff = a.mean() - b.mean()
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {"estimate": diff, "se": se, "ci_low": diff - q * se, "ci_high": diff + q * se,
            "t": float(t), "p_value": float(p), "cohens_d": diff / pooled, "n_treated": len(a), "n_control": len(b)}


def paired_ttest(after, before) -> dict:
    a, b = np.asarray(after, float), np.asarray(before, float)
    t, p = stats.ttest_rel(a, b)
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    q = stats.t.ppf(0.975, len(d) - 1)
    return {"estimate": d.mean(), "se": se, "ci_low": d.mean() - q * se, "ci_high": d.mean() + q * se,
            "t": float(t), "p_value": float(p), "n_pairs": len(d)}


def _coef(fit, name: str) -> dict:
    ci = fit.conf_int().loc[name]
    return {"estimate": float(fit.params[name]), "se": float(fit.bse[name]), "ci_low": float(ci[0]),
            "ci_high": float(ci[1]), "p_value": float(fit.pvalues[name])}


def regression_effect(df: pd.DataFrame, formula: str, term: str, cluster: str | None = None) -> dict:
    model = smf.ols(formula, data=df)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df[cluster])[0]}) if cluster else model.fit(cov_type="HC1")
    out = _coef(fit, term)
    out["r2"] = float(fit.rsquared)
    out["n"] = int(fit.nobs)
    return out


def binomial_effect(df: pd.DataFrame, successes: str, trials: str, rhs: str, term: str, cluster: str) -> dict:
    """Binomial GLM on (successes, failures) counts with cluster-robust standard errors."""
    X = patsy.dmatrix(rhs, df, return_type="dataframe")
    y = np.column_stack([df[successes], df[trials] - df[successes]])
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df[cluster])[0]})
    out = _coef(fit, term)
    out["odds_ratio"] = float(np.exp(out["estimate"]))
    return out


def bootstrap_diff_ci(treated, control, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a, b = np.asarray(treated, float), np.asarray(control, float)
    diffs = rng.choice(a, (n_boot, len(a))).mean(1) - rng.choice(b, (n_boot, len(b))).mean(1)
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def sample_size(effect: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Observations per arm to detect `effect` with a two-sample t-test."""
    n = TTestIndPower().solve_power(effect_size=effect / sd, alpha=alpha, power=power, alternative="two-sided")
    return int(np.ceil(n))


# ------------------------------------------------------------------ case analyses
def analyze_dealer_experiment(df: pd.DataFrame, dealers_per_pit: int = 9, extra_share: float = 0.15,
                              dealer_wage: float = 200.0, avg_bet: float = 500.0, house_edge: float = 0.01,
                              tables: int = 7) -> dict:
    """+15% dealers at peak: effect on hands/hour and satisfaction, and whether it pays for itself."""
    peak = df[df["shift"] == "peak"]
    naive = welch_ttest(peak.loc[peak.treated == 1, "hands_per_hour"], peak.loc[peak.treated == 0, "hands_per_hour"])
    naive["bootstrap_ci"] = bootstrap_diff_ci(peak.loc[peak.treated == 1, "hands_per_hour"],
                                              peak.loc[peak.treated == 0, "hands_per_hour"])
    adjusted = regression_effect(peak, "hands_per_hour ~ treated + players + C(game) + C(dow)", "treated", cluster="table")
    sat = regression_effect(peak, "satisfaction ~ treated + players + C(game) + C(dow)", "treated", cluster="table")
    sd = peak.loc[peak.treated == 0, "hands_per_hour"].std()
    avg_players = peak.players.mean()
    extra_win = tables * theoretical_win_per_hour(adjusted["estimate"], avg_players, avg_bet, house_edge)
    extra_win_low = tables * theoretical_win_per_hour(adjusted["ci_low"], avg_players, avg_bet, house_edge)
    extra_cost = dealers_per_pit * extra_share * dealer_wage
    return {
        "naive_welch": naive, "adjusted_ols": adjusted, "satisfaction_ols": sat,
        "n_per_arm_for_3_hands": sample_size(3.0, sd),
        "economics_per_peak_hour": {"extra_theoretical_win": extra_win, "extra_win_ci_low": extra_win_low,
                                    "extra_labor_cost": extra_cost, "net": extra_win - extra_cost,
                                    "net_at_ci_low": extra_win_low - extra_cost,
                                    "assumptions": {"tables": tables, "avg_players": round(avg_players, 2),
                                                    "avg_bet": avg_bet, "house_edge": house_edge}},
    }


def analyze_textile_workflow(df: pd.DataFrame) -> dict:
    treated = df[df.treated_line == 1]
    before = treated[treated.post == 0].groupby("line").units_per_labor_hour.mean()
    after = treated[treated.post == 1].groupby("line").units_per_labor_hour.mean()
    naive = paired_ttest(after.loc[before.index], before)
    did = regression_effect(df, "units_per_labor_hour ~ workflow_new + C(line) + C(week)", "workflow_new", cluster="line")
    # defects: binomial GLM with line and week effects
    d = binomial_effect(df, "defects", "units", "workflow_new + C(line) + C(week)", "workflow_new", cluster="line")
    # parallel-trends check: pre-period trend difference between treated and control lines
    pre = df[df.post == 0]
    pt = regression_effect(pre, "units_per_labor_hour ~ C(line) + week + week:treated_line", "week:treated_line", cluster="line")
    return {"naive_before_after": naive, "diff_in_diff": did, "defects_glm": d, "pretrend": pt}


def analyze_retail_staffing(df: pd.DataFrame) -> dict:
    c = binomial_effect(df, "transactions", "visitors", "dynamic_staffing + C(store) + post + weekend",
                        "dynamic_staffing", cluster="store")
    base = df[(df.treated_store == 1) & (df.post == 0)]
    p0 = base.transactions.sum() / base.visitors.sum()
    odds = p0 / (1 - p0) * np.exp(c["estimate"])
    c["baseline_conversion"] = float(p0)
    c["implied_conversion"] = float(odds / (1 + odds))
    return c
