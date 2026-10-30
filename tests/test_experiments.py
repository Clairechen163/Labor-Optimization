import numpy as np
import pytest
from scipy import stats

from labor_opt import experiments as ex
from labor_opt import synthetic as syn


def test_dealer_experiment_recovers_planted_effect():
    r = ex.analyze_dealer_experiment(syn.dealer_experiment(effect_hands=6.0))
    adj = r["adjusted_ols"]
    assert adj["ci_low"] < 6.0 < adj["ci_high"]
    assert adj["se"] < r["naive_welch"]["se"]           # controls tighten the estimate
    assert r["economics_per_peak_hour"]["net"] > 0


def test_textile_did_vs_naive():
    r = ex.analyze_textile_workflow(syn.textile_workflow_panel(effect_uph=1.5))
    did, naive = r["diff_in_diff"], r["naive_before_after"]
    assert did["ci_low"] < 1.5 < did["ci_high"]
    assert naive["estimate"] > 1.5 + 0.5                # naive before/after absorbs the common trend
    assert r["pretrend"]["p_value"] > 0.05
    assert r["defects_glm"]["odds_ratio"] < 1


def test_retail_conversion_effect():
    r = ex.analyze_retail_staffing(syn.retail_staffing_test(effect_logit=0.12))
    assert r["ci_low"] < 0.12 < r["ci_high"]
    assert r["implied_conversion"] > r["baseline_conversion"]


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 40), rng.normal(0, 2, 60)
    r = ex.welch_ttest(a, b)
    assert r["p_value"] == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue)


def test_sample_size():
    assert ex.sample_size(0.5, 1.0) == 64             # textbook: d = 0.5, alpha 0.05, power 0.8
