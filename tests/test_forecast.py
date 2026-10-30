import numpy as np
import pytest

from labor_opt.forecast import CountForecaster, backtest, forecast_requirements, summarize_backtest
from labor_opt.pipeline import PIT, newsvendor_quantile
from labor_opt.synthetic import pit_traffic


@pytest.fixture(scope="module")
def traffic():
    return pit_traffic(weeks=20, seed=42)


def test_regression_beats_baselines(traffic):
    s = summarize_backtest(backtest(traffic, test_weeks=3, dealer_kw=PIT)).set_index("method")
    nb = s.loc["NB regression (mean)", "wape"]
    assert nb < s.loc["Seasonal naive (same hour last week)", "wape"]
    assert nb < s.loc["4-week same-hour average", "wape"]


def test_higher_quantile_reduces_understaffing(traffic):
    s = summarize_backtest(backtest(traffic, test_weeks=3, staffing_quantile=0.8, dealer_kw=PIT)).set_index("method")
    assert s.loc["NB regression (P80 for staffing)", "under_dealer_hours_per_week"] < \
        s.loc["NB regression (mean)", "under_dealer_hours_per_week"]


def test_overdispersion_detected(traffic):
    m = CountForecaster().fit(traffic)
    assert m.alpha > 0 and m.dispersion_ratio_ > 1
    test = traffic[traffic.week == traffic.week.max()]
    assert (m.quantile(test, 0.9) >= np.floor(m.predict(test))).all()


def test_requirements_shape(traffic):
    req = forecast_requirements(traffic, quantile=2 / 3, dealer_kw=PIT)
    assert len(req) == 168
    assert req.t.tolist() == list(range(168))
    assert req.required.min() >= 3 and req.required.max() <= 10


def test_newsvendor():
    assert newsvendor_quantile(2, 1) == pytest.approx(2 / 3)
