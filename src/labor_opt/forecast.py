"""Hourly demand forecasting with a count regression, plus a staffing-aware backtest.

Model: negative-binomial GLM (log link) with hour-of-day and day-of-week effects, a Friday/Saturday
evening interaction, an event flag and a linear trend. Overdispersion is estimated from the data
(Pearson method of moments), which gives honest prediction quantiles for staffing buffers.

The backtest is rolling-origin: train on all weeks before week w, forecast week w, repeat. It is scored
both on forecast error (WAPE) and on what matters operationally: dealer-hours under- and over-staffed
when the forecast is turned into a dealer requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .demand import table_game_dealers


def design_matrix(df: pd.DataFrame, week0: float = 0.0) -> pd.DataFrame:
    X = pd.get_dummies(df["hour"].astype(int), prefix="h", drop_first=True, dtype=float)
    X = X.reindex(columns=[f"h_{h}" for h in range(1, 24)], fill_value=0.0)
    D = pd.get_dummies(df["dow"].astype(int), prefix="dow", drop_first=True, dtype=float)
    D = D.reindex(columns=[f"dow_{d}" for d in range(1, 7)], fill_value=0.0)
    X = pd.concat([X, D], axis=1)
    X["peak_evening"] = (((df["dow"] == 4) | (df["dow"] == 5)) & (df["hour"] >= 19)).astype(float).to_numpy()
    X["event"] = df["event"].astype(float).to_numpy()
    X["trend"] = (df["week"].to_numpy() - week0) / 10.0
    X.insert(0, "const", 1.0)
    X.index = df.index
    return X


@dataclass
class CountForecaster:
    params: pd.Series | None = None
    alpha: float = 0.0

    def fit(self, df: pd.DataFrame, target: str = "players") -> "CountForecaster":
        X = design_matrix(df)
        y = df[target].to_numpy()
        pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        mu = pois.fittedvalues.to_numpy()
        # NB2 alpha by method of moments: Var = mu + alpha * mu^2
        self.alpha = float(max(0.0, np.mean(((y - mu) ** 2 - mu) / mu ** 2)))
        if self.alpha > 1e-6:
            nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=self.alpha)).fit()
            self.params = nb.params
        else:
            self.params = pois.params
        self.dispersion_ratio_ = float(pois.pearson_chi2 / pois.df_resid)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(design_matrix(df).to_numpy() @ self.params.to_numpy())

    def quantile(self, df: pd.DataFrame, q: float) -> np.ndarray:
        mu = self.predict(df)
        if self.alpha <= 1e-6:
            return stats.poisson.ppf(q, mu)
        n = 1.0 / self.alpha
        p = n / (n + mu)
        return stats.nbinom.ppf(q, n, p)


def seasonal_naive(history: pd.DataFrame, horizon: pd.DataFrame, target: str = "players") -> np.ndarray:
    """Same hour last week."""
    last = history.iloc[-168:][target].to_numpy()
    return np.resize(last, len(horizon)).astype(float)


def moving_average_seasonal(history: pd.DataFrame, horizon: pd.DataFrame, k: int = 4, target: str = "players") -> np.ndarray:
    """Average of the same hour over the last k weeks."""
    y = history[target].to_numpy()[-168 * k:].reshape(k, 168)
    return np.resize(y.mean(axis=0), len(horizon))


def wape(actual, forecast) -> float:
    actual = np.asarray(actual, float)
    return float(np.abs(actual - forecast).sum() / actual.sum())


def staffing_error(actual_players, planned_players, **dealer_kw) -> dict:
    need = table_game_dealers(np.asarray(actual_players), **dealer_kw)["dealers"]
    plan = table_game_dealers(np.asarray(planned_players), **dealer_kw)["dealers"]
    return {"under_dealer_hours": int(np.clip(need - plan, 0, None).sum()),
            "over_dealer_hours": int(np.clip(plan - need, 0, None).sum())}


def backtest(df: pd.DataFrame, test_weeks: int = 6, staffing_quantile: float = 0.75,
             dealer_kw: dict | None = None) -> pd.DataFrame:
    """Rolling-origin backtest over the last `test_weeks` weeks."""
    dealer_kw = dealer_kw or {"min_tables": 2}
    rows = []
    last_week = int(df.week.max())
    for w in range(last_week - test_weeks + 1, last_week + 1):
        train = df[df.week < w]
        test = df[df.week == w]
        model = CountForecaster().fit(train)
        preds = {
            "Seasonal naive (same hour last week)": seasonal_naive(train, test),
            "4-week same-hour average": moving_average_seasonal(train, test),
            "NB regression (mean)": model.predict(test),
            f"NB regression (P{round(staffing_quantile * 100)} for staffing)": model.quantile(test, staffing_quantile),
        }
        for name, f in preds.items():
            se = staffing_error(test.players, f, **dealer_kw)
            rows.append({"week": w, "method": name, "wape": wape(test.players, f),
                         "mae": float(np.abs(test.players - f).mean()), **se})
    return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    return (bt.groupby("method", sort=False)
              .agg(wape=("wape", "mean"), mae=("mae", "mean"),
                   under_dealer_hours_per_week=("under_dealer_hours", "mean"),
                   over_dealer_hours_per_week=("over_dealer_hours", "mean"))
              .reset_index())


def forecast_requirements(df: pd.DataFrame, horizon_week: int | None = None, quantile: float = 0.75,
                          role: str = "dealer", dealer_kw: dict | None = None) -> pd.DataFrame:
    """Forecast next week's pit traffic and convert it into an hourly dealer requirement.

    Returns rows (role, day, hour, t, required, forecast_players) for one 168-hour week, ready for
    `Scenario.with_requirements` and the roster optimizer.
    """
    dealer_kw = dealer_kw or {"min_tables": 2}
    model = CountForecaster().fit(df)
    w = int(df.week.max()) + 1 if horizon_week is None else horizon_week
    future = pd.DataFrame({"week": w, "dow": np.repeat(np.arange(7), 24), "hour": np.tile(np.arange(24), 7), "event": 0})
    planned = model.quantile(future, quantile)
    dealers = table_game_dealers(planned, **dealer_kw)["dealers"]
    return pd.DataFrame({"role": role, "day": future.dow, "hour": future.hour, "t": future.dow * 24 + future.hour,
                         "required": dealers, "forecast_players": model.predict(future), "planned_players": planned})
