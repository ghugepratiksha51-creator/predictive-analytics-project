"""
models.py
---------
Two families of forecasting model:

1. Regression-on-features models (use engineered calendar/lag/rolling
   features as X to predict the target):
     - Linear Regression   (fast, interpretable baseline)
     - Random Forest       (captures non-linear patterns/interactions)

2. Classical time-series model:
     - Holt-Winters triple exponential smoothing, implemented from
       scratch (level + trend + weekly seasonality). This is the
       standard statsmodels ExponentialSmoothing algorithm; it's
       reimplemented here directly because statsmodels isn't
       installable in this offline environment. Swap in
       `from statsmodels.tsa.holtwinters import ExponentialSmoothing`
       if you have internet access -- the interface below is a
       drop-in for that idea.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


FEATURE_COLS = [
    "marketing_spend", "promo_flag", "month", "day_of_week", "is_weekend",
    "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    "lag_1", "lag_7", "rolling_mean_7", "rolling_mean_30",
]


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


class HoltWintersForecaster:
    """
    Triple exponential smoothing (additive trend + additive weekly
    seasonality), fit with simple grid search over the smoothing
    parameters alpha (level), beta (trend), gamma (seasonality).
    """

    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.level_ = None
        self.trend_ = None
        self.season_ = None
        self.fitted_ = None

    def _fit_once(self, series: np.ndarray, alpha, beta, gamma):
        n = len(series)
        L = self.season_length
        level = np.zeros(n)
        trend = np.zeros(n)
        season = np.zeros(n + L)
        fitted = np.zeros(n)

        # Initialize
        level[0] = series[:L].mean()
        trend[0] = (series[L:2 * L].mean() - series[:L].mean()) / L
        for i in range(L):
            season[i] = series[i] - level[0]

        for t in range(1, n):
            season_idx = t % L
            prev_season = season[t - L] if t - L >= 0 else season[season_idx]
            fitted[t] = level[t - 1] + trend[t - 1] + prev_season

            level[t] = alpha * (series[t] - prev_season) + (1 - alpha) * (level[t - 1] + trend[t - 1])
            trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
            season[t] = gamma * (series[t] - level[t]) + (1 - gamma) * prev_season

        sse = np.sum((series[1:] - fitted[1:]) ** 2)
        return sse, level, trend, season, fitted

    def fit(self, series: np.ndarray):
        series = np.asarray(series, dtype=float)
        best_sse = np.inf
        best_params = (0.3, 0.1, 0.1)

        # Small grid search -- good enough for a from-scratch implementation
        for alpha in [0.1, 0.3, 0.5, 0.7]:
            for beta in [0.01, 0.05, 0.1, 0.2]:
                for gamma in [0.05, 0.1, 0.2, 0.3]:
                    sse, *_ = self._fit_once(series, alpha, beta, gamma)
                    if sse < best_sse:
                        best_sse = sse
                        best_params = (alpha, beta, gamma)

        self.alpha, self.beta, self.gamma = best_params
        _, level, trend, season, fitted = self._fit_once(series, *best_params)
        self.level_, self.trend_, self.season_, self.fitted_ = level, trend, season, fitted
        self.n_ = len(series)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        L = self.season_length
        last_level = self.level_[-1]
        last_trend = self.trend_[-1]
        preds = []
        for h in range(1, steps + 1):
            season_idx = (self.n_ - L + (h - 1) % L)
            s = self.season_[season_idx]
            preds.append(last_level + h * last_trend + s)
        return np.array(preds)
