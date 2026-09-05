"""GARCH(1,1) volatility forecast.

We use the `arch` library with Student-t innovations because equity
returns have fat tails. The headline number we expose is:

  - `forecast_one()`: 1-step-ahead conditional std in % per day, plus
    annualized vol in % per year.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class GARCHForecaster:
    """Fit GARCH(1,1) (Student-t) on a return series and forecast next-day vol."""

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "t",
        mean: str = "Zero",
        annualization: int = 252,
    ):
        self.p = p
        self.q = q
        self.dist = dist
        self.mean = mean
        self.annualization = annualization
        self.result_ = None  # arch ARCHModelResult

    def fit(self, returns: np.ndarray | pd.Series) -> "GARCHForecaster":
        """Fit GARCH(p,q) on `returns` (in raw decimal form, e.g. 0.01 = 1%)."""
        from arch import arch_model

        x = _to_series(returns) * 100.0  # arch works in % for numerical stability
        x = x.dropna()
        if len(x) < 100:
            raise ValueError(
                f"GARCH needs at least ~100 observations, got {len(x)}."
            )

        # arch can be noisy on stderr — silence its convergence chatter.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(
                x,
                mean=self.mean,
                vol="GARCH",
                p=self.p,
                q=self.q,
                dist=self.dist,
                rescale=False,
            )
            self.result_ = am.fit(disp="off", show_warning=False)
        return self

    def forecast_one(self) -> dict:
        """Forecast 1-step-ahead conditional variance and return it as %.

        Returns a dict with keys:
          - daily_vol_pct       : next-day std, in percent (e.g. 1.42)
          - annual_vol_pct      : annualized std, in percent
          - conditional_var     : h_{T+1} in (pct)^2
          - horizon             : always 1
        """
        if self.result_ is None:
            raise RuntimeError("Call .fit() before .forecast_one().")

        f = self.result_.forecast(horizon=1, reindex=False)
        # arch returns a DataFrame; the last row's `h.1` is the 1-step var.
        var_series = f.variance.iloc[-1]
        h1 = float(var_series.iloc[0])  # in (pct)^2
        daily_vol_pct = float(np.sqrt(max(h1, 0.0)))
        annual_vol_pct = daily_vol_pct * np.sqrt(self.annualization)
        return {
            "daily_vol_pct": daily_vol_pct,
            "annual_vol_pct": annual_vol_pct,
            "conditional_var": h1,
            "horizon": 1,
        }

    def conditional_vol(self) -> pd.Series:
        """The fitted in-sample conditional std series, in % (aligned to returns)."""
        if self.result_ is None:
            raise RuntimeError("Call .fit() before .conditional_vol().")
        vol = self.result_.conditional_volatility
        vol.name = "garch_vol_pct"
        return vol

    @property
    def params(self) -> pd.Series:
        """Estimated GARCH parameters (omega, alpha[1], beta[1], ...)."""
        if self.result_ is None:
            raise RuntimeError("Call .fit() before .params.")
        return self.result_.params


def _to_series(x: np.ndarray | pd.Series) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype(float)
    return pd.Series(np.asarray(x, dtype=float))
