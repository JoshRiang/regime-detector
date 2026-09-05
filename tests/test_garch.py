"""Tests for the GARCH(1,1) forecaster on synthetic GARCH data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.garch import GARCHForecaster


def _simulate_garch(n: int, omega: float, alpha: float, beta: float,
                    seed: int = 17, dist: str = "normal") -> np.ndarray:
    """Simulate a GARCH(1,1) series with the given parameters.

    Returns the *return* series (in raw decimal form, NOT in %).
    """
    rng = np.random.default_rng(seed)
    if dist == "normal":
        z = rng.standard_normal(n)
    else:  # fat tails
        z = rng.standard_t(df=5, size=n) / np.sqrt(5 / (5 - 2))
    eps = np.empty(n)
    h = np.empty(n)
    h0 = omega / max(1 - alpha - beta, 1e-6)
    h[0] = h0
    eps[0] = z[0] * np.sqrt(h0)
    for t in range(1, n):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
        eps[t] = z[t] * np.sqrt(h[t])
    return eps


def test_garch_fit_basic():
    np.random.seed(0)
    omega, alpha, beta = 1e-5, 0.08, 0.90
    ret = _simulate_garch(n=2500, omega=omega, alpha=alpha, beta=beta,
                          seed=42, dist="normal") * 100.0  # in %

    g = GARCHForecaster().fit(ret)
    assert g.result_ is not None

    params = g.params
    # Parameter names for GARCH(1,1) with zero mean are: omega, alpha[1], beta[1]
    # arch sometimes prefixes "mu" depending on mean model; zero mean => no mu.
    assert "omega" in params.index
    assert any(p.startswith("alpha") for p in params.index)
    assert any(p.startswith("beta") for p in params.index)


def test_garch_forecast_positive_and_finite():
    np.random.seed(0)
    ret = _simulate_garch(n=2000, omega=1e-5, alpha=0.08, beta=0.90, seed=7) * 100.0

    g = GARCHForecaster().fit(ret)
    f = g.forecast_one()
    assert f["horizon"] == 1
    assert np.isfinite(f["daily_vol_pct"])
    assert f["daily_vol_pct"] > 0
    # annualized is daily * sqrt(252); with daily ~1% that's ~16%, plausible
    assert f["annual_vol_pct"] > f["daily_vol_pct"]


def test_garch_conditional_vol_length():
    ret = _simulate_garch(n=1500, omega=1e-5, alpha=0.08, beta=0.90, seed=11) * 100.0
    g = GARCHForecaster().fit(ret)
    vol = g.conditional_vol()
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(ret)
    # vol should be positive and finite everywhere
    assert np.isfinite(vol).all()
    assert (vol > 0).all()


def test_garch_recovers_alpha_beta_plausibly():
    """Fit GARCH on a long synthetic series; alpha+beta should be near truth.

    We allow generous tolerance because MLE on a single path is noisy,
    but the sum alpha+beta must be in (0, 1) and within ~0.1 of the true
    sum for a 2500-obs run with normal innovations.
    """
    np.random.seed(0)
    omega, alpha, beta = 1e-5, 0.08, 0.90
    ret = _simulate_garch(n=2500, omega=omega, alpha=alpha, beta=beta,
                          seed=1234) * 100.0

    g = GARCHForecaster().fit(ret)
    p = g.params
    a = float(next(v for k, v in p.items() if k.startswith("alpha")))
    b = float(next(v for k, v in p.items() if k.startswith("beta")))
    assert 0 < a < 1
    assert 0 < b < 1
    assert a + b < 1  # stationarity
    # not too far from the true sum
    assert abs((a + b) - (alpha + beta)) < 0.1, f"a+b={a+b:.3f}, expected near {alpha+beta:.3f}"


def test_garch_fat_tails_still_fits():
    """GARCH with Student-t innovations should fit even when returns are fat-tailed."""
    ret = _simulate_garch(n=2000, omega=1e-5, alpha=0.08, beta=0.90,
                          seed=99, dist="t") * 100.0
    g = GARCHForecaster().fit(ret)
    f = g.forecast_one()
    assert np.isfinite(f["daily_vol_pct"])
    assert f["daily_vol_pct"] > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
