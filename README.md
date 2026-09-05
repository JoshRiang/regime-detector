# regime-detector

A volatility regime detector that combines a **3-state Gaussian Hidden Markov Model (HMM)** with a **GARCH(1,1)** volatility forecast to identify whether a market is currently in a **low / mid / high volatility regime** and to forecast next-day volatility.

## What it does

1. Loads daily price data for a ticker (default: `SPY`).
2. Computes log returns and rolling volatility features.
3. Fits a **3-state Gaussian HMM** on a rolling 252-day window of returns — states are labeled by their mean so the lowest-vol state is `low_vol`, the highest is `high_vol`, the middle is `mid_vol`.
4. Fits a **GARCH(1,1)** model (via the `arch` library) to produce a 1-step-ahead volatility forecast.
5. Detects regime changes and emits alerts.
6. Produces a 2-panel chart: price colored by regime on top, GARCH vol forecast on bottom.

## Layout

```
regime-detector/
├── regime/
│   ├── __init__.py
│   ├── __main__.py     # CLI entry point
│   ├── data.py         # price loader + returns + vol features
│   ├── hmm.py          # 3-state Gaussian HMM (fit / predict)
│   ├── garch.py        # GARCH(1,1) vol forecast
│   ├── alerts.py       # regime-change detection + alert formatting
│   └── visualize.py    # matplotlib 2-panel chart
├── tests/
│   ├── test_hmm.py
│   └── test_garch.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## CLI usage

```bash
# default: SPY from 2010-01-01
python -m regime

# custom symbol / start date
python -m regime --symbol SPY --start 2010-01-01
python -m regime --symbol ^VIX --start 2015-01-01 --window 252

# save the chart
python -m regime --symbol SPY --start 2010-01-01 --plot regime_spy.png
```

Sample output:

```
[regime] symbol=SPY start=2010-01-01 n_obs=3789
[regime] state labels (by mean): {0: 'low_vol', 1: 'mid_vol', 2: 'high_vol'}
[regime] current regime: HIGH_VOL  (p=0.872)
[regime] regime probabilities: low=0.041  mid=0.087  high=0.872
[garch] next-day vol forecast: 1.42% (annualized 22.6%)
[alert] regime change: mid_vol → high_vol on 2024-08-05
```

## Programmatic use

```python
from regime.data import load_returns
from regime.hmm import RegimeHMM
from regime.garch import GARCHForecaster

ret = load_returns("SPY", start="2010-01-01")

hmm = RegimeHMM(n_states=3, window=252).fit(ret)
regime, probs = hmm.predict_current(ret)

garch = GARCHForecaster().fit(ret)
forecast = garch.forecast_one()
```

## Tech notes

- **HMM** — 3 states, Gaussian emissions, full covariance. Fitted on the last `window` (default 252) returns so the regime call reflects the *current* market. State means are used to assign `low_vol` / `mid_vol` / `high_vol` labels so the API is stable even when the model relabels its internal states.
- **GARCH(1,1)** — `arch.arch_model(..., mean="Zero", vol="GARCH", p=1, q=1, dist="t")`. Student-t innovations because equity returns have fat tails. The 1-step-ahead conditional variance is annualized as `sqrt(h * 252)` for the headline number.
- **Regime change** — flagged when the most-likely state of the *current* observation differs from the previous. Alerts include the date, the transition, and current regime probability.

## Tests

```bash
python -m pytest tests/ -v
```

Both tests use **synthetic data** so they don't need a network or any cached prices:

- `test_hmm.py` — generates returns drawn from two Gaussians with different variances and verifies the model recovers distinct volatility states.
- `test_garch.py` — generates a synthetic GARCH(1,1) series and verifies the fitted parameters are within plausible bounds of the true ones, plus that the forecast variance is positive.

## License

MIT
