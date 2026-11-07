# Regime Detector

Volatility regime detection: 3-state Gaussian HMM + GARCH(1,1) forecasting.

Fits a 3-state Gaussian HMM (`hmmlearn.GaussianHMM`) on daily log returns. Internal states are labelled `low_vol` / `mid_vol` / `high_vol` by their **emission standard deviation** (ascending), so the labels are stable across fits. Provides per-observation regime labels, posterior probability vectors (columns reordered to match label order), and `predict_current` returning the regime of the last observation with its probability vector.

`GARCHForecaster` (`arch` library, Student-t innovations) fits GARCH(1,1) and returns a 1-step-ahead daily vol forecast in percent plus the annualised figure. `detect_regime_changes` flags label transitions with an optional `min_consecutive` confirmation to suppress single-bar flickers. `format_alert` renders human-readable lines. CLI prints current regime, vol forecast, and recent regime changes; optionally saves a 2-panel chart (price coloured by regime + GARCH conditional vol with forecast marker). Exit code is 3 when the current regime is `high_vol` — convenient for cron alerting.

```bash
git clone https://github.com/JoshRiang/regime-detector.git
cd regime-detector
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# Detect SPY regime since 2010, save chart
python -m regime --symbol SPY --start 2010-01-01 --plot spy_regime.png

# Custom HMM window, tighter change confirmation
python -m regime --symbol QQQ --window 500 --min-consecutive 3

# Cron-friendly: exit code 3 when high vol
python -m regime --symbol SPY || echo "High vol alert"
```

```python
import pandas as pd
from regime.data import load_prices, load_returns
from regime.hmm import RegimeHMM, REGIME_LABELS
from regime.garch import GARCHForecaster
from regime.alerts import detect_regime_changes, format_alert

prices = load_prices('SPY', start='2010-01-01')
returns = load_returns('SPY', price=prices)

hmm = RegimeHMM(n_states=3, window=252).fit(returns)
labeled = pd.Series(hmm.predict_labeled(returns), index=returns.index)
current, probs = hmm.predict_current(returns)
print(f'Current regime: {current.upper()} (p={probs.max():.3f})')
print(f'Probs: low={probs[0]:.3f} mid={probs[1]:.3f} high={probs[2]:.3f}')

garch = GARCHForecaster().fit(returns)
forecast = garch.forecast_one()
print(f'Next-day vol: {forecast["daily_vol_pct"]:.2f}% (annualized {forecast["annual_vol_pct"]:.1f}%)')

changes = detect_regime_changes(labeled, min_consecutive=2)
if len(changes):
    last = changes.iloc[-1]
    print(format_alert(last['date'], last['from_regime'], last['to_regime']))
```

Project structure:

```
regime-detector/
├── regime/
│   ├── __init__.py     # exports: RegimeHMM, GARCHForecaster, load_returns, load_prices, detect_regime_changes, format_alert
│   ├── hmm.py          # RegimeHMM: 3-state Gaussian HMM, label by emission std, predict / predict_labeled / predict_proba / predict_current
│   ├── garch.py        # GARCHForecaster: GARCH(1,1) Student-t, forecast_one, conditional_vol
│   ├── data.py         # load_prices, load_returns (log), rolling_vol_features
│   ├── alerts.py       # detect_regime_changes (min_consecutive), format_alert
│   ├── visualize.py    # plot_regime: 2-panel chart (price by regime + GARCH vol + forecast marker)
│   └── __main__.py     # CLI: --symbol --start --end --window --min-consecutive --plot --quiet
├── tests/
│   ├── test_hmm.py     # state recovery, current label, ordering by volatility, labeled predict, rolling window fit
│   ├── test_garch.py   # param recovery, forecast positive/finite, conditional vol, fat-tails fit
│   └── test_alerts.py  # change detection with/without min_consecutive, alert formatting
├── requirements.txt    # numpy, pandas, yfinance, hmmlearn, arch, matplotlib
├── LICENSE
└── README.md
```

Testing:

```bash
pytest -v
```

Tests use synthetic GARCH and two-regime returns — no network required.

License: MIT — see [LICENSE](LICENSE).

Author: Joshua Riangkamang — [github.com/JoshRiang](https://github.com/JoshRiang)
