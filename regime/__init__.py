"""regime — volatility regime detection via HMM + GARCH."""

from .hmm import RegimeHMM
from .garch import GARCHForecaster
from .data import load_returns, load_prices
from .alerts import detect_regime_changes, format_alert

__all__ = [
    "RegimeHMM",
    "GARCHForecaster",
    "load_returns",
    "load_prices",
    "detect_regime_changes",
    "format_alert",
]

__version__ = "0.1.0"
