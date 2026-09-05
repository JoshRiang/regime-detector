"""Data loading + return / volatility feature engineering.

The HMM expects a 1-D array of returns. The GARCH expects the same.
This module is the single source of truth for "what is a return" in the
project, so the rest of the package never talks to yfinance directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_prices(
    symbol: str = "SPY",
    start: str = "2010-01-01",
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.Series:
    """Download daily close prices for `symbol` from Yahoo Finance.

    Returns a pandas Series indexed by date with the close price.
    """
    import yfinance as yf

    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise ValueError(f"No price data for symbol={symbol!r} start={start!r}")

    # yfinance sometimes returns a MultiIndex columns on newer versions; collapse it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    close = df["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = symbol
    return close.dropna()


def load_returns(
    symbol: str = "SPY",
    start: str = "2010-01-01",
    end: str | None = None,
    price: pd.Series | None = None,
    log: bool = True,
) -> pd.Series:
    """Return daily returns (log returns by default).

    Pass `price` to skip the yfinance round-trip (used by tests + caching).
    """
    if price is None:
        price = load_prices(symbol, start=start, end=end)
    price = price.astype(float).dropna()

    if log:
        ret = np.log(price).diff()
    else:
        ret = price.pct_change()
    ret = ret.dropna()
    ret.name = f"{symbol}_return"
    return ret


def rolling_vol_features(
    returns: pd.Series,
    windows: tuple[int, ...] = (5, 21, 63),
) -> pd.DataFrame:
    """Rolling volatility features (annualized, in %).

    Useful as side features if you later want a multivariate HMM. The
    primary HMM in this project only uses the raw returns, but these
    features are exposed for callers that want them.
    """
    out = pd.DataFrame(index=returns.index)
    for w in windows:
        out[f"vol_{w}"] = returns.rolling(w).std() * np.sqrt(252) * 100.0
    return out.dropna()
