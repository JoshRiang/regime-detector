"""2-panel matplotlib chart: price colored by regime + GARCH vol forecast."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# A muted palette so regimes are distinguishable but not garish.
REGIME_COLORS = {
    "low_vol": "#2ca02c",   # green
    "mid_vol": "#1f77b4",   # blue
    "high_vol": "#d62728",  # red
}


def plot_regime(
    price: pd.Series,
    regime_labels: pd.Series,
    garch_vol_pct: pd.Series | None = None,
    current_regime: str | None = None,
    forecast_daily_vol_pct: float | None = None,
    out: str | Path | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a 2-panel chart.

    Top panel:
      Price line colored by regime. To keep it readable we draw the
      line in light grey and overlay colored vertical spans for each
      regime period, plus colored dots at the midpoint of each regime
      run so the legend reads cleanly.

    Bottom panel:
      GARCH in-sample conditional vol (%) as a line, with a horizontal
      marker for the 1-step-ahead forecast if provided.

    Returns (fig, axes).
    """
    # align indices
    if not price.index.equals(regime_labels.index):
        regime_labels = regime_labels.reindex(price.index)
    if garch_vol_pct is not None and not garch_vol_pct.index.equals(price.index):
        garch_vol_pct = garch_vol_pct.reindex(price.index)

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_price, ax_vol = axes

    # ---- top panel: price colored by regime ----------------------------
    ax_price.plot(price.index, price.values, color="#cccccc", lw=0.8, zorder=1)

    # Vertical spans per regime run
    runs = _runs(regime_labels)
    for start, end, label in runs:
        color = REGIME_COLORS.get(label, "#888888")
        ax_price.axvspan(start, end, color=color, alpha=0.18, lw=0, zorder=2)

    # Markers at run midpoints (so legend has distinct color swatches)
    handles = []
    seen = set()
    for start, end, label in runs:
        mid = start + (end - start) / 2
        # pick the price at the midpoint (use the nearest index)
        try:
            y = float(price.loc[start:end].iloc[len(price.loc[start:end]) // 2])
        except Exception:
            y = float(price.iloc[0])
        c = REGIME_COLORS.get(label, "#888888")
        h = ax_price.scatter([mid], [y], color=c, s=22, zorder=3,
                             edgecolor="black", linewidth=0.3)
        if label not in seen:
            handles.append((label, h))
            seen.add(label)

    for label, h in handles:
        ax_price.scatter([], [], color=h.get_facecolor()[0], s=22,
                         edgecolor="black", linewidth=0.3, label=label)
    ax_price.legend(loc="upper left", frameon=False, fontsize=9, ncol=3)

    title_str = title or "Volatility Regime"
    if current_regime is not None:
        title_str += f"   |   current: {current_regime.upper()}"
        if forecast_daily_vol_pct is not None:
            title_str += f"   |   GARCH(1,1) next-day vol: {forecast_daily_vol_pct:.2f}%"
    ax_price.set_title(title_str)
    ax_price.set_ylabel("Price")

    # ---- bottom panel: GARCH vol ---------------------------------------
    if garch_vol_pct is not None:
        vol = garch_vol_pct.dropna()
        ax_vol.plot(vol.index, vol.values, color="#1f1f1f", lw=0.9, label="GARCH(1,1) cond. vol")
        if forecast_daily_vol_pct is not None:
            last_date = vol.index[-1] if len(vol) else price.index[-1]
            # extend line to next date for the forecast point
            next_date = last_date + pd.tseries.offsets.BDay(1) if isinstance(last_date, pd.Timestamp) else last_date + 1
            ax_vol.plot([last_date, next_date], [vol.iloc[-1] if len(vol) else forecast_daily_vol_pct, forecast_daily_vol_pct],
                        color="#d62728", lw=1.2, ls="--", label=f"1-step forecast: {forecast_daily_vol_pct:.2f}%")
            ax_vol.scatter([next_date], [forecast_daily_vol_pct], color="#d62728", s=30, zorder=4,
                           edgecolor="black", linewidth=0.4)
        ax_vol.set_ylabel("GARCH vol (%)")
        ax_vol.legend(loc="upper left", frameon=False, fontsize=9)
    else:
        ax_vol.text(0.5, 0.5, "GARCH vol unavailable",
                    ha="center", va="center", transform=ax_vol.transAxes)

    ax_vol.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    if out is not None:
        fig.savefig(out, dpi=130)

    return fig, axes


def _runs(labels: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Return [(start, end, label), ...] for each contiguous run of `labels`."""
    labels = labels.dropna()
    if labels.empty:
        return []
    out = []
    cur_label = labels.iloc[0]
    cur_start = labels.index[0]
    prev_idx = labels.index[0]
    for idx, val in labels.items():
        if val != cur_label:
            out.append((cur_start, prev_idx, cur_label))
            cur_label = val
            cur_start = idx
        prev_idx = idx
    out.append((cur_start, prev_idx, cur_label))
    return out
