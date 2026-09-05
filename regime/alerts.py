"""Regime change detection and human-readable alert formatting."""

from __future__ import annotations

import pandas as pd


def detect_regime_changes(
    labels: pd.Series,
    min_consecutive: int = 1,
) -> pd.DataFrame:
    """Return a DataFrame of regime transitions.

    A "change" is flagged when label[t] != label[t-1]. With
    `min_consecutive > 1` the change is only confirmed after that many
    bars in the new regime, which suppresses 1-bar flickers.

    Output columns:
      - date       : the date the new regime BEGINS
      - from_regime: previous regime label
      - to_regime  : new regime label
    """
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels)
    labels = labels.dropna()
    if labels.empty:
        return pd.DataFrame(columns=["date", "from_regime", "to_regime"])

    changes = labels != labels.shift(1)
    candidate_dates = labels.index[changes.fillna(False)]

    if min_consecutive <= 1:
        prev = labels.shift(1)
        rows = []
        for d in candidate_dates:
            from_regime = prev.loc[d]
            # Skip the very first observation: its "from_regime" is NaN by construction
            if pd.isna(from_regime):
                continue
            rows.append({
                "date": d,
                "from_regime": from_regime,
                "to_regime": labels.loc[d],
            })
        return pd.DataFrame(rows)

    # Confirm changes: require min_consecutive bars of the new regime.
    confirmed = []
    i = 0
    dates = list(labels.index)
    vals = list(labels.values)
    while i < len(dates):
        if i == 0 or vals[i] != vals[i - 1]:
            j = i
            while j < len(dates) and vals[j] == vals[i]:
                j += 1
            run_len = j - i
            if i > 0 and run_len >= min_consecutive:
                confirmed.append(dates[i])
            i = j
        else:
            i += 1

    prev = labels.shift(1)
    rows = []
    for d in confirmed:
        rows.append({
            "date": d,
            "from_regime": prev.loc[d],
            "to_regime": labels.loc[d],
        })
    return pd.DataFrame(rows)


def format_alert(date, from_regime: str, to_regime: str, prob: float | None = None) -> str:
    """Format a single alert line for printing / logging."""
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    base = f"[alert] regime change: {from_regime} -> {to_regime} on {d}"
    if prob is not None:
        base += f"  (p={prob:.3f})"
    return base
