"""Smoke tests for the alerts module."""

from __future__ import annotations

import pandas as pd

from regime.alerts import detect_regime_changes, format_alert


def test_detect_changes_simple():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    labels = pd.Series(["low_vol", "low_vol", "mid_vol", "mid_vol", "high_vol", "high_vol"], index=idx)
    changes = detect_regime_changes(labels, min_consecutive=1)
    assert len(changes) == 2
    assert list(changes["to_regime"]) == ["mid_vol", "high_vol"]
    assert list(changes["from_regime"]) == ["low_vol", "mid_vol"]


def test_detect_changes_with_min_consecutive():
    """A 1-bar flicker shouldn't count as a change with min_consecutive=2."""
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    labels = pd.Series(
        ["low_vol", "low_vol", "mid_vol", "low_vol", "low_vol", "mid_vol"],
        index=idx,
    )
    # With min_consecutive=1, there are 3 transitions
    # (low->mid at i=2, mid->low at i=3, low->mid at i=5).
    # The very first row is NaN, so 3 real transitions are reported.
    all_changes = detect_regime_changes(labels, min_consecutive=1)
    assert len(all_changes) == 3
    # With min_consecutive=2, the 1-bar mid_vol flicker at i=2 and the
    # 1-bar mid_vol at i=5 (last entry is alone, run_len=1) are filtered.
    # The mid->low transition at i=3 IS confirmed (low_vol then runs 2 bars).
    filtered = detect_regime_changes(labels, min_consecutive=2)
    assert len(filtered) == 1
    assert filtered.iloc[0]["to_regime"] == "low_vol"
    assert filtered.iloc[0]["from_regime"] == "mid_vol"


def test_format_alert():
    s = format_alert("2024-08-05", "mid_vol", "high_vol", prob=0.87)
    assert "mid_vol -> high_vol" in s
    assert "2024-08-05" in s
    assert "p=0.870" in s
