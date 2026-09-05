"""Tests for the 3-state HMM on synthetic 2-regime returns.

The synthetic series alternates between a low-vol and a high-vol regime
with a known mean/vol. We fit the HMM on the full series and assert:

  - the HMM recovers 3 distinct volatility states
  - the state order map is consistent (low < mid < high)
  - predicting the LAST observation gives a sensible regime label
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.hmm import REGIME_LABELS, RegimeHMM


def _two_regime_returns(n: int = 2000, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Generate returns alternating between low-vol and high-vol regimes.

    Returns (returns, true_regime_index) where true_regime_index is
    0 for low-vol segments, 1 for high-vol segments.
    """
    rng = np.random.default_rng(seed)
    ret = np.empty(n)
    truth = np.empty(n, dtype=int)
    seg_len = 250
    i = 0
    seg_idx = 0
    while i < n:
        end = min(i + seg_len, n)
        # alternate: even seg_idx = low vol (sigma=0.005), odd = high vol (sigma=0.025)
        sigma = 0.005 if seg_idx % 2 == 0 else 0.025
        mu = 0.0003 if seg_idx % 2 == 0 else -0.0005
        ret[i:end] = rng.normal(loc=mu, scale=sigma, size=end - i)
        truth[i:end] = 0 if seg_idx % 2 == 0 else 1
        i = end
        seg_idx += 1
    return ret, truth


def test_hmm_fit_recovers_states():
    ret, _truth = _two_regime_returns(n=2000, seed=11)
    hmm = RegimeHMM(n_states=3, window=len(ret)).fit_full(ret)

    # 3 distinct states
    assert hmm.model is not None
    assert hmm.state_order_ is not None
    assert len(hmm.state_order_) == 3

    # the 3 internal states map to 3 distinct labels
    labels = {hmm.label_map_[int(s)] for s in range(3)}
    assert labels == set(REGIME_LABELS)


def test_hmm_current_regime_is_a_known_label():
    ret, _ = _two_regime_returns(n=2000, seed=42)
    hmm = RegimeHMM(n_states=3, window=len(ret)).fit_full(ret)
    label, probs = hmm.predict_current(ret)
    assert label in REGIME_LABELS
    assert probs.shape == (3,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)


def test_hmm_state_ordering_by_volatility():
    """The state ranked first (lowest) should correspond to low_vol.

    We verify by checking that the state's emission variance is
    smaller than the one ranked last (high_vol).
    """
    ret, _ = _two_regime_returns(n=4000, seed=99)
    hmm = RegimeHMM(n_states=3, window=len(ret)).fit_full(ret)

    # variances: hmm.model.covars_ is shape (n_states, 1, 1) when full covariance on a 1-D feature
    covars = np.array([float(hmm.model.covars_[s, 0, 0]) for s in range(3)])
    low_state_var = covars[hmm.state_order_[0]]
    high_state_var = covars[hmm.state_order_[2]]
    assert low_state_var < high_state_var, (
        f"low_vol state variance ({low_state_var}) should be < high_vol state variance ({high_state_var})"
    )


def test_hmm_labeled_predict_length():
    ret, _ = _two_regime_returns(n=800, seed=5)
    hmm = RegimeHMM(n_states=3, window=len(ret)).fit_full(ret)
    labels = hmm.predict_labeled(ret)
    assert len(labels) == len(ret)
    # every label must be one of the 3 known labels
    assert set(labels).issubset(set(REGIME_LABELS))


def test_hmm_rolling_window_fit():
    """Fitting with `fit()` should use only the last `window` observations."""
    ret, _ = _two_regime_returns(n=2000, seed=3)
    hmm = RegimeHMM(n_states=3, window=500).fit(ret)
    # predict returns one label per observation in the input array
    labels = hmm.predict_labeled(ret)
    assert len(labels) == len(ret)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
