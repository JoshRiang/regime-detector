"""3-state Gaussian HMM for volatility regimes.

State ordering convention (stable across fits):
  - the state with the LOWEST  absolute mean of |return| is `low_vol`
  - the state with the HIGHEST absolute mean of |return| is `high_vol`
  - the remaining state is `mid_vol`

We order by absolute mean because regime detection is about *variance*,
not drift. A state with near-zero mean but large variance is still a
high-vol regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


REGIME_LABELS = ("low_vol", "mid_vol", "high_vol")


class RegimeHMM:
    """3-state Gaussian HMM fit on the last `window` returns."""

    def __init__(
        self,
        n_states: int = 3,
        window: int = 252,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int | None = 42,
    ):
        if n_states != 3:
            raise ValueError(
                f"This project assumes n_states=3 (low/mid/high). Got {n_states}."
            )
        self.n_states = n_states
        self.window = window
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: GaussianHMM | None = None
        self.state_order_: np.ndarray | None = None  # mapping internal_state -> label_idx
        self.label_map_: dict[int, str] | None = None

    # ----- fit ----------------------------------------------------------

    def fit(self, returns: np.ndarray | pd.Series) -> "RegimeHMM":
        """Fit on the last `window` observations of `returns`."""
        x = _as_2d(returns)
        x = x[-self.window:]
        if len(x) < self.window // 2:
            raise ValueError(
                f"Not enough observations to fit HMM: got {len(x)}, need >= {self.window // 2}."
            )

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(x)
        self.model = model
        self._build_label_map(x)
        return self

    def fit_full(self, returns: np.ndarray | pd.Series) -> "RegimeHMM":
        """Fit on the full series (not just the rolling window).

        Useful for tests that want a single fit on a long synthetic run.
        """
        x = _as_2d(returns)
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(x)
        self.model = model
        self._build_label_map(x)
        return self

    # ----- predict ------------------------------------------------------

    def predict(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """Return per-observation *internal* state assignments (0..n_states-1)."""
        if self.model is None:
            raise RuntimeError("Call .fit() before .predict().")
        x = _as_2d(returns)
        return self.model.predict(x)

    def predict_labeled(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """Per-observation regime labels (low_vol / mid_vol / high_vol)."""
        states = self.predict(returns)
        idx = self.state_order_
        # idx[internal_state] = 0/1/2 ; REGIME_LABELS[2] gives the string
        return np.array([REGIME_LABELS[idx[s]] for s in states], dtype=object)

    def predict_proba(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """Per-observation posterior state probabilities, shape (T, n_states).

        Columns are reordered so that column 0 == low_vol, 1 == mid_vol, 2 == high_vol.
        """
        if self.model is None:
            raise RuntimeError("Call .fit() before .predict_proba().")
        x = _as_2d(returns)
        # posteriors is shape (T, n_states) in the model's *internal* state order
        _, posteriors = self.model.score_samples(x)
        # reorder columns according to state_order_
        return posteriors[:, self.state_order_]

    def predict_current(
        self, returns: np.ndarray | pd.Series
    ) -> tuple[str, np.ndarray]:
        """The regime of the *last* observation + its probability vector.

        Returns (label, prob_vector) where prob_vector is ordered
        [low_vol, mid_vol, high_vol].
        """
        if self.model is None:
            raise RuntimeError("Call .fit() before .predict_current().")
        x = _as_2d(returns)
        last = x[-1:]
        _, post = self.model.score_samples(last)
        post = post[0]  # shape (n_states,)
        # pick the highest-prob internal state, then map to label
        internal = int(np.argmax(post))
        prob_vector = post[self.state_order_]
        label = REGIME_LABELS[self.state_order_[internal]]
        return label, prob_vector

    # ----- internals ----------------------------------------------------

    def _build_label_map(self, x: np.ndarray) -> None:
        """Assign low_vol / mid_vol / high_vol to the 3 internal states.

        We rank internal states by their **emission standard deviation**
        (the diagonal of covars_) — that's the actual quantity that
        distinguishes volatility regimes. Means are typically near zero
        so ranking by |mean| would be ambiguous.

        Mapping: rank 0 -> low_vol, rank 1 -> mid_vol, rank 2 -> high_vol.
        """
        covars = self.model.covars_  # shape depends on covariance_type
        # diagonal std for each state
        if self.covariance_type == "full":
            # covars_ shape (n_states, n_features, n_features)
            stds = np.sqrt(np.maximum(np.array([np.diag(c) for c in covars]).reshape(-1), 0.0))
        elif self.covariance_type == "diag":
            # covars_ shape (n_states, n_features)
            stds = np.sqrt(np.maximum(covars.reshape(-1), 0.0))
        elif self.covariance_type == "spherical":
            stds = np.sqrt(np.maximum(covars.reshape(-1), 0.0))
        elif self.covariance_type == "tied":
            # tied across states — diag of the shared cov matrix
            diag = np.diag(covars)
            stds = np.sqrt(np.maximum(diag, 0.0))
            stds = np.tile(stds, 3)
        else:
            raise ValueError(f"Unknown covariance_type: {self.covariance_type}")
        # stable sort: ascending std, then ascending index
        order = np.argsort(stds, kind="stable")
        self.state_order_ = order
        self.label_map_ = {int(internal): REGIME_LABELS[idx] for idx, internal in enumerate(order)}


def _as_2d(returns: np.ndarray | pd.Series) -> np.ndarray:
    """Return shape (T, 1) float array, regardless of input container."""
    if isinstance(returns, pd.Series):
        arr = returns.to_numpy(dtype=float)
    else:
        arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D returns, got shape {arr.shape}.")
    return arr
