"""CLI entry point: python -m regime --symbol SPY --start 2010-01-01

Prints current regime, vol forecast, and regime probability. Optionally
saves a 2-panel regime/vol chart.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .alerts import detect_regime_changes, format_alert
from .data import load_prices, load_returns
from .garch import GARCHForecaster
from .hmm import REGIME_LABELS, RegimeHMM


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="regime",
        description="Volatility regime detector: 3-state HMM + GARCH(1,1) on daily returns.",
    )
    p.add_argument("--symbol", default="SPY", help="Yahoo Finance ticker (default: SPY).")
    p.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD (default: 2010-01-01).")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today).")
    p.add_argument("--window", type=int, default=252, help="HMM rolling window in days (default: 252).")
    p.add_argument("--min-consecutive", type=int, default=2,
                   help="Min consecutive bars before a regime change is confirmed (default: 2).")
    p.add_argument("--plot", default=None, help="If set, save the regime/vol chart to this path.")
    p.add_argument("--quiet", action="store_true", help="Suppress non-essential output.")
    args = p.parse_args(argv)

    log = (lambda *a, **kw: None) if args.quiet else print

    try:
        prices = load_prices(args.symbol, start=args.start, end=args.end)
        returns = load_returns(args.symbol, start=args.start, end=args.end, price=prices)
    except Exception as e:
        print(f"[regime] failed to load data for {args.symbol}: {e}", file=sys.stderr)
        return 1

    log(f"[regime] symbol={args.symbol} start={args.start} n_obs={len(returns)}")

    # ---- HMM -------------------------------------------------------------
    hmm = RegimeHMM(n_states=3, window=args.window)
    try:
        hmm.fit(returns)
    except Exception as e:
        print(f"[regime] HMM fit failed: {e}", file=sys.stderr)
        return 2

    log(f"[regime] state labels (by mean): "
        f"{{{hmm.label_map_[0]!r}: '0', {hmm.label_map_[1]!r}: '1', {hmm.label_map_[2]!r}: '2'}}"
        .replace("'0'", "0").replace("'1'", "1").replace("'2'", "2"))

    labeled = pd.Series(hmm.predict_labeled(returns), index=returns.index, name="regime")
    current, probs = hmm.predict_current(returns)
    p_low, p_mid, p_high = probs.tolist()
    log(f"[regime] current regime: {current.upper()}  (p={probs.max():.3f})")
    log(f"[regime] regime probabilities: low={p_low:.3f}  mid={p_mid:.3f}  high={p_high:.3f}")

    # ---- GARCH -----------------------------------------------------------
    garch = GARCHForecaster()
    try:
        garch.fit(returns)
    except Exception as e:
        print(f"[regime] GARCH fit failed: {e}", file=sys.stderr)
        forecast = {"daily_vol_pct": float("nan"), "annual_vol_pct": float("nan"),
                    "conditional_var": float("nan"), "horizon": 1}
    else:
        forecast = garch.forecast_one()
    log(f"[garch] next-day vol forecast: {forecast['daily_vol_pct']:.2f}% "
        f"(annualized {forecast['annual_vol_pct']:.1f}%)")

    # ---- Regime changes --------------------------------------------------
    changes = detect_regime_changes(labeled, min_consecutive=args.min_consecutive)
    n_changes = len(changes)
    log(f"[regime] {n_changes} regime changes detected "
        f"(min_consecutive={args.min_consecutive}).")

    # Print the most recent change, if any, plus the one before for context.
    if n_changes:
        last = changes.iloc[-1]
        log(format_alert(last["date"], last["from_regime"], last["to_regime"], prob=p_high))
        if n_changes >= 2:
            prev = changes.iloc[-2]
            log(format_alert(prev["date"], prev["from_regime"], prev["to_regime"]))

    # ---- Plot ------------------------------------------------------------
    if args.plot:
        from .visualize import plot_regime
        garch_vol = garch.conditional_vol() if garch.result_ is not None else None
        plot_regime(
            price=prices,
            regime_labels=labeled,
            garch_vol_pct=garch_vol,
            current_regime=current,
            forecast_daily_vol_pct=forecast["daily_vol_pct"],
            out=args.plot,
            title=f"{args.symbol} — Volatility Regime",
        )
        log(f"[regime] chart saved to {args.plot}")

    # ---- Exit code -------------------------------------------------------
    # 0 if low/mid vol (calm), 3 if high vol (alert). Lets cron/alerting
    # systems react to regime transitions without parsing text.
    return 3 if current == "high_vol" else 0


if __name__ == "__main__":
    raise SystemExit(main())
