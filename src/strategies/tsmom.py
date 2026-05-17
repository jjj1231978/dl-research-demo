"""Time-series momentum reference signals (Phase 1).

Self-contained per FR-005 / spec.md §"Clarifications" — the math comes
directly from the cited papers + brief §13.5. The QIS_Commodities repo
implements a different signal family (multi-lookback `sign(momentum)`
with sector equal-weight) and is NOT referenced here.

Signal catalogue:

- ``long_only(prices)`` — constant `+1` position (vol-scaled at the
  portfolio layer by `src.strategies.vol_targeting`). Paper baseline.
- ``sgn_returns(prices, lookback_days=252)`` —
  `sign(prices.pct_change(lookback))`. Moskowitz / Ooi / Pedersen 2012,
  "Time Series Momentum", JFE 104(2). Annual lookback per paper Table 3.
- ``macd_signal(prices, short_span, long_span)`` — volatility-normalized
  MACD per Baz et al. 2015 "Dissecting Investment Strategies in the
  Cross Section and Time Series", §3.2.
- ``macd_ensemble(prices, shorts=(8, 16, 32), longs=(24, 48, 96))`` —
  mean across the three Baz timescales. The 8/24, 16/48, 32/96 triplet
  is exactly the configuration in Lim/Zohren/Roberts 2019 (arXiv
  1904.04912) §4.1.

Inputs accept either a pandas Series (single contract) or a long-format
DataFrame indexed by date with one column per contract. Output shape
matches input.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

PriceLike = Union[pd.Series, pd.DataFrame]


def long_only(prices: PriceLike) -> PriceLike:
    """Constant `+1` position, same shape as ``prices``.

    Per Lim et al. 2019 §4.1 (baseline) — the static buy-and-hold
    benchmark. Vol-scaling is the caller's responsibility (apply
    ``src.strategies.vol_targeting.vol_target`` after).
    """
    if isinstance(prices, pd.Series):
        return pd.Series(1.0, index=prices.index, name=prices.name)
    return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)


def sgn_returns(prices: PriceLike, lookback_days: int = 252) -> PriceLike:
    """Sign of past-`lookback_days` return.

    `X_t = sign(P_t / P_{t-L} - 1)` with `L = lookback_days`. The first
    `lookback_days` rows are NaN (no look-ahead leakage). Per
    Moskowitz/Ooi/Pedersen 2012 — the canonical TSMOM signal.
    """
    past_returns = prices.pct_change(lookback_days)
    return np.sign(past_returns)


def macd_signal(prices: PriceLike, short_span: int, long_span: int) -> PriceLike:
    """Single-timescale volatility-normalized MACD per Baz et al. 2015.

    ``signal_t = (EWMA(P, short_span) - EWMA(P, long_span))
                  / rolling_std(P, long_span)``

    The rolling-std denominator normalizes for price scale so the signal
    is dimensionless and comparable across contracts. The first
    `long_span` rows are NaN (rolling window warm-up).
    """
    short_ewm = prices.ewm(span=short_span, adjust=False).mean()
    long_ewm = prices.ewm(span=long_span, adjust=False).mean()
    denom = prices.rolling(long_span).std()
    return (short_ewm - long_ewm) / denom


def macd_ensemble(
    prices: PriceLike,
    shorts: tuple[int, ...] = (8, 16, 32),
    longs: tuple[int, ...] = (24, 48, 96),
) -> PriceLike:
    """Mean of `macd_signal` across the three Baz/Lim timescales.

    Default `shorts=(8,16,32)` / `longs=(24,48,96)` is the exact triplet
    used by Lim/Zohren/Roberts 2019 §4.1 "MACD Indicator" and Baz et al.
    2015 Table 1. NaN treatment: arithmetic mean over the three
    components, with `np.nan` in any component propagating to the output
    (the first `max(longs)` rows are NaN).
    """
    if len(shorts) != len(longs):
        raise ValueError(
            f"shorts and longs must have the same length; got "
            f"{len(shorts)} and {len(longs)}"
        )
    components = [
        macd_signal(prices, s, long_span)
        for s, long_span in zip(shorts, longs)
    ]
    if isinstance(prices, pd.DataFrame):
        arr = np.stack([c.to_numpy() for c in components], axis=-1)
        return pd.DataFrame(arr.mean(axis=-1), index=prices.index, columns=prices.columns)
    return pd.concat(components, axis=1).mean(axis=1)
