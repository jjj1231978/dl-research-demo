"""Volatility-targeting helper (Phase 1, FR-006).

Rescales a raw signal so its realised annualised volatility matches a
target. Per Lim/Zohren/Roberts 2019 Eq. (1) — the σ_target=15% scaling
applied to all five strategies before the Exhibit 3 / Exhibit 4 figures.

The math:

    realised_vol_t = EWMA-std of past returns × sqrt(252)
    scaled_position_t = raw_signal_t × (σ_target / realised_vol_t)

Where ``returns`` is taken to be ``raw_signal * future_return`` —
i.e., the realised strategy-return one step ago. For position-only
inputs (where the underlying asset return isn't passed), callers should
multiply by the asset returns first and pass that compound series.

For the Phase 1 momentum page, the strategy daily return panel
(`data/backtests/momentum_results.parquet`, E4) stores both
`vol_scaling=True` and `vol_scaling=False` rows so downstream
visualizations don't need to recompute the rescaling at view time.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

SignalLike = Union[pd.Series, pd.DataFrame]


def vol_target(
    signal: SignalLike,
    target_vol: float = 0.15,
    ewma_span: int = 60,
    trading_days: int = 252,
) -> SignalLike:
    """Rescale ``signal`` so its realised annualised vol matches ``target_vol``.

    Args:
        signal: Daily strategy return series (or DataFrame of series).
            Interpreted as already-computed realised P&L per day —
            *not* a raw position; if the caller has only positions and
            asset returns, multiply them first.
        target_vol: Annualised volatility target (default 0.15 = 15%
            per Lim et al. 2019 Eq. 1).
        ewma_span: EWMA half-life span for realised-vol estimation
            (default 60 trading days per brief §13.5).
        trading_days: Annualisation factor (default 252).

    Returns:
        Same type/shape as ``signal``. The first ``ewma_span`` rows are
        unreliable (EWMA warmup); callers may want to drop them.
    """
    if target_vol <= 0:
        raise ValueError(f"target_vol must be positive; got {target_vol}")
    if ewma_span < 2:
        raise ValueError(f"ewma_span must be >= 2; got {ewma_span}")

    # Use t-1 vol so the scaling at time t doesn't peek at the t-th
    # return (avoids look-ahead bias). EWMA std is already a smoothed
    # estimator, but the .shift(1) makes the no-look-ahead explicit.
    realised_vol = signal.ewm(span=ewma_span, adjust=False).std().shift(1)
    realised_vol_annual = realised_vol * np.sqrt(trading_days)

    # Avoid divide-by-zero: where realised vol is 0/NaN, scale is NaN
    # (caller can fillna(0) if they want a flat position).
    scale = target_vol / realised_vol_annual
    return signal * scale
