"""Performance metrics for return series.

The original `report_metrics(ret)` (annual_ret, annual_std, annual_sharpe)
is the developer's canonical starter. Phase 0 extends it with six additional
keys per Project_brief.md §13.7, while preserving the three original keys
byte-identically (Constitution Principle V — verified by
tests/unit/test_metrics.py:test_original_keys_regression).
"""
import numpy as np


def report_metrics(ret):
    """Compute a dictionary of annualized performance metrics from a daily
    returns series.

    Parameters
    ----------
    ret : array-like
        1-D numpy array (or anything `np.asarray`-compatible) of daily
        decimal returns. No NaN/Inf — the caller is responsible for cleaning.

    Returns
    -------
    dict[str, float]
        Nine keys: the three original (annual_ret, annual_std,
        annual_sharpe) plus six new (downside_dev, sortino, max_drawdown,
        calmar, pct_positive, avg_p_over_l). See
        specs/001-phase-0-skeleton-data/contracts/metrics_report_schema.md
        for the formulas and invariants.
    """
    res = {}

    # ---- Original keys (Principle V — DO NOT MODIFY) ----
    res['annual_ret'] = np.mean(ret) * 252
    res['annual_std'] = np.std(ret) * np.sqrt(252)
    res['annual_sharpe'] = (np.mean(ret) / np.std(ret)) * np.sqrt(252)

    # ---- Extension keys (Phase 0, per brief §13.7) ----
    # Downside deviation — std of negative returns only, annualized
    res['downside_dev'] = np.std(ret[ret < 0]) * np.sqrt(252)

    # Sortino — Sharpe-like ratio using downside deviation
    res['sortino'] = (np.mean(ret) / np.std(ret[ret < 0])) * np.sqrt(252)

    # Maximum drawdown — max peak-to-trough decline of the equity curve
    equity = (1 + ret).cumprod()
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    res['max_drawdown'] = abs(drawdown.min())

    # Calmar — annualized return / max drawdown
    res['calmar'] = res['annual_ret'] / res['max_drawdown']

    # Hit rate — fraction of days with positive returns
    res['pct_positive'] = (ret > 0).mean()

    # Average profit / average loss — magnitude ratio of wins vs losses
    res['avg_p_over_l'] = ret[ret > 0].mean() / abs(ret[ret < 0].mean())

    return res
