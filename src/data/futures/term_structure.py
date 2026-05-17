# =============================================================================
# Copied verbatim from ~/projects/QIS_Commodities/src/data/term_structure.py
# on 2026-05-17. Per /clarify OD-2: this is a copy-with-attribution; QIS
# remains the source of truth. Re-sync MANUALLY when the QIS source
# changes (no automated import); record the new date here.
#
# Single local edit: the QIS source imports `from src.data.contracts import ...`
# (QIS package layout). Rewritten to `from src.data.futures.contracts import ...`
# to match the Deep Finance Showcase layout. No other changes.
# =============================================================================
"""Build per-commodity term structure panels (F0..F12) from raw contract data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.futures.contracts import build_roll_calendar, identify_active_contracts

log = logging.getLogger(__name__)


def build_term_structure(
    raw_prices: pd.DataFrame,
    root: str,
    months: str,
    start_year: int,
    end_year: int,
    n_contracts: int = 13,
    roll_offset_bdays: int = -5,
    price_col: str = "close",
) -> pd.DataFrame:
    """Build a term structure panel for one commodity.

    Maps raw contract prices to a panel of F0 (front) through F(n-1) (deferred)
    using a deterministic roll calendar.

    Args:
        raw_prices: Long-format DataFrame with columns [ts_event, symbol, close, ...].
        root: Futures root symbol.
        months: Valid contract month codes.
        start_year: First year for roll calendar.
        end_year: Last year for roll calendar.
        n_contracts: Number of curve points (default 13: F0..F12).
        roll_offset_bdays: Roll timing relative to expiry.
        price_col: Which price column to use (default "close").

    Returns:
        DataFrame with DatetimeIndex (business days) and columns F0, F1, ..., F12.
        Values are unadjusted settlement prices.
    """
    # Build roll calendar
    roll_cal = build_roll_calendar(root, months, start_year, end_year, roll_offset_bdays)

    # Get the date column from raw prices
    ts_col = "ts_event" if "ts_event" in raw_prices.columns else raw_prices.columns[0]

    # Filter to pure outright contracts only (root + month_code + year digits)
    import re
    outright_pattern = re.compile(rf"^{re.escape(root)}[A-Z]\d+$")
    outrights = raw_prices[raw_prices["symbol"].str.match(outright_pattern, na=False)].copy()

    if outrights.empty:
        log.warning(f"No outright contracts found for {root}")
        return pd.DataFrame()

    # Pivot raw prices to wide format: dates × symbols (keep Databento's native symbols)
    prices_wide = outrights.pivot_table(
        index=ts_col, columns="symbol", values=price_col, aggfunc="last"
    )
    prices_wide.index = pd.to_datetime(prices_wide.index).tz_localize(None)
    prices_wide = prices_wide.sort_index()

    # Only use business days where we have data
    dates = prices_wide.index

    # Identify active contracts for each date
    active = identify_active_contracts(roll_cal, dates, n_contracts)
    if active.empty:
        log.warning(f"No active contracts found for {root}")
        return pd.DataFrame()

    # Build mapping from roll calendar symbols (CLG24) to Databento symbols (CLG4)
    # Databento uses 1-digit years for current decade, 2-digit for far-out contracts
    available_syms = set(prices_wide.columns)
    def _find_databento_symbol(cal_sym: str) -> str | None:
        """Map roll calendar symbol (CLG24) to Databento format (CLG4)."""
        if cal_sym in available_syms:
            return cal_sym
        # Try 1-digit year: CLG24 -> CLG4
        m = re.match(rf"^({re.escape(root)}[A-Z])(\d{{2}})$", cal_sym)
        if m:
            short = f"{m.group(1)}{m.group(2)[1]}"  # drop leading digit
            if short in available_syms:
                return short
        return None

    # Build the term structure panel
    records = []
    for date in dates:
        try:
            date_contracts = active.loc[date]
        except KeyError:
            continue

        row = {"date": date}
        for pos, sym_row in date_contracts.iterrows():
            symbol = sym_row["symbol"]
            db_sym = _find_databento_symbol(symbol)
            col_name = f"F{pos}"
            if db_sym and db_sym in prices_wide.columns:
                price = prices_wide.loc[date, db_sym]
                if pd.notna(price) and price > 0:
                    row[col_name] = price
        records.append(row)

    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame()

    result = result.set_index("date").sort_index()
    result.index.name = "date"

    # Ensure columns are ordered F0..F12
    expected_cols = [f"F{i}" for i in range(n_contracts)]
    for col in expected_cols:
        if col not in result.columns:
            result[col] = np.nan
    result = result[expected_cols]

    log.info(
        f"{root}: term structure {result.shape[0]} days × {result.dropna(axis=1, how='all').shape[1]} contracts"
    )
    return result


def build_ratio_adjusted_series(term_structure: pd.DataFrame, position: int = 0) -> pd.Series:
    """Build a ratio-adjusted continuous price series for a given position.

    Used for trend-following signals to avoid spurious jumps at roll dates.
    The ratio adjustment preserves percentage returns across roll boundaries.

    Args:
        term_structure: Output of build_term_structure (dates × F0..F12).
        position: Which curve position to make continuous (default 0 = front).

    Returns:
        Series of ratio-adjusted prices (latest value matches actual price).
    """
    col = f"F{position}"
    prices = term_structure[col].dropna()

    if prices.empty:
        return pd.Series(dtype=float)

    # Compute daily returns of the position
    returns = prices.pct_change()

    # The adjusted series compounds these returns backward from the last price
    adjusted = (1 + returns).cumprod()
    adjusted = adjusted * (prices.iloc[-1] / adjusted.iloc[-1])

    return adjusted
