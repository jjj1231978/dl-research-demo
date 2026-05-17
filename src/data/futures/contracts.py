# =============================================================================
# Copied verbatim from ~/projects/QIS_Commodities/src/data/contracts.py
# on 2026-05-17. Per /clarify OD-2: this is a copy-with-attribution; QIS
# remains the source of truth. Re-sync MANUALLY when the QIS source
# changes (no automated import); record the new date here.
#
# Phase 1 (Deep Finance Showcase): used by scripts/fetch_futures.py and
# the Phase 1 momentum-page back end. Not modified post-copy.
# =============================================================================
"""Contract roll calendar and expiry logic for commodity futures."""

from __future__ import annotations

import pandas as pd
import numpy as np

# Standard futures month codes
MONTH_CODES = "FGHJKMNQUVXZ"
MONTH_CODE_TO_NUM = {code: i + 1 for i, code in enumerate(MONTH_CODES)}
MONTH_NUM_TO_CODE = {v: k for k, v in MONTH_CODE_TO_NUM.items()}


def contract_symbol(root: str, year: int, month: int) -> str:
    """Build contract symbol, e.g. 'CLZ24' for Dec 2024 crude."""
    code = MONTH_NUM_TO_CODE[month]
    yr = year % 100
    return f"{root}{code}{yr:02d}"


def enumerate_contracts(
    root: str,
    months: str,
    start_year: int,
    end_year: int,
) -> list[dict]:
    """List all contract symbols for a root within a year range.

    Args:
        root: Futures root symbol (e.g. "CL").
        months: Valid contract month codes (e.g. "FGHJKMNQUVXZ" or "HKNUZ").
        start_year: First year (inclusive).
        end_year: Last year (inclusive).

    Returns:
        List of dicts with keys: symbol, root, year, month, month_code, expiry_month.
    """
    contracts = []
    for year in range(start_year, end_year + 1):
        for code in months:
            month = MONTH_CODE_TO_NUM[code]
            contracts.append(
                {
                    "symbol": contract_symbol(root, year, month),
                    "root": root,
                    "year": year,
                    "month": month,
                    "month_code": code,
                    # Approximate expiry: last business day of the month prior to delivery
                    # This is a simplification; real expiry varies by commodity
                    "expiry_approx": pd.Timestamp(year, month, 1)
                    - pd.offsets.BDay(1),
                }
            )
    return contracts


def build_roll_calendar(
    root: str,
    months: str,
    start_year: int,
    end_year: int,
    roll_offset_bdays: int = -5,
) -> pd.DataFrame:
    """Build a deterministic roll calendar for a commodity.

    The roll date is `roll_offset_bdays` business days before the approximate
    expiry (first of the delivery month). After the roll date, the next
    contract becomes the front month.

    Args:
        root: Futures root symbol.
        months: Valid contract month codes.
        start_year: First year.
        end_year: Last year.
        roll_offset_bdays: Business days before expiry to roll (negative = before).

    Returns:
        DataFrame with columns: symbol, year, month, expiry_approx, roll_date
        Sorted by roll_date ascending.
    """
    contracts = enumerate_contracts(root, months, start_year, end_year)
    df = pd.DataFrame(contracts)

    # Roll date: N business days before the start of the delivery month
    # (i.e., before the first of the contract month)
    first_of_month = df.apply(
        lambda row: pd.Timestamp(row["year"], row["month"], 1), axis=1
    )
    df["roll_date"] = first_of_month + pd.offsets.BDay(roll_offset_bdays)
    df = df.sort_values("roll_date").reset_index(drop=True)

    return df[["symbol", "root", "year", "month", "month_code", "expiry_approx", "roll_date"]]


def identify_active_contracts(
    roll_calendar: pd.DataFrame,
    dates: pd.DatetimeIndex,
    n_contracts: int = 13,
) -> pd.DataFrame:
    """For each date, identify which contracts are F0, F1, ..., F(n-1).

    Logic: On any given date, F0 is the nearest contract whose roll_date
    has not yet passed (i.e., roll_date > date). F1 is the next one, etc.

    Args:
        roll_calendar: Output of build_roll_calendar.
        dates: Business dates to map.
        n_contracts: Number of contracts along the curve (default 13 for F0..F12).

    Returns:
        DataFrame with MultiIndex (date, position) and column 'symbol'.
        position = 0 means F0 (front), 1 = F1, etc.
    """
    roll_dates = roll_calendar["roll_date"].values
    symbols = roll_calendar["symbol"].values

    records = []
    for date in dates:
        # Find contracts not yet rolled (roll_date > current date)
        mask = roll_dates > date
        future_indices = np.where(mask)[0]

        for pos in range(min(n_contracts, len(future_indices))):
            idx = future_indices[pos]
            records.append(
                {"date": date, "position": pos, "symbol": symbols[idx]}
            )

    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(columns=["date", "position", "symbol"])
    return result.set_index(["date", "position"])
