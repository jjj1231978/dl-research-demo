"""FMP (Financial Modeling Prep) data fetchers.

Mirrors the architecture of ~/projects/ML_short_reversion/src/data/fmp.py:

- All endpoints use the `/stable/` namespace; the legacy `/api/v3/*` endpoints
  return 403 on Premium plans.
- Per-ticker parquet caches at `~/data_lake/fmp/deep-finance/` with
  non-destructive merge-and-write (atomic, append-only).
- Trailing N-day re-fetch on each incremental update to pick up FMP back-
  revisions (split adjustments, late corrections) without rewriting history.
- `urllib.request` for HTTP (smaller dep footprint, no `requests` needed).
- Configurable rate limit (default 240 rpm — FMP Starter cap is 250 rpm).

Public entry points consumed by scripts/fetch_data.py:
    fetch_historical_prices(symbols, start, end, ...)         → long-format DataFrame
    fetch_current_sp500_constituents(...)                      → current with sector
    fetch_historical_sp500_constituents(...)                   → add/remove events
    fetch_shares_outstanding(symbols, start, end, ...)         → None on 404 (OD-3)
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load .env at import so FMP_API_KEY is available regardless of entry point.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger(__name__)

# Cache convention (mirrors reference) — co-located with the developer's
# existing data lake. Distinct from `DEEP_FINANCE_DATA_DIR` (which controls
# the OUTPUT parquet location, not the cache).
DEFAULT_FMP_CACHE = Path("~/data_lake/fmp/deep-finance").expanduser()

FMP_BASE = "https://financialmodelingprep.com/stable"

# Trailing window re-fetched on each incremental update. Captures back-revisions
# (split adjustments, late corrections) without rewriting historical rows.
DEFAULT_REFRESH_DAYS = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise EnvironmentError("FMP_API_KEY missing. Add it to .env at the repo root.")
    return key


def _fmp_url(path: str, api_key: str, **params) -> str:
    """Build an FMP stable-API URL with apikey appended."""
    qs = urllib.parse.urlencode(
        {**{k: v for k, v in params.items() if v is not None}, "apikey": api_key}
    )
    return f"{FMP_BASE}/{path.lstrip('/')}?{qs}"


def _fetch_with_429_retry(url: str, timeout: int = 30, max_retries: int = 1) -> list | dict | None:
    """HTTP GET with one retry on 429. Returns None on 404/403 (so callers can
    treat unavailable endpoints as warn-and-skip per OD-3).

    FMP returns no `Retry-After` header; sleep 60s on 429.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if e.code == 429 and attempt < max_retries:
                log.warning("HTTP 429 from FMP — sleeping 60s, then retrying once")
                time.sleep(60)
                last_error = e
                continue
            raise
        except Exception as e:
            last_error = e
            raise
    if last_error:
        raise last_error
    raise RuntimeError("unreachable")


def _safe_filename(symbol: str) -> str:
    """FMP returns symbols like 'BRK.B' / '^GSPC' — sanitize for use as a path."""
    return symbol.replace("/", "_").replace("^", "idx_")


def _merge_and_write(path: Path, new_df: pd.DataFrame, key_cols: list[str]) -> int:
    """Append-only merge: existing rows win on key collision. Atomic write.

    Reads `path` if it exists, identifies rows in `new_df` whose key tuple is
    NOT already present, concatenates, then atomically replaces `path`.

    Returns the number of NEW rows added (0 if nothing new).
    """
    if not new_df.empty:
        new_df = new_df.drop_duplicates(subset=key_cols, keep="first")

    if path.exists():
        existing = pd.read_parquet(path)
        if not new_df.empty:
            keys_existing = set(map(tuple, existing[key_cols].itertuples(index=False, name=None)))
            keys_new = list(map(tuple, new_df[key_cols].itertuples(index=False, name=None)))
            mask_new = [k not in keys_existing for k in keys_new]
            new_df = new_df[mask_new]
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df

    added = len(merged) - (len(existing) if path.exists() else 0)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return added


# ---------------------------------------------------------------------------
# Prices — per-ticker cache, multi-symbol assembler
# ---------------------------------------------------------------------------


def _fetch_one_price(symbol: str, start: date, end: date, api_key: str) -> pd.DataFrame | None:
    """Single-ticker price fetch. Returns long-format DataFrame or None."""
    url = _fmp_url(
        "historical-price-eod/full",
        api_key,
        symbol=symbol,
        **{"from": start.isoformat(), "to": end.isoformat()},
    )
    data = _fetch_with_429_retry(url)
    if data is None:
        return None
    # Stable endpoint returns either {historical: [...]} or [...] depending on
    # quota tier. Normalize.
    rows = data.get("historical", data) if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        return None
    df = pd.DataFrame(rows)
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.date
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    return df[[c for c in keep if c in df.columns]].sort_values("date").reset_index(drop=True)


def fetch_historical_prices(
    symbols: list[str],
    start: date,
    end: date,
    *,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    rate_limit_per_min: int = 240,
    refresh_days: int = DEFAULT_REFRESH_DAYS,
) -> pd.DataFrame:
    """Multi-symbol historical price fetch with per-ticker parquet cache.

    For each symbol:
    - If `~/data_lake/fmp/deep-finance/prices/by_symbol/{TICKER}.parquet`
      exists, re-fetch only the last `refresh_days` days (to pick up back-
      revisions) plus any days after the cache's max date.
    - Merge-and-write atomically.

    Returns the long-format concatenation of all per-symbol caches, filtered
    to [start, end].
    """
    api_key = _resolve_api_key(api_key)
    cache_dir = (cache_dir or DEFAULT_FMP_CACHE) / "prices" / "by_symbol"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sleep = 60.0 / max(rate_limit_per_min, 1)
    fetched = skipped_uptodate = failed = 0

    for i, sym in enumerate(symbols):
        cache_path = cache_dir / f"{_safe_filename(sym)}.parquet"
        # Determine the fetch window: incremental if cache exists
        if cache_path.exists():
            existing = pd.read_parquet(cache_path)
            cache_max = pd.to_datetime(existing["date"]).max().date() if not existing.empty else start
            fetch_from = max(start, cache_max - timedelta(days=refresh_days))
            if fetch_from > end:
                skipped_uptodate += 1
                continue
        else:
            fetch_from = start

        try:
            log.info("fetching %s prices %s → %s", sym, fetch_from, end)
            new_df = _fetch_one_price(sym, fetch_from, end, api_key)
            if new_df is not None and not new_df.empty:
                added = _merge_and_write(cache_path, new_df, key_cols=["date", "symbol"])
                fetched += 1
                log.info("  %s: %d new rows", sym, added)
            else:
                log.warning("%s: FMP returned no data", sym)
        except Exception as exc:  # noqa: BLE001
            log.warning("Price fetch failed for %s (%s: %s); skipping", sym, type(exc).__name__, exc)
            failed += 1

        if (i + 1) % 50 == 0:
            log.info("  prices %d/%d (%d updated, %d cached, %d failed)",
                     i + 1, len(symbols), fetched, skipped_uptodate, failed)
        time.sleep(sleep)

    log.info("Prices done: %d updated, %d cached, %d failed",
             fetched, skipped_uptodate, failed)

    # Refuse to assemble a partial cache if too many fetches failed
    if len(symbols) >= 20 and failed / len(symbols) > 0.05:
        raise RuntimeError(
            f"Price fetch failure rate {failed}/{len(symbols)} > 5%. "
            "Refusing to build universe parquet from a partial cache. "
            "Investigate before retrying."
        )

    # Assemble: read every per-ticker cache and concat
    frames = []
    for sym in symbols:
        path = cache_dir / f"{_safe_filename(sym)}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"]).dt.date
    long = long[(long["date"] >= start) & (long["date"] <= end)]
    return long.sort_values(["symbol", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# S&P 500 constituents
# ---------------------------------------------------------------------------


def fetch_current_sp500_constituents(
    *, cache_dir: Path | None = None, api_key: str | None = None,
) -> pd.DataFrame:
    """Current S&P 500 membership with sector tags. Cached as a single parquet."""
    api_key = _resolve_api_key(api_key)
    cache_dir = (cache_dir or DEFAULT_FMP_CACHE) / "membership"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "sp500_current.parquet"

    log.info("fetching current S&P 500 constituents …")
    data = _fetch_with_429_retry(_fmp_url("sp500-constituent", api_key))
    if not isinstance(data, list):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df.to_parquet(cache_path, index=False)
    return df


def fetch_historical_sp500_constituents(
    *, cache_dir: Path | None = None, api_key: str | None = None,
) -> pd.DataFrame:
    """Historical S&P 500 add/remove events. Cached as a single parquet."""
    api_key = _resolve_api_key(api_key)
    cache_dir = (cache_dir or DEFAULT_FMP_CACHE) / "membership"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "sp500_historical.parquet"

    log.info("fetching historical S&P 500 add/remove events …")
    data = _fetch_with_429_retry(_fmp_url("historical-sp500-constituent", api_key))
    if not isinstance(data, list):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df.to_parquet(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Shares outstanding (OD-3 warn-and-skip)
# ---------------------------------------------------------------------------


def fetch_shares_outstanding(
    symbols: list[str],
    start: date,
    end: date,
    *,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    rate_limit_per_min: int = 240,
) -> pd.DataFrame | None:
    """Pull shares-outstanding history for a list of symbols.

    Returns None if the FMP tier does not expose the endpoint (OD-3 / FR-012
    warn-and-skip path) — caller should log a clear message and omit the
    shares_outstanding column from the universe parquet.
    """
    api_key = _resolve_api_key(api_key)
    cache_dir = (cache_dir or DEFAULT_FMP_CACHE) / "shares" / "by_symbol"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sleep = 60.0 / max(rate_limit_per_min, 1)
    frames = []
    any_data = False
    any_unavailable = False
    for sym in symbols:
        url = _fmp_url("shares-float", api_key, symbol=sym)
        data = _fetch_with_429_retry(url)
        if data is None:
            any_unavailable = True
            log.warning("shares-outstanding unavailable for %s on this FMP tier", sym)
            time.sleep(sleep)
            continue
        if not isinstance(data, list) or not data:
            time.sleep(sleep)
            continue
        any_data = True
        df = pd.DataFrame(data)
        df["symbol"] = sym
        # Normalize column names — FMP returns various keys depending on endpoint
        if "freeFloat" in df.columns and "outstandingShares" in df.columns:
            df = df.rename(columns={"outstandingShares": "shares_outstanding"})
        elif "floatShares" in df.columns:
            df = df.rename(columns={"floatShares": "shares_outstanding"})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df[(df["date"] >= start) & (df["date"] <= end)]
        cache_path = cache_dir / f"{_safe_filename(sym)}.parquet"
        keep = [c for c in ("date", "symbol", "shares_outstanding") if c in df.columns]
        if keep:
            df[keep].to_parquet(cache_path, index=False)
            frames.append(df[keep])
        time.sleep(sleep)

    # If we got 4+ symbols back as unavailable and none had data, declare the
    # endpoint unavailable (OD-3 warn-and-skip)
    if any_unavailable and not any_data:
        return None

    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "shares_outstanding"])
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
