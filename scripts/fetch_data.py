#!/usr/bin/env python3
"""One-shot FMP fetch script — produces three named parquets.

Contract: specs/001-phase-0-skeleton-data/contracts/fetch_data_cli.md

Architecture mirrors ~/projects/ML_short_reversion/src/data/__main__.py:
- FMP work is delegated to src.fmp (per-ticker parquet cache at
  ~/data_lake/fmp/deep-finance/, /stable/ namespace, urllib + rate limit).
- This script only assembles per-universe OUTPUT parquets from those caches
  and writes them to $DEEP_FINANCE_DATA_DIR (default ./data/).

Per OD-2 the sp500_100 universe requires a confirmation flow
(proposed.json → confirmed.json). Per OD-3 a missing shares-outstanding
endpoint is warn-and-skip (exit 0).

Run with --help for the full CLI surface.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src.* import ...` works when
# invoked as `python scripts/fetch_data.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("fetch_data")

# Exit codes per fetch_data_cli.md §"Exit codes"
EXIT_OK = 0
EXIT_NO_KEY = 1
EXIT_AUTH_REJECTED = 2
EXIT_RATE_LIMIT = 3
EXIT_FS_ERROR = 4
EXIT_OD2_PENDING = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_data",
        description=(
            "Fetch ETF, 20-stock, and 100-stock parquets from FMP. "
            "Per-ticker FMP cache at ~/data_lake/fmp/deep-finance/ (mirrors "
            "ML_short_reversion convention). Output parquets go to "
            "$DEEP_FINANCE_DATA_DIR (default ./data/). See "
            "specs/001-phase-0-skeleton-data/contracts/fetch_data_cli.md."
        ),
    )
    parser.add_argument(
        "--universe",
        action="append",
        choices=["etf_basket", "sp500_20", "sp500_100"],
        help="Restrict to specific universes (repeatable). Default: all three.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir for assembled parquets. Default: DEEP_FINANCE_DATA_DIR (else ./data/).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="FMP per-ticker cache dir. Default: ~/data_lake/fmp/deep-finance/.",
    )
    parser.add_argument(
        "--start",
        type=_dt.date.fromisoformat,
        default=None,
        help="Earliest date to fetch (ISO YYYY-MM-DD). Universe default if omitted.",
    )
    parser.add_argument(
        "--end",
        type=_dt.date.fromisoformat,
        default=None,
        help="Latest date to fetch (ISO YYYY-MM-DD). Default: today (UTC).",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=240,
        help="FMP requests-per-minute cap. Default 240 (Starter tier cap is 250).",
    )
    parser.add_argument(
        "--refresh-days",
        type=int,
        default=7,
        help="Trailing window re-fetched on incremental update (default 7).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-universe plan and exit without making FMP calls.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="INFO logging (default: WARNING).",
    )
    return parser


_DEFAULT_END = _dt.date.today()
_UNIVERSE_DEFAULT_START = {
    "etf_basket": _dt.date(2011, 1, 1),    # VIXY inception
    "sp500_20":   _dt.date(1992, 11, 25),  # matches existing portfolio_data.csv
    "sp500_100":  _dt.date(1995, 1, 3),    # historical-constituent coverage start
}


# ---------------------------------------------------------------------------
# Atomic write + sidecar
# ---------------------------------------------------------------------------


def _atomic_write_parquet(df, path: Path) -> None:
    """Write parquet to a tempfile then rename — readers never see a partial."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp, engine="pyarrow", index=False)
    os.replace(tmp, path)


def _write_sidecar(parquet_path: Path, *, universe: str, df,
                   shares_outstanding_available: bool, endpoints: list[str]) -> None:
    import pandas as pd
    date_col = "date" if "date" in df.columns else df.columns[0]
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    sidecar = {
        "universe": universe,
        "fetched_at": _dt.datetime.now(_dt.timezone.utc)
                      .isoformat(timespec="seconds").replace("+00:00", "Z"),
        "fmp_endpoint_set": endpoints,
        "shares_outstanding_available": shares_outstanding_available,
        "row_count": int(len(df)),
        "symbols": sorted(df["symbol"].unique().tolist()) if "symbol" in df.columns else [],
        "date_range": {
            "min": dates.min().date().isoformat() if not dates.empty else None,
            "max": dates.max().date().isoformat() if not dates.empty else None,
        },
    }
    sidecar_path = parquet_path.with_suffix(parquet_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))


def _summary_line(universe: str, df, *, shares_outstanding_available: bool) -> str:
    import pandas as pd
    n_symbols = df["symbol"].nunique() if "symbol" in df.columns else 0
    dates = pd.to_datetime(df.get("date", pd.Series([], dtype="datetime64[ns]")),
                            errors="coerce").dropna()
    date_range = f"{dates.min().date()} → {dates.max().date()}" if not dates.empty else "n/a"
    so = "yes" if shares_outstanding_available else "no"
    return (f"[fetch_data] {universe}: {len(df)} rows, {n_symbols} symbols, "
            f"{date_range} (shares_outstanding: {so})")


# ---------------------------------------------------------------------------
# Per-universe assemblers
# ---------------------------------------------------------------------------


def assemble_universe(
    universe_name: str,
    symbols: list[str],
    start: _dt.date,
    end: _dt.date,
    out_dir: Path,
    cache_dir: Path | None,
    rate_limit: int,
    refresh_days: int,
) -> str:
    """Fetch + cache + assemble one universe's parquet.

    Returns the human-readable summary line.
    """
    from src.fmp import fetch_historical_prices, fetch_shares_outstanding

    prices = fetch_historical_prices(
        symbols, start, end,
        cache_dir=cache_dir, rate_limit_per_min=rate_limit, refresh_days=refresh_days,
    )

    # OD-3: try shares_outstanding; warn-and-skip if endpoint unavailable
    shares = fetch_shares_outstanding(
        symbols, start, end,
        cache_dir=cache_dir, rate_limit_per_min=rate_limit,
    )
    has_shares = shares is not None and not shares.empty
    if has_shares:
        prices = prices.merge(shares, on=["date", "symbol"], how="left")

    target = out_dir / f"{universe_name}.parquet"
    _atomic_write_parquet(prices, target)
    endpoints = ["historical-price-eod/full"]
    if shares is not None:
        endpoints.append("shares-float")
    _write_sidecar(target, universe=universe_name, df=prices,
                   shares_outstanding_available=has_shares, endpoints=endpoints)
    return _summary_line(universe_name, prices, shares_outstanding_available=has_shares)


def propose_sp500_100(out_dir: Path, cache_dir: Path | None) -> None:
    """OD-2 proposal generator. Pulls current S&P 500 with sectors + historical
    constituents, picks 10 names per GICS sector preferring longest inclusion.
    Writes sp500_100.proposed.json next to the parquets for developer review.
    """
    import pandas as pd
    from src.fmp import (
        fetch_current_sp500_constituents,
        fetch_historical_sp500_constituents,
    )

    current = fetch_current_sp500_constituents(cache_dir=cache_dir)
    hist = fetch_historical_sp500_constituents(cache_dir=cache_dir)

    if current.empty:
        log.warning("FMP returned no current S&P 500 constituents — cannot propose")
        return

    sym_col = "symbol" if "symbol" in current.columns else "ticker"
    sector_col = "sector" if "sector" in current.columns else None

    if sector_col is None:
        log.warning("FMP current constituents missing 'sector' column — bucketing as Unknown")
        current["sector"] = "Unknown"
        sector_col = "sector"

    # Compute longest-inclusion priority from historical events (best-effort)
    inclusion_priority: dict[str, str] = {}
    if not hist.empty and "dateAdded" in hist.columns:
        hist_sym_col = "addedTicker" if "addedTicker" in hist.columns else (
            "symbol" if "symbol" in hist.columns else None
        )
        if hist_sym_col:
            for _, row in hist.sort_values("dateAdded").iterrows():
                t = row.get(hist_sym_col)
                if t and t not in inclusion_priority:
                    inclusion_priority[t] = str(row.get("dateAdded", ""))

    proposal: dict[str, list[str]] = {}
    for sector, group in current.groupby(sector_col, dropna=False):
        tickers = group[sym_col].dropna().tolist()
        # Sort by earliest-known inclusion (longest in index first)
        tickers.sort(key=lambda t: inclusion_priority.get(t, "9999-12-31"))
        proposal[str(sector)] = tickers[:10]

    target = out_dir / "sp500_100.proposed.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "instructions": (
            "Review this list. To accept, rename to sp500_100.confirmed.json. "
            "To edit, modify in place then rename. The fetch script then uses "
            "confirmed.json to produce sp500_100.parquet."
        ),
        "method": (
            "Grouped current S&P 500 constituents by GICS sector; picked up to "
            "10 per sector preferring longest historical inclusion (per FR-011 — "
            "historical constituents avoid survivorship bias when consumed by "
            "Phase 2 backtests)."
        ),
        "generated_at": _dt.datetime.now(_dt.timezone.utc)
                        .isoformat(timespec="seconds").replace("+00:00", "Z"),
        "sectors": {k: {"tickers": v, "count": len(v)} for k, v in proposal.items()},
        "total_tickers": sum(len(v) for v in proposal.values()),
    }, indent=2))
    log.warning(
        "OD-2: 100-stock proposal written to %s — review and rename to "
        "sp500_100.confirmed.json before re-running fetch", target,
    )


def fetch_sp500_100(out_dir: Path, cache_dir: Path | None, start: _dt.date, end: _dt.date,
                    rate_limit: int, refresh_days: int) -> str | None:
    """OD-2 flow: if confirmed.json exists, fetch + assemble. Else propose and return None."""
    confirmed = out_dir / "sp500_100.confirmed.json"
    if not confirmed.exists():
        proposed = out_dir / "sp500_100.proposed.json"
        if not proposed.exists():
            propose_sp500_100(out_dir, cache_dir)
        return None

    data = json.loads(confirmed.read_text())
    members: list[str] = []
    if "sectors" in data:
        for s in data["sectors"].values():
            members.extend(s.get("tickers", []))
    elif "tickers" in data:
        members = data["tickers"]
    if not members:
        raise RuntimeError(f"sp500_100.confirmed.json: empty ticker list in {confirmed}")

    log.info("sp500_100: %d confirmed tickers", len(members))
    return assemble_universe(
        "sp500_100", members, start, end, out_dir, cache_dir, rate_limit, refresh_days,
    )


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.data import data_root
    out_dir: Path = args.out_dir.expanduser().resolve() if args.out_dir else data_root()
    cache_dir: Path | None = args.cache_dir.expanduser().resolve() if args.cache_dir else None
    end: _dt.date = args.end or _DEFAULT_END

    universes: list[str] = args.universe or ["etf_basket", "sp500_20", "sp500_100"]
    explicit = bool(args.universe)

    # --dry-run: print plan and exit (no FMP key required)
    if args.dry_run:
        from src.fmp import DEFAULT_FMP_CACHE
        eff_cache = cache_dir or DEFAULT_FMP_CACHE
        print(f"[fetch_data] --dry-run; out_dir={out_dir}; cache_dir={eff_cache}")
        for u in universes:
            start = args.start or _UNIVERSE_DEFAULT_START[u]
            print(f"[fetch_data]   would fetch {u}: {start} → {end} (rate={args.rate_limit} rpm)")
        return EXIT_OK

    # Validate FMP_API_KEY before doing any work
    from src.fmp import _resolve_api_key  # noqa: PLC2701 (intentional internal use)
    try:
        _resolve_api_key(None)
    except EnvironmentError as exc:
        log.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_NO_KEY

    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[str] = []

    from src.universes import UNIVERSES
    try:
        for u in universes:
            start = args.start or _UNIVERSE_DEFAULT_START[u]
            if u == "etf_basket":
                members = UNIVERSES["etf_basket"].members
                summaries.append(assemble_universe(
                    "etf_basket", members, start, end, out_dir, cache_dir,
                    args.rate_limit, args.refresh_days,
                ))
            elif u == "sp500_20":
                members = UNIVERSES["sp500_20"].members
                summaries.append(assemble_universe(
                    "sp500_20", members, start, end, out_dir, cache_dir,
                    args.rate_limit, args.refresh_days,
                ))
            elif u == "sp500_100":
                result = fetch_sp500_100(
                    out_dir, cache_dir, start, end, args.rate_limit, args.refresh_days,
                )
                if result is None:
                    summaries.append(
                        "[fetch_data] sp500_100:  SKIPPED (OD-2 confirmation pending — "
                        f"review {out_dir / 'sp500_100.proposed.json'} and rename to "
                        "sp500_100.confirmed.json)"
                    )
                    if explicit:
                        for s in summaries:
                            print(s)
                        return EXIT_OD2_PENDING
                else:
                    summaries.append(result)
    except EnvironmentError as exc:
        log.error("auth error: %s", exc)
        return EXIT_AUTH_REJECTED
    except OSError as exc:
        log.error("filesystem error: %s", exc)
        return EXIT_FS_ERROR
    except RuntimeError as exc:
        # e.g. partial-cache refusal from fetch_historical_prices
        log.error("%s", exc)
        return EXIT_RATE_LIMIT

    for s in summaries:
        print(s)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
