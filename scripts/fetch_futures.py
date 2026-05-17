#!/usr/bin/env python3
"""Fetch + assemble the 18 BCOM continuous-futures series into one parquet.

Contract: specs/002-phase-1-momentum/contracts/fetch_futures_cli.md

Pipeline per root:
    lake (raw outright contracts)
      → contracts.build_roll_calendar(months, start_year, end_year, roll_offset_bdays=-5)
      → contracts.identify_active_contracts(roll_cal, dates, n_contracts=13)
      → term_structure.build_term_structure(raw, ..., position=0 implied by next call)
      → term_structure.build_ratio_adjusted_series(term_structure, position=0)
      → long-format row stream with (date, contract=root, asset_class='commodity',
                                     price, return=price.pct_change())

Output: ${DEEP_FINANCE_DATA_DIR}/cme_futures.parquet + sidecar JSON.
Atomic write per FR-004.

Per OD-1: 18 BCOM commodity roots, all `asset_class='commodity'`.
Per OD-2 follow-up: F0 front-month ratio-adjusted (research.md §R1).
Per FR-004a: covers 2010-06-06 → today (lake limit).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path

# Ensure repo root on sys.path so `from src.* import ...` works
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("fetch_futures")

EXIT_OK = 0
EXIT_LAKE_MISSING = 1
EXIT_ROOTS_MISSING = 2
EXIT_FS_ERROR = 3

# Commodity-root → valid month codes mapping (mirrors
# ~/projects/QIS_Commodities/configs/default.yaml). Kept inline here so
# scripts/fetch_futures.py is self-contained — no YAML parsing dep.
ROOT_MONTHS: dict[str, str] = {
    # Energy
    "CL": "FGHJKMNQUVXZ", "CB": "FGHJKMNQUVXZ", "NG": "FGHJKMNQUVXZ",
    "RB": "FGHJKMNQUVXZ", "HO": "FGHJKMNQUVXZ",
    # Grains
    "ZC": "HKNUZ", "ZS": "FHKNQUX", "ZW": "HKNUZ",
    "ZL": "FHKNQUVZ", "ZM": "FHKNQUVZ",
    # Livestock
    "LE": "GJMQVZ", "HE": "GJKMNQVZ",
    # Base metals
    "HG": "HKNUZ", "ALI": "FGHJKMNQUVXZ", "ZNC": "FGHJKMNQUVXZ", "NI": "FGHJKMNQUVXZ",
    # Precious metals
    "GC": "GJMQVZ", "SI": "HKNUZ",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_futures",
        description=(
            "Build a single long-format parquet with the 18 BCOM "
            "F0 ratio-adjusted continuous futures series. Reads from the "
            "developer's local databento lake; writes to "
            "$DEEP_FINANCE_DATA_DIR (default ./data/). See "
            "specs/002-phase-1-momentum/contracts/fetch_futures_cli.md."
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output dir. Default: $DEEP_FINANCE_DATA_DIR (else ./data/).",
    )
    parser.add_argument(
        "--lake-parquet", type=Path, default=None,
        help="Override the source lake parquet path.",
    )
    parser.add_argument(
        "--roots", type=str, default=None,
        help="Comma-separated subset of BCOM roots (e.g. 'CL,ZC,NG'). "
             "Default: all 18.",
    )
    parser.add_argument(
        "--start", type=_dt.date.fromisoformat, default=None,
        help="Truncate output start date (ISO YYYY-MM-DD). Default: 2010-06-06.",
    )
    parser.add_argument(
        "--end", type=_dt.date.fromisoformat, default=None,
        help="Truncate output end date (ISO YYYY-MM-DD). Default: today UTC.",
    )
    parser.add_argument(
        "--roll-offset", type=int, default=-5,
        help="Business-day offset before delivery-month start. Default: -5.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the per-root plan and exit.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="INFO logging (default: WARNING).",
    )
    return parser


def _atomic_write_parquet(df, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp, engine="pyarrow", index=False)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    import pandas as pd

    from src.data import data_root
    from src.data.futures import BCOM_ROOTS
    from src.data.futures import lake_loader
    from src.data.futures.contracts import build_roll_calendar
    from src.data.futures.term_structure import (
        build_term_structure,
        build_ratio_adjusted_series,
    )

    # ── Resolve flags ─────────────────────────────────────────────
    out_dir: Path = args.out_dir.expanduser().resolve() if args.out_dir else data_root()
    start_date: _dt.date = args.start or _dt.date(2010, 6, 6)
    end_date: _dt.date = args.end or _dt.date.today()
    requested_roots = (
        [r.strip().upper() for r in args.roots.split(",") if r.strip()]
        if args.roots else list(BCOM_ROOTS)
    )
    unknown = sorted(set(requested_roots) - set(ROOT_MONTHS))
    if unknown:
        log.error("Unknown roots requested: %s. Known: %s",
                  unknown, sorted(ROOT_MONTHS))
        return EXIT_ROOTS_MISSING

    # Allow --lake-parquet to override the module-level constant
    if args.lake_parquet is not None:
        lake_loader.LAKE_PARQUET = args.lake_parquet.expanduser().resolve()

    if args.dry_run:
        print(f"[fetch_futures] --dry-run; out_dir={out_dir}; lake={lake_loader.LAKE_PARQUET}")
        for r in requested_roots:
            print(f"[fetch_futures]   would fetch {r}: months={ROOT_MONTHS[r]}, "
                  f"window {start_date}→{end_date}, roll_offset={args.roll_offset}")
        return EXIT_OK

    # ── Load the lake parquet (one in-process read, cached) ────────
    try:
        raw = lake_loader._load_lake()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return EXIT_LAKE_MISSING

    # ── Per-root: build term structure → ratio-adjusted F0 series ──
    rows: list[pd.DataFrame] = []
    for root in requested_roots:
        months = ROOT_MONTHS[root]
        # Filter raw to this root's contracts (cheap pre-filter)
        sub = raw[raw["symbol"].str.startswith(root, na=False)].copy()
        if sub.empty:
            log.warning("  %s: 0 rows in lake — skipping", root)
            continue

        ts_panel = build_term_structure(
            raw_prices=sub,
            root=root,
            months=months,
            start_year=start_date.year,
            end_year=end_date.year,
            n_contracts=13,
            roll_offset_bdays=args.roll_offset,
        )
        if ts_panel.empty:
            log.warning("  %s: empty term structure — skipping", root)
            continue

        adj = build_ratio_adjusted_series(ts_panel, position=0)
        if adj.empty:
            log.warning("  %s: empty F0 series — skipping", root)
            continue

        # Truncate to the requested window
        adj.index = pd.to_datetime(adj.index).tz_localize(None)
        adj = adj[(adj.index.date >= start_date) & (adj.index.date <= end_date)]
        if adj.empty:
            log.warning("  %s: 0 rows in window — skipping", root)
            continue

        out_df = pd.DataFrame(
            {
                "date": adj.index.date,
                "contract": root,
                "asset_class": "commodity",
                "price": adj.values.astype("float64"),
                "return": adj.pct_change().values.astype("float64"),
            }
        )
        log.info("  %s: %d rows (%s → %s)", root,
                 len(out_df), out_df["date"].iloc[0], out_df["date"].iloc[-1])
        rows.append(out_df)

    if not rows:
        log.error("No roots produced output rows. Aborting.")
        return EXIT_ROOTS_MISSING

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["contract", "date"], kind="stable").reset_index(drop=True)

    # ── Atomic write ───────────────────────────────────────────────
    target = out_dir / "cme_futures.parquet"
    try:
        _atomic_write_parquet(panel, target)
    except OSError as exc:
        log.error("filesystem error: %s", exc)
        return EXIT_FS_ERROR

    sidecar = {
        "universe": "cme_futures",
        "fetched_at": _dt.datetime.now(_dt.timezone.utc)
                      .isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": str(lake_loader.LAKE_PARQUET),
        "roots": sorted(panel["contract"].unique().tolist()),
        "row_count": int(len(panel)),
        "date_range": {
            "min": panel["date"].min().isoformat(),
            "max": panel["date"].max().isoformat(),
        },
        "roll_offset_bdays": args.roll_offset,
        "position": 0,
    }
    target.with_suffix(target.suffix + ".json").write_text(json.dumps(sidecar, indent=2))

    n_contracts = panel["contract"].nunique()
    print(
        f"[fetch_futures] {n_contracts} BCOM roots: {len(panel)} rows, "
        f"{panel['date'].min()} → {panel['date'].max()} "
        f"(F0 ratio-adjusted, roll_offset={args.roll_offset})"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
