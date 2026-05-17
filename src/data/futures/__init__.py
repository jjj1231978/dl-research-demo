"""Continuous-futures construction package for the Deep Finance Showcase.

Three submodules carry the heavy lifting (all copy-with-attribution from
~/projects/QIS_Commodities per OD-2 — see each file's header for the
copy date and re-sync policy):

- `contracts`        — roll calendar + active-contract identification
- `term_structure`   — F0..F12 panel + ratio-adjusted continuous series
- `lake_loader`      — read the developer's local databento BCOM parquet

This `__init__.py` itself owns project-specific constants that are NOT
in the QIS source (Phase 1 spec scope, not commodity-curve infrastructure):

- `TRAIN_START` / `TRAIN_END` / `TEST_START` / `TEST_END` — the 60/40
  chronological split locked in /clarify OD-3 (see research.md §R3).
- `BCOM_ROOTS` — the 18 BCOM commodity roots covered by the developer's
  databento lake (5 energy + 5 grains + 2 livestock + 4 base metals + 2
  precious metals). Sourced from
  `~/projects/QIS_Commodities/configs/default.yaml`.

FR-004a (train/test split) + FR-001/002/003 (BCOM universe) — see
specs/002-phase-1-momentum/spec.md.
"""
from __future__ import annotations

from datetime import date

TRAIN_START: date = date(2010, 6, 6)
TRAIN_END: date = date(2019, 12, 31)
TEST_START: date = date(2020, 1, 1)
TEST_END: date | None = None  # forward — uses whatever data exists at backtest time

BCOM_ROOTS: list[str] = [
    # Energy (5)
    "CL", "CB", "NG", "RB", "HO",
    # Grains (5)
    "ZC", "ZS", "ZW", "ZL", "ZM",
    # Livestock (2)
    "LE", "HE",
    # Base metals (4)
    "HG", "ALI", "ZNC", "NI",
    # Precious metals (2)
    "GC", "SI",
]
assert len(BCOM_ROOTS) == 18, "BCOM_ROOTS must list 18 commodities (FR-001)"
