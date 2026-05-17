# =============================================================================
# Extracted from ~/projects/QIS_Commodities/src/data/fetch.py on 2026-05-17.
# Per /clarify OD-2: this is a copy-with-attribution; QIS remains the source
# of truth for the lake-loading primitives (`_load_lake`, `LAKE_PARQUET`,
# `_lake_cache`, `_lake_lock`). Re-sync MANUALLY when the QIS source changes;
# record the new date here.
#
# Deliberately omitted from the copy:
#   - `fetch_futures_data(...)` — QIS-specific wrapper; our
#     `scripts/fetch_futures.py` owns this role with Phase-1-specific
#     column normalization.
#   - `fetch_all_commodities(...)` — same reason.
# =============================================================================
"""Read futures OHLCV data from the local Databento data lake.

The data lake at ~/data_lake/databento is populated by
~/data_lake/databento/scripts/download_databento.py and stores all 18 BCOM
commodity roots in a single combined parquet. This module exposes only the
load primitive; higher-level fetching logic lives in
`scripts/fetch_futures.py`.
"""

import logging
from pathlib import Path
from threading import Lock

import pandas as pd

log = logging.getLogger(__name__)

LAKE_PARQUET = (
    Path.home()
    / "data_lake/databento/futures/L0/ohlcv-1d"
    / "ohlcv-1d_GLBX.MDP3_bcom_2010-06-06_2026-05-05.parquet"
)
ALLOWED_DATASETS = {"GLBX.MDP3"}
MONTH_CODES = "FGHJKMNQUVXZ"

_lake_cache: pd.DataFrame | None = None
_lake_lock = Lock()


def _load_lake() -> pd.DataFrame:
    """Lazily load and cache the lake parquet (one in-process load)."""
    global _lake_cache
    with _lake_lock:
        if _lake_cache is None:
            if not LAKE_PARQUET.exists():
                raise FileNotFoundError(
                    f"Data lake parquet not found at {LAKE_PARQUET}. "
                    f"Populate it via "
                    f"`python ~/data_lake/databento/scripts/download_databento.py`."
                )
            df = pd.read_parquet(LAKE_PARQUET).reset_index()
            ts_col = "ts_event" if "ts_event" in df.columns else df.columns[0]
            if ts_col != "ts_event":
                df = df.rename(columns={ts_col: "ts_event"})
            df["ts_event"] = pd.to_datetime(df["ts_event"]).dt.normalize()
            _lake_cache = df
            log.info(f"Loaded {len(df):,} rows from lake: {LAKE_PARQUET.name}")
    return _lake_cache
