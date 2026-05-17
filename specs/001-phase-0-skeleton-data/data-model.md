# Data Model: Phase 0 — Skeleton & Data Foundation

**Branch**: `001-phase-0-skeleton-data`
**Date**: 2026-05-16
**Inputs**: `spec.md` (Key Entities section), `Project_brief.md` §§4, 13.7

Phase 0 introduces three entities. None of them are database tables — by
Constitution Principle IV (Data-as-Artifact), persistent state lives in files
on disk. The entities here describe the **logical** shape of those files and
the in-memory dictionaries / dataclasses the code consumes.

---

## E1 — `Universe`

A named collection of tradeable instruments along with the data source and
which page(s) consume it. Defined statically in `src/universes.py` as a
module-level dictionary mapping `name` → `Universe`. Construction happens
at import time; the dictionary is read-only from the rest of the codebase.

**Fields**:

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `str` | ✅ | Lowercase snake_case key. Used as the parquet filename stem and as the dictionary key. Values: `etf_basket`, `sp500_20`, `sp500_100`, `cme_futures`, `aapl_toy`. |
| `label` | `str` | ✅ | Human-readable label for the UI (e.g., "ETF Basket (VTI / AGG / DBC / VIXY)"). |
| `members` | `list[str] \| None` | ✅ | Static member identifiers where known (`["VTI", "AGG", "DBC", "VIXY"]` for the ETF basket; the 20 tickers for `sp500_20`). `None` for `cme_futures` (members come from the data lake at fetch time) and conditionally `None` for `sp500_100` until OD-2 confirmation lands. |
| `source` | `Literal["fmp", "cme_data_lake", "kaggle", "local_csv"]` | ✅ | Provenance tag. Drives which fetch script materializes the universe. |
| `consumers` | `list[Literal["momentum", "portfolio", "order_book", "landing"]]` | ✅ | Which page(s) read this universe. Used by the data-status sidebar to scope the per-universe rows. |
| `parquet_path` | `pathlib.Path \| None` | derived (`@property`) | Computed lazily as `data_root() / f"{name}.parquet"` per FR-022 — re-resolves on every access so changing `DEEP_FINANCE_DATA_DIR` mid-process is visible. `None` for `aapl_toy` (uses CSV directly). |
| `csv_fallback_path` | `pathlib.Path \| None` | derived (`@property`) | Resolves to `BUNDLED_CSV_DIR / "aapl.csv"` for `aapl_toy`, `BUNDLED_CSV_DIR / "portfolio_data.csv"` for `sp500_20`, `None` otherwise. `BUNDLED_CSV_DIR` is a fixed repo-relative path and does NOT honour `DEEP_FINANCE_DATA_DIR` (FR-022). |

**Validation rules**:

- `members` MUST equal the column header set of `csv_fallback_path` when both
  are non-`None` (FR-006, plus the consistency check captured as an edge case).
- `parquet_path` and `csv_fallback_path` MUST NOT both be `None` — every
  universe needs at least one materialisation path.
- `name` MUST match `^[a-z][a-z0-9_]*$` (lowercase identifier).

**State transitions**: none (definitions are immutable at import time).

---

## E2 — `DataSnapshot`

A point-in-time materialisation of a universe to disk. Constructed by the data
loader (`src/data.py`) on each `get_data_snapshot(universe)` call. Returned
to the landing page (and future per-paper pages) for the data-status sidebar.
Not persisted as a separate file — derived freshly from the on-disk parquet
or CSV.

**Fields**:

| Field | Type | Required | Notes |
|---|---|---|---|
| `universe` | `Universe` | ✅ | Reference to the universe definition. |
| `path` | `pathlib.Path` | ✅ | The actual file used (parquet preferred, CSV fallback). |
| `source_kind` | `Literal["parquet", "csv", "missing"]` | ✅ | `"missing"` covers the case where both parquet and CSV are absent — relevant for `sp500_100` before OD-2 confirmation. |
| `refresh_ts` | `datetime.datetime \| None` | ✅ | File mtime in UTC. `None` when `source_kind == "missing"`. |
| `row_count` | `int \| None` | ✅ | Loaded row count. `None` when `source_kind == "missing"`. Computed lazily on first access. |
| `date_range` | `tuple[date, date] \| None` | ✅ | `(min_date, max_date)` of the universe's date column. `None` when `source_kind == "missing"`. |

**Validation rules**:

- `source_kind == "parquet"` ⇒ `path.suffix == ".parquet"` and
  `path == universe.parquet_path`.
- `source_kind == "csv"` ⇒ `path == universe.csv_fallback_path`.
- `source_kind == "missing"` ⇒ all of `refresh_ts`, `row_count`,
  `date_range` are `None`.

**State transitions**: none (immutable value object).

---

## E3 — `MetricsReport`

The dictionary returned by `src/metrics.py:report_metrics(ret)`. Not a class
— a plain `dict[str, float]` for backward compatibility with the user's
canonical signature.

**Keys and formulas** (per brief §13.7):

| Key | Provenance | Formula |
|---|---|---|
| `annual_ret` | **Original (immutable)** | `np.mean(ret) * 252` |
| `annual_std` | **Original (immutable)** | `np.std(ret) * np.sqrt(252)` |
| `annual_sharpe` | **Original (immutable)** | `(np.mean(ret) / np.std(ret)) * np.sqrt(252)` |
| `downside_dev` | NEW (Phase 0) | `np.std(ret[ret < 0]) * np.sqrt(252)` |
| `sortino` | NEW (Phase 0) | `(np.mean(ret) / np.std(ret[ret < 0])) * np.sqrt(252)` |
| `max_drawdown` | NEW (Phase 0) | `abs(((1+ret).cumprod() - np.maximum.accumulate((1+ret).cumprod())) / np.maximum.accumulate((1+ret).cumprod())).min())` (using the equity-curve derivation in brief §13.7) |
| `calmar` | NEW (Phase 0) | `annual_ret / max_drawdown` |
| `pct_positive` | NEW (Phase 0) | `(ret > 0).mean()` |
| `avg_p_over_l` | NEW (Phase 0) | `ret[ret > 0].mean() / abs(ret[ret < 0].mean())` |

**Validation rules**:

- The three original keys (`annual_ret`, `annual_std`, `annual_sharpe`)
  MUST return byte-identical values for any fixed-seed returns series before
  and after the extension (Principle V; FR-004; regression test under US3).
- All values are finite Python `float`s. NaN/Inf for a metric is a test
  failure — the synthetic fixture is designed to avoid division-by-zero in
  any of the formulas.
- The function signature `report_metrics(ret: np.ndarray) -> dict[str, float]`
  MUST NOT change (Principle V).

**State transitions**: none (pure function output).
