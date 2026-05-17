# Contract: `scripts/fetch_data.py`

**Type**: one-shot CLI, developer-local invocation only.
**Runs on**: CPU (Constitution Principle III — no GPU operations).
**Reads from**: FMP REST API (`/stable/` namespace) + `.env` for `FMP_API_KEY`.
**Writes to**:
- OUTPUT parquets (assembled, per-universe): `{data_root()}/etf_basket.parquet`,
  `sp500_20.parquet`, `sp500_100.parquet` (last one after OD-2 confirmation).
- CACHE parquets (per-ticker, append-only): `~/data_lake/fmp/deep-finance/`
  (mirrors the `~/data_lake/<source>/` convention from
  `~/projects/ML_short_reversion/`). Cache survives across runs; re-running
  the script does incremental updates instead of full re-fetch.

**FMP work delegated to `src/fmp.py`** (function-based, mirrors the reference
project's architecture). See `contracts/data_loader_api.md` §"FMP module"
for the FMP module's surface.

## CLI surface

```bash
python scripts/fetch_data.py [OPTIONS]
```

**Options**:

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--universe NAME` | repeatable | (all) | Restrict to specific universes (`etf_basket`, `sp500_20`, `sp500_100`). When omitted, fetches all configured universes. |
| `--out-dir PATH` | path | `data_root()` (which honours `DEEP_FINANCE_DATA_DIR`, defaulting to `./data/`) | OUTPUT directory for assembled parquets. When set explicitly, this flag wins over the env var. (FR-022) |
| `--cache-dir PATH` | path | `~/data_lake/fmp/deep-finance/` | FMP per-ticker parquet cache root. Mirrors the convention from `~/projects/ML_short_reversion/`. Distinct from `--out-dir`. |
| `--start DATE` | ISO date | universe-dependent | Override the earliest date to fetch. |
| `--end DATE` | ISO date | today (UTC) | Override the latest date to fetch. |
| `--rate-limit N` | int | 240 | FMP requests-per-minute cap (Starter tier ceiling is 250). |
| `--refresh-days N` | int | 7 | Trailing window re-fetched on each incremental update (picks up FMP back-revisions). |
| `--dry-run` | flag | off | Print the per-universe plan and exit without making any FMP calls. |
| `-v / --verbose` | flag | off | INFO logging (default is WARNING). |

## Environment variables

| Name | Required | Purpose |
|---|---|---|
| `FMP_API_KEY` | yes | FMP Premium credential. Loaded from `.env` via `python-dotenv` at process start. |
| `DEEP_FINANCE_DATA_DIR` | no | Output directory for parquet files. Default: `./data/`. Recommended developer setting: `~/data_lake/deep-finance/` (co-locates with the existing data lake). HF Space leaves it unset. Overridden by `--out-dir` when both are present. (FR-022) |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. At least one parquet was written (or `--dry-run` completed). A `shares_outstanding` skip warning does NOT change the exit code (per OD-3). |
| `1` | Missing or unreadable `FMP_API_KEY`. No parquet files written. |
| `2` | FMP authentication rejected (HTTP 401/403). No parquet files written. |
| `3` | FMP rate-limit exhaustion mid-fetch. Parquets that completed before the limit are kept on disk; the exit message names the universe whose fetch was interrupted so the operator can re-run only the missing universe. |
| `4` | Filesystem error writing the parquet (permissions, disk full). |
| `5` | `sp500_100` requested but OD-2 ticker confirmation is not on disk (see "100-stock confirmation flow" below). |

## Output schema (per parquet)

All three parquets share the same long-format schema:

| Column | Type | Notes |
|---|---|---|
| `date` | `date32[day]` | Trading day. UTC. |
| `symbol` | `string` | Ticker (e.g., `VTI`). |
| `open` | `float64` | |
| `high` | `float64` | |
| `low` | `float64` | |
| `close` | `float64` | Adjusted close. |
| `volume` | `int64` | |
| `shares_outstanding` | `float64` (nullable) | Present only if the OD-3 endpoint is available. When absent, the entire column is omitted from the parquet (not present-with-nulls). |

A sidecar file `data/<universe>.parquet.json` records:

```json
{
  "universe": "etf_basket",
  "fetched_at": "2026-05-16T14:23:11Z",
  "fmp_endpoint_set": ["historical-price-full", "key-metrics"],
  "shares_outstanding_available": true,
  "row_count": 7842,
  "symbols": ["VTI", "AGG", "DBC", "VIXY"],
  "date_range": {"min": "2011-01-04", "max": "2026-05-15"}
}
```

## Standard-out summary (success path)

On clean exit, prints one line per universe written:

```
[fetch_data] etf_basket: 7842 rows, 4 symbols, 2011-01-04 → 2026-05-15 (shares_outstanding: yes)
[fetch_data] sp500_20:   158430 rows, 20 symbols, 1992-11-25 → 2026-05-15 (shares_outstanding: yes)
[fetch_data] sp500_100:  SKIPPED (OD-2 confirmation pending — see specs/001-phase-0-skeleton-data/contracts/fetch_data_cli.md)
```

## 100-stock confirmation flow (OD-2)

The `sp500_100` universe ships its proposed ticker list as a separate file
that the developer reviews. Both the proposed and confirmed files live under
`${DEEP_FINANCE_DATA_DIR:-data/}` (FR-023) so the developer's data-lake copy
is the single source of truth for the confirmation state.

1. First run of `python scripts/fetch_data.py --universe sp500_100` writes
   `${DEEP_FINANCE_DATA_DIR}/sp500_100.proposed.json` (the Claude-Code-
   proposed sector-balanced 100-ticker list with rationale per sector) and
   exits with code 5.
2. The developer reviews and either:
   - Renames `sp500_100.proposed.json` → `sp500_100.confirmed.json` to accept
     as-is, OR
   - Edits the list inline and renames.
3. Next run with `--universe sp500_100` reads `sp500_100.confirmed.json` and
   proceeds with the fetch. If only `proposed.json` is present, the script
   re-exits with code 5 and the same message.
4. The default invocation (no `--universe`) silently skips `sp500_100` until
   `confirmed.json` exists. The summary line reports `SKIPPED (confirmation
   pending)` instead of failing the whole run.
5. Publish step: when the developer is ready to deploy, they copy the
   confirmed file alongside the parquets into `./data/`
   (`cp $DEEP_FINANCE_DATA_DIR/sp500_100.confirmed.json ./data/` followed by
   `git add`); HF Space then reads it from `/app/data/sp500_100.confirmed.json`
   the same way the live app does locally.

## Idempotency

- Re-running with no flags overwrites the parquets and sidecar JSONs atomically
  (write to a tempfile, fsync, rename). A partially-written parquet is never
  visible to the live app.
- Running concurrently in two shells is undefined behavior; the script does
  not take a lock. Developers run it interactively.

## Logging

Default level WARNING (only skip/error notices reach the terminal). `-v` lifts
to INFO. Each universe gets a per-symbol progress line at INFO. No DEBUG
level — FMP responses can contain the API key in URL traces; we don't log them.
