# Contract: `scripts/fetch_futures.py`

**Type**: one-shot CLI, developer-local invocation only.
**Runs on**: CPU (Constitution Principle III — no GPU operations).
**Reads from**: `~/data_lake/databento/futures/L0/ohlcv-1d/` (the
developer's existing databento lake parquet).
**Writes to**: `${DEEP_FINANCE_DATA_DIR}/cme_futures.parquet` + sidecar
(see Phase 0 `fetch_data_cli.md` for sidecar JSON conventions).

## CLI surface

```bash
python scripts/fetch_futures.py [OPTIONS]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--out-dir PATH` | path | `data_root()` (honours `DEEP_FINANCE_DATA_DIR`) | OUTPUT directory for `cme_futures.parquet`. When set explicitly, this flag wins over the env var. |
| `--lake-parquet PATH` | path | `~/data_lake/databento/futures/L0/ohlcv-1d/ohlcv-1d_GLBX.MDP3_bcom_2010-06-06_2026-05-05.parquet` | Override the source lake parquet. Useful when the developer's databento dump has been re-named or re-versioned. |
| `--roots LIST` | comma-separated string | (all 18 BCOM roots from `src/data/futures/contracts.py`) | Restrict to specific roots (e.g., `--roots CL,ZC,NG`). |
| `--start DATE` | ISO date | `2010-06-06` (lake earliest) | Truncate the output start date. |
| `--end DATE` | ISO date | today (UTC) | Truncate the output end date. |
| `--roll-offset N` | int | `-5` | Business-day offset before delivery-month start. Negative = before. |
| `--dry-run` | flag | off | Print the per-root plan and exit. |
| `-v / --verbose` | flag | off | INFO logging (default: WARNING). |

## Environment variables

| Name | Required | Purpose |
|---|---|---|
| `DEEP_FINANCE_DATA_DIR` | no | Output dir override (FR-022, Phase 0). |

Note: no FMP / API credentials needed; this script only reads the local
databento lake.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success — `cme_futures.parquet` written. |
| `1` | Lake parquet not found (default path or `--lake-parquet` override). Suggests `python ~/data_lake/databento/scripts/download_databento.py`. |
| `2` | One or more requested roots not present in the lake (the BCOM 18 list is hard-coded in `src/data/futures/contracts.py`). |
| `3` | Output file write error (permissions, disk full). |

## Output schema

| Column | Type | Notes |
|---|---|---|
| `date` | `date32[day]` | Trading day. |
| `contract` | `string` | BCOM root, e.g. "CL", "ZC". 18 distinct values. |
| `asset_class` | `string` | All `"commodity"` for Phase 1 (per OD-1). |
| `price` | `float64` | F0 ratio-adjusted price. |
| `return` | `float64` | `pct_change` of `price`. |

Sorted lexicographically by `(contract, date)` for determinism (FR-004).

A sidecar file `cme_futures.parquet.json` records:

```json
{
  "universe": "cme_futures",
  "fetched_at": "2026-05-17T15:00:00Z",
  "source": "~/data_lake/databento/futures/L0/ohlcv-1d/...",
  "roots": ["BO", "C", "CL", "CT", "GC", "HG", "HO", "KC", "LE", "NG",
            "PA", "PL", "RB", "SB", "SI", "S", "W", "ZS"],
  "row_count": 71200,
  "date_range": {"min": "2010-06-06", "max": "2026-05-05"},
  "roll_offset_bdays": -5,
  "position": 0
}
```

(Root list illustrative — actual list comes from
`src/data/futures/contracts.py:BCOM_ROOTS` constant.)

## Standard-out summary

```
[fetch_futures] 18 BCOM roots: 71200 rows, 2010-06-06 → 2026-05-05 (F0 ratio-adjusted, roll_offset=-5)
```

## Idempotency

Re-running with the same flags against unchanged lake data produces a
byte-identical output (FR-004) — atomic write (tempfile → rename) per
the Phase 0 `fetch_data.py` pattern.
