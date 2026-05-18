# Contract: `scripts/fetch_lob_fi2010.py`

**Type**: developer-local CLI (Kaggle download + parse → parquet).
**Runs on**: CPU.
**Reads from**: Kaggle dataset `praanj/limit-orderbook-data`.
**Writes to**: `${DEEP_FINANCE_DATA_DIR}/lob_fi2010.parquet` (full) +
optionally `data/lob_fi2010_demo.parquet` (committed-to-git slice).

## CLI

```bash
python scripts/fetch_lob_fi2010.py [OPTIONS]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--out-dir PATH` | path | `data_root()` | Output dir for `lob_fi2010.parquet`. |
| `--raw-dir PATH` | path | `~/data_lake/fi2010/` | Where to download/cache the 4 Kaggle .txt files. |
| `--demo-only` | flag | off | Skip the full parquet; only produce `data/lob_fi2010_demo.parquet` (1 stock × 1 day from test set). |
| `--no-redownload` | flag | off | Skip the Kaggle download step (use existing files in `--raw-dir`). |
| `-v` | flag | off | INFO logging. |

## Environment variables

| Name | Required | Purpose |
|---|---|---|
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | one of these OR `~/.kaggle/kaggle.json` | Kaggle CLI auth. |
| `DEEP_FINANCE_DATA_DIR` | no | Output dir override per Phase 0 FR-022. |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Kaggle auth failed (no kaggle.json, no env vars) |
| 2 | Download failed (network, dataset not found) |
| 3 | Parse failed (file format unexpected) |
| 4 | Output write failed (FS, permissions) |

## Output schema

Per `data-model.md` §E1.

## Sidecar

`{out_dir}/lob_fi2010.parquet.json`:

```json
{
  "dataset": "FI-2010",
  "kaggle_source": "praanj/limit-orderbook-data",
  "fetched_at": "2026-05-17T...",
  "row_count": 4334567,
  "days": [1,2,3,4,5,6,7,8,9,10],
  "stocks": [1,2,3,4,5],
  "horizons": [10, 20, 30, 50, 100]
}
```
