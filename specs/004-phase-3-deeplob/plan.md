# Implementation Plan: Phase 3 — DeepLOB Page (Zhang et al. 2019)

**Branch**: `004-phase-3-deeplob` | **Date**: 2026-05-17 | **Spec**: [spec.md](spec.md)

## Summary

Add the DeepLOB page replicating Zhang et al. 2019 Table II (Setup 2, k=10)
on the FI-2010 benchmark. 5 deep models (DeepLOB + MLP + CNN-I + CNN-II +
LSTM) trained on Modal; LDA classical baseline computed in-page via
sklearn. Live page reads a tiny demo slice (1 stock × 1 day, ~10 MB) +
pre-computed metrics panel. Full ~940 MB FI-2010 lives on the Modal Volume
for training only.

## Technical Context

**Language / runtime**: Python 3.11.

**Production deps** (`requirements.txt`): no new deps. scikit-learn (LDA),
plotly (LOB snapshot + heatmaps), pandas, numpy, torch all present.

**Training deps** (`requirements-train.txt`): no new deps.

**Datasets**:
- `~/data_lake/fi2010/`  — raw .txt files from Kaggle (~940 MB)
- `${DEEP_FINANCE_DATA_DIR}/lob_fi2010.parquet` — full benchmark
  (~300 MB compressed; lives on developer's machine + Modal Volume only)
- `data/lob_fi2010_demo.parquet` — ~10 MB demo slice committed to git

**Modal Volume**: `dl-research-data` (already created).

**Test pretrained**: 5 new `.pt` + `.json` sidecars committed to
`data/pretrained/` joining the existing 4 (Phase 1 + Phase 2).

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| I. Paper-Faithful Replication | ✓ | Tab 4A replicates Table II directly; Tab 1 cites Eq. 1 + Eq. 4; substrate disclosure block notes "reproduced here" vs "paper-reported" rows. |
| II. Device-Agnostic Torch (NON-NEGOTIABLE) | ✓ | All 5 models follow the same no-`.cuda()` pattern as Phase 1/2. |
| III. Two-Compute-Environment Discipline | ✓ | Full parquet stays off the production image — only the demo slice + pre-computed metrics ship to HF. "Train your own" surfaces CLI command. |
| IV. Data-as-Artifact (parquet, not DB) | ✓ | Two parquets + 5 .pt + 5 .json sidecars + a metrics panel parquet. |
| V. Pre-existing Canon | ✓ | `src/losses.py` is unused for this phase (classification uses `nn.CrossEntropyLoss`); `src/early_stopper.py` reused. New `_DatasetLOBClassification` class is local to `train_deeplob.py`. |
| VI. Test Critical Paths | ✓ | 5 new test modules (FRs 018-022) cover models, smoothed labels, checkpoint smoke, train smoke, page render. |

**Result: 6/6 PASS** — no waivers.

## Project Structure

### Documentation (this feature)

```
specs/004-phase-3-deeplob/
├── spec.md
├── plan.md                ← this file
├── research.md            ← decisions: architecture details, label smoothing, Modal setup
├── data-model.md          ← 4 entities (LOBDataset, LOBDemoSlice, DeepLOBCheckpoint, LOBMetricsPanel)
├── quickstart.md          ← operator walkthrough
├── tasks.md               ← dependency-ordered task list
└── contracts/
    ├── lob_models.md
    ├── train_deeplob_cli.md
    ├── fetch_lob_fi2010_cli.md
    └── order_book_page_ui.md
```

### Source Code (new in Phase 3)

```
src/
├── models/
│   └── deeplob.py                [+] 5 model classes
├── strategies/
│   └── lob_classical.py          [+] LDA wrapper
└── training/
    └── train_deeplob.py          [+] Modal app + train() body

pages/
└── 3_📖_Order_Book.py            [+] rewrite (replace Phase 0 placeholder)

scripts/
├── fetch_lob_fi2010.py           [+] Kaggle download + parquet build
└── run_backtests.py              [~] add --lob flag

data/
├── lob_fi2010_demo.parquet       [+] tiny demo slice (~10 MB)
├── pretrained/
│   ├── deeplob_fi2010_k10.{pt,json}    [+] from Modal training
│   ├── mlp_fi2010_k10.{pt,json}        [+]
│   ├── cnn1_fi2010_k10.{pt,json}       [+]
│   ├── cnn2_fi2010_k10.{pt,json}       [+]
│   └── lstm_fi2010_k10.{pt,json}       [+]
└── backtests/
    └── lob_results.parquet       [+] pre-computed metrics + confusion matrices

tests/
├── unit/
│   ├── test_deeplob_models.py             [+]
│   ├── test_smoothed_labels.py            [+]
│   ├── test_lob_checkpoint_smoke.py       [+]
│   └── test_lob_train_smoke.py            [+]
└── integration/
    └── test_order_book_page.py            [+]
```

## Primary Dependencies

No new deps. `kaggle` CLI installed on developer's machine but not in
requirements.txt (Kaggle download is a developer-local one-time step).

## Testing strategy

- Same self-contained pattern as Phase 1/2.
- Demo slice IS the test fixture for `test_lob_train_smoke.py` — gives
  realistic data without needing the full benchmark.

## Post-Design Constitution Re-Check

After drafting contracts: all 6 principles still hold.

## Complexity Tracking

| Element | Estimated lines | Risk |
|---|---|---|
| `src/models/deeplob.py` | ~250 | Medium (DeepLOB has Inception + LSTM, more complex than P1/P2 MLPs) |
| `src/strategies/lob_classical.py` | ~50 | Low (sklearn wrapper) |
| `src/training/train_deeplob.py` | ~450 | Medium (per-arch dispatch + classification metrics) |
| `scripts/fetch_lob_fi2010.py` | ~200 | Medium (Kaggle CLI integration + .txt parsing) |
| `scripts/run_backtests.py` --lob | +150 | Low (mirror existing --portfolio) |
| `pages/3_📖_Order_Book.py` | ~600 | Medium (4 tabs × interactive LOB snapshot + confusion matrices) |
| Tests (5 files) | ~400 | Low |

Total: ~2,100 LOC. Largest phase yet.
