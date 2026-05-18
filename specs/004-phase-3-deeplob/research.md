# Research: Phase 3 — DeepLOB Page

**Branch**: `004-phase-3-deeplob` | **Date**: 2026-05-17

---

## R1 — DeepLOB architecture (paper §III)

**Decision**: Implement exactly per paper §III and `02_predictive_signal_lob.ipynb`.

```
input: (B, 1, T=100, 40)    — 100-tick lookback × 40 LOB features (10 levels each side, price+volume per level)

Conv-Block-1 (volume-imbalance / micro-price features):
  → Conv2d(1, 16, kernel=(1, 2), stride=(1, 2))     # pair adjacent bid/ask price-volume
  → LeakyReLU
  → Conv2d(16, 16, kernel=(4, 1))                    # temporal smoothing
  → LeakyReLU
  → Conv2d(16, 16, kernel=(4, 1))
  → LeakyReLU

Conv-Block-2 (next-level features):
  → Conv2d(16, 16, kernel=(1, 2), stride=(1, 2))
  → LeakyReLU
  → Conv2d(16, 16, kernel=(4, 1))
  → LeakyReLU
  → Conv2d(16, 16, kernel=(4, 1))
  → LeakyReLU

Conv-Block-3 (whole-book features):
  → Conv2d(16, 16, kernel=(1, 10))                   # collapse spatial dim to 1
  → LeakyReLU
  → Conv2d(16, 16, kernel=(4, 1))
  → LeakyReLU
  → Conv2d(16, 16, kernel=(4, 1))
  → LeakyReLU

Inception module (parallel-filter Network-in-Network):
  Branch 1: Conv2d(16, 32, kernel=(1, 1)) → LeakyReLU → Conv2d(32, 32, kernel=(3, 1))
  Branch 2: Conv2d(16, 32, kernel=(1, 1)) → LeakyReLU → Conv2d(32, 32, kernel=(5, 1))
  Branch 3: MaxPool2d(kernel=(3, 1), padding=(1, 0)) → Conv2d(16, 32, kernel=(1, 1))
  Concat along channel dim → (B, 96, T*, 1)

→ Reshape to (B, T*, 96)
→ LSTM(96, hidden=64, batch_first=True)
→ Take last time-step (B, 64)
→ Linear(64, 3)
→ Softmax(dim=-1)
→ output (B, 3)
```

Total params: ~140k.

**Alternatives considered**: Smaller LSTM hidden (32 → faster, ~50k params, but
paper specifies 64); dropout (paper doesn't use any).

---

## R2 — Baseline architectures

All baselines from Tsantekidis et al. 2017 (the immediate predecessor paper):

- **LOBSimpleMLP**: Flatten(T*40=4000) → Linear(4000, 64) → ReLU → Linear(64, 3)
  → Softmax. ~257k params. Tiny in compute but most params.
- **LOBCNN_I**: Conv1d on flattened sequence with kernel=10, stride=2,
  channels (1 → 14 → 14 → 32) → Flatten → Linear → Softmax. ~30k params.
- **LOBCNN_II**: Same shape as DeepLOB Conv-Block-1+2 but no Inception
  and no LSTM. ~10k params.
- **LOBLSTM**: Flatten LOB features per time step → LSTM(40, hidden=64) →
  Linear(64, 3) → Softmax. ~30k params.
- **LDA** (classical, sklearn): `LinearDiscriminantAnalysis()` on
  Flatten(T*40=4000). No deep learning; closed-form.

---

## R3 — Loss + training defaults

| Item | Value | Source |
|---|---|---|
| Loss | `nn.CrossEntropyLoss()` (with `label_smoothing=0` since FI-2010 labels are already discrete) | Paper §IV.B + standard |
| Optimizer | `Adam(lr=1e-3)` | Paper |
| Batch size | 64 | Paper §IV.B |
| Max epochs | 100 (cap; early-stops earlier) | Paper §IV.B |
| Patience | 10 | More aggressive than P1/P2 since FI-2010 is large enough that 10 epochs without improvement is conclusive |
| Min delta | 1e-3 | val loss is in [0, ~1.5]; meaningful improvement is ~0.001 |
| Seed | 42 | `torch.manual_seed(42)` + `np.random.seed(42)` |
| Val split | Last 10% of train | Paper §IV.B uses last fold of train as val for Setup 2 |

---

## R4 — Modal training structure

Mirrors Phase 1/2 trainers. Single file
`src/training/train_deeplob.py`. `--arch` dispatch:

```bash
modal run src/training/train_deeplob.py --arch DeepLOB
modal run src/training/train_deeplob.py --arch MLP
modal run src/training/train_deeplob.py --arch CNN1
modal run src/training/train_deeplob.py --arch CNN2
modal run src/training/train_deeplob.py --arch LSTM
```

5 sequential runs, ~$0.50 for DeepLOB + ~$0.10 each for the 4 baselines
= ~$0.90 total. Within budget (SC-006 says ≤ $1.50).

GPU choice: T4. Same as Phase 1/2; paper recommends A10G for production
but T4 is sufficient for FI-2010 (largest model is DeepLOB at 140k
params, well within T4's 16 GB VRAM).

---

## R5 — Demo-slice strategy

The full FI-2010 parquet (~300 MB compressed) is too large to ship in the
HF Space repo. The live app reads:
- `data/lob_fi2010_demo.parquet` (1 stock × 1 day from test set, ~10 MB)
  — for Tab 1 LOB snapshot, Tab 3B prediction visualization
- `data/backtests/lob_results.parquet` — pre-computed per-method metrics
  + confusion matrices (~10 KB)

The full parquet stays in `~/data_lake/fi2010/` (developer machine) +
the Modal Volume (training time).

Selection rule for the demo slice:
- Test set, day 8 (middle of the 3-day test window)
- Stock 1 (the most-liquid name in FI-2010 — most interesting LOB activity)
- All 40 features + all 5 horizon labels

---

## R6 — FI-2010 file structure + parsing

The Kaggle dataset has 4 .txt files in space-separated 149-row format:

```
Train_Dst_NoAuction_DecPre_CF_7.txt   607 MB  ← all 7 train days
Test_Dst_NoAuction_DecPre_CF_7.txt    132 MB  ← test day 8 (file naming
Test_Dst_NoAuction_DecPre_CF_8.txt    124 MB    is 1-indexed; CF_7 ≡ test
Test_Dst_NoAuction_DecPre_CF_9.txt     76 MB    day 1, CF_8 = day 2, etc.)
```

Each file: 149 rows × N columns (N varies by file, ~700k-3M timestamps).

Row mapping (per FI-2010 spec):
- Rows 1-40: 40 LOB features (z-score normalized,
  `NoAuction_DecPre_CF` variant)
- Rows 41-144: hand-crafted features (Hu invariants, etc.) — NOT used
  by DeepLOB (paper uses raw LOB only)
- Rows 145-149: 5 horizon labels at k ∈ {10, 20, 30, 50, 100} ticks
  - Values 1, 2, 3 → mapped to 0, 1, 2 (down / stationary / up)

Parsing pseudo-code:
```python
arr = np.loadtxt(path)           # (149, N)
features = arr[:40, :].T         # (N, 40)
labels = arr[144:149, :].T - 1   # (N, 5) — convert 1/2/3 → 0/1/2
day = inferred_from_filename
stock_ids = inferred_from_blocks_in_file   # FI-2010 packs 5 stocks per file in order
```

---

## R7 — Confusion-matrix + metrics computation

Pre-computed in `scripts/run_backtests.py --lob`:

For each (method, k) tuple:
- Load matching `.pt` (DeepLOB / MLP / CNN1 / CNN2 / LSTM) or fit LDA
- Predict on full test set
- Compute Accuracy, Precision (macro), Recall (macro), F1 (macro)
- Compute 3×3 confusion matrix
- Save to `lob_results.parquet`:
  `(method, k, accuracy, precision, recall, f1, cm_00, cm_01, ..., cm_22)`

---

## R8 — Tab 4A bijection (paper Table II columns)

| Paper column | Source |
|---|---|
| Accuracy % | `sklearn.metrics.accuracy_score` × 100 |
| Precision % | `precision_score(average='macro')` × 100 |
| Recall % | `recall_score(average='macro')` × 100 |
| F1 % | `f1_score(average='macro')` × 100 |

Macro averaging matches the paper's footnote (equal weight to each of
the 3 classes, accounting for the stationary class's higher prevalence).
