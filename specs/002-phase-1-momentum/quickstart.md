# Quickstart: Phase 1 — Momentum Page

**Audience**: developers running the Phase 1 deliverable for the first time.
**Prerequisites**: Phase 0 complete, `.venv` populated with
`requirements.txt`, `~/data_lake/databento/futures/L0/ohlcv-1d/` populated
with the BCOM parquet (or `--lake-parquet` override available).

This walkthrough exercises all three Phase 1 user stories end-to-end so a
reviewer can validate the feature.

---

## Pre-flight (one-time)

```bash
# Modal CLI on the developer's machine (NOT in requirements.txt)
pip install modal
modal token new                                # opens browser; writes ~/.modal.toml
modal volume create dl-research-data           # idempotent if Phase 0 already created it

# Populate the Modal Volume with the just-fetched parquets
modal volume put dl-research-data \
       "$DEEP_FINANCE_DATA_DIR/cme_futures.parquet" \
       /cme_futures.parquet
```

---

## US1 — Read the Momentum story end-to-end (P1)

**Goal**: prove the four-tab Momentum page renders with real data.

### One-time: fetch the futures parquet

```bash
# Honours DEEP_FINANCE_DATA_DIR; default ./data/
python scripts/fetch_futures.py
```

Expected stdout:

```
[fetch_futures] 18 BCOM roots: 71200 rows, 2010-06-06 → 2026-05-05 (F0 ratio-adjusted, roll_offset=-5)
```

### Render the page

```bash
.venv/bin/streamlit run streamlit_app.py
# open http://localhost:7860 → click the 📈 Momentum card
```

**Expected**:

- Page header shows Lim/Zohren/Roberts 2019 citation + arXiv link.
- Sidebar shows the six widgets listed in `contracts/momentum_page_ui.md`.
- Tab 1 shows the 18-root universe summary + substrate disclosure
  ("BCOM commodity roots", "Pinnacle CLC", commodities-only ordering
  claim).
- Tab 2 overlays Long Only / Sgn(Returns) / MACD cumulative-return
  curves.
- Tab 3 shows the MLP equity curve and a model-card badge with
  `trained_with`, `trained_on`, `test_annual_sharpe`.
- Tab 4 sub-tabs 4A/4B render 5-row metric tables (best-per-column
  bolded); 4C is a single overlay chart; 4D is three box plots.
- Math + Code expanders work.

---

## US2 — Train the deep models on Modal (P2)

**Goal**: prove `src/training/train_deep_momentum.py` produces both
checkpoints reproducibly on Modal.

### Train MLP

```bash
modal run src/training/train_deep_momentum.py --arch MLP
```

Expected: container spins up (~30 s for warm image, longer first time),
training runs for up to 200 epochs with early-stopping cutoff, prints
per-epoch loss at INFO, prints `modal volume get` command at the end.

### Train LSTM

```bash
modal run src/training/train_deep_momentum.py --arch LSTM
```

Same shape; produces `lstm_sharpe.pt`.

### Pull checkpoints back to the repo

```bash
modal volume get dl-research-data /pretrained/mlp_sharpe.pt   ./data/pretrained/mlp_sharpe.pt
modal volume get dl-research-data /pretrained/mlp_sharpe.json ./data/pretrained/mlp_sharpe.json
modal volume get dl-research-data /pretrained/lstm_sharpe.pt  ./data/pretrained/lstm_sharpe.pt
modal volume get dl-research-data /pretrained/lstm_sharpe.json ./data/pretrained/lstm_sharpe.json
git add data/pretrained/*.pt data/pretrained/*.json
git commit -m "feat(phase-1): train mlp_sharpe + lstm_sharpe (Modal T4)"
```

### CPU smoke run (no Modal required)

```bash
.venv/bin/python -m src.training.train_deep_momentum --arch MLP --max-epochs 1
```

Expected: same code path runs on CPU (slower, 1-epoch cap, smaller
batch). Writes a `.pt` to a tempdir or `data/pretrained/` depending on
the script's `--checkpoint-dir` default.

---

## US3 — Reviewer verifies paper-faithful replication (P3)

**Goal**: prove all four test modules pass.

```bash
.venv/bin/python -m pytest tests/unit/test_tsmom.py             -v
.venv/bin/python -m pytest tests/unit/test_vol_targeting.py     -v
.venv/bin/python -m pytest tests/unit/test_checkpoint_smoke.py  -v
.venv/bin/python -m pytest tests/unit/test_train_smoke.py       -v
.venv/bin/python -m pytest tests/integration/test_momentum_page.py -v
```

**Expected**:

- `test_tsmom.py`: 3 tests pass (Long Only, Sgn(Returns), MACD parity
  with `~/projects/QIS_Commodities/src/signals/trend.py` within 1e-10).
- `test_vol_targeting.py`: 3 tests pass (σ_target = 0.15 produces
  realized vol ≈ 0.15 within 5 % tolerance; edge cases 0.01 and 1.0
  produce finite output).
- `test_checkpoint_smoke.py`: 2 tests pass (both `.pt` files load on
  CPU, non-NaN forward pass).
- `test_train_smoke.py`: 1 test passes (CPU `train()` call writes a
  loadable `.pt`).
- `test_momentum_page.py`: 6 parametrised cases pass.

### Re-generate the backtest panel

```bash
.venv/bin/python scripts/run_backtests.py --momentum
```

Expected: `data/backtests/momentum_results.parquet` written with rows
for all 5 strategies × 2 vol-scaling × 18 contracts × ~4000 trading days
(~720 k rows, < 50 MB).

---

## Acceptance check (run before declaring Phase 1 done)

```bash
.venv/bin/python -m pytest -v
# expected: 21 (Phase 0) + 15 (Phase 1) = ~36 tests pass

ls data/pretrained/mlp_sharpe.pt data/pretrained/mlp_sharpe.json \
   data/pretrained/lstm_sharpe.pt data/pretrained/lstm_sharpe.json \
   data/backtests/momentum_results.parquet \
   "$DEEP_FINANCE_DATA_DIR/cme_futures.parquet"
# all six paths should exist

.venv/bin/streamlit run streamlit_app.py
# manually click Momentum → walk through all four tabs → verify A1–A11 acceptance
```

When all of those are green, Phase 1 is done and Phase 2 can begin.
