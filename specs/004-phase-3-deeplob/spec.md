# Feature Specification: Phase 3 — DeepLOB Page (Zhang et al. 2019 replication)

**Feature Branch**: `004-phase-3-deeplob`
**Created**: 2026-05-17
**Status**: Draft
**Paper**: Zhang, Zohren, Roberts (2019). *DeepLOB: Deep Convolutional Neural
Networks for Limit Order Books*. IEEE TSP. arXiv:1808.03668.

## Summary

Build the third per-paper page — replication of Zhang et al. 2019 **Table II**
on the FI-2010 benchmark. End-to-end deep classification: a CNN + Inception +
LSTM that consumes 10 levels of LOB depth × 100-tick lookback and outputs a
3-way softmax over (down / stationary / up) mid-price movement at k=10
ticks ahead. Cross-entropy loss on smoothed labels (paper Eq. 4). Five
classical/baseline models trained alongside (LDA + MLP + CNN-I + CNN-II +
LSTM) for the Tab 4 master table. Paper-replication-grade page on the same
footing as Momentum and Portfolio.

## Clarifications

### Session 2026-05-17

- Q: How many DeepLOB checkpoints? → **A: Setup 2, k=10 only.** Single
  checkpoint, paper's headline configuration. Setup 1 (anchored-forward
  cross-validation, 9 folds) and other k values are paper-reported only
  (cited from Table I / Table II). Modal cost ~$0.50 on T4.
- Q: How many classical baselines to implement vs cite from paper? → **A:
  All Setup-2 deep baselines (MLP + CNN-I + CNN-II + LSTM) + LDA.** Five
  models reproduced locally; the remaining Setup-2 baselines (SVM, BoF,
  TABL variants, MCSDA) are surfaced as paper-reported with a clear
  "(paper-reported)" badge. Modal cost for the 4 deep baselines ≈ $0.40
  combined (each smaller than DeepLOB). LDA runs on CPU via scikit-learn,
  no Modal needed.
- Q: Kaggle credentials? → **A: Yes — `~/data_lake/Kaggle/.env` contained
  the API key; persisted at canonical `~/.kaggle/kaggle.json` with username
  jianjianjin.** Confirmed working with `kaggle datasets list`. FI-2010
  dataset (`praanj/limit-orderbook-data`, ~940 MB across 4 .txt files)
  reachable.
- Q: Data subset to ship on the HF Space? → **A: Pre-computed metrics
  panel + a 1-stock × 1-day "demo slice" parquet (~10 MB).** Full FI-2010
  parquet (~200-300 MB after compression) is too large for the HF Space
  Docker image and is not needed at render time — the page reads
  pre-computed metrics + the demo slice for the Tab 3B interactive
  prediction visualizer. Full parquet lives on the Modal Volume for
  training only.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Researcher reads the DeepLOB story end-to-end (Priority: P1) — MVP

Researcher arrives from arXiv link, clicks the **📖 Order Book** card,
lands on a fully working page. Tab 1 shows what a LOB is + how FI-2010 is
labeled. Tab 2 surfaces the 4 classical baselines (LDA / MLP / CNN-I /
CNN-II / LSTM) — each loadable from a committed checkpoint. Tab 3A is a
static architecture explainer; Tab 3B steps through the demo slice with
DeepLOB's predictions overlaid; Tab 3C surfaces the CLI command for
"train your own". Tab 4 sub-tabs render the Table II replication, the
3×3 confusion matrices, and the per-class metric curves.

**Independent Test**:
- (a) `python scripts/fetch_lob_fi2010.py --demo-only` produces the demo
  slice + metrics panel without requiring the full benchmark
- (b) `streamlit run streamlit_app.py` → Order Book card → all four tabs
  render without exception
- (c) Tab 4A shows a 6-row × 4-column Table II replication (5 baselines +
  DeepLOB; Accuracy / Precision / Recall / F1)

### User Story 2 — Developer retrains via Modal (Priority: P2)

`modal run src/training/train_deeplob.py` produces all 5 deep checkpoints
(MLP, CNN-I, CNN-II, LSTM, DeepLOB) reproducibly. Developer pulls them
with `modal volume get` and commits.

### User Story 3 — Reviewer audits paper-faithful replication (Priority: P3)

`pytest tests/` exercises 4 new test modules: `test_deeplob_models.py`
(forward-pass smoke for each architecture, output is a 3-way softmax),
`test_smoothed_labels.py` (Eq. 4 formula equality on a fixed-seed
fixture), `test_lob_checkpoint_smoke.py` (each `.pt` loads cleanly,
skip-when-absent), `test_lob_train_smoke.py` (CPU one-epoch training on
the demo slice).

## Functional Requirements *(mandatory)*

**Data layer**:

- **FR-001**: `scripts/fetch_lob_fi2010.py` MUST authenticate with
  Kaggle (via `~/.kaggle/kaggle.json` OR env vars), download
  `praanj/limit-orderbook-data` to `~/data_lake/fi2010/`, parse the 4
  `.txt` files into a single long-format parquet at
  `${DEEP_FINANCE_DATA_DIR}/lob_fi2010.parquet`. Idempotent (re-download
  is skipped if files exist).
- **FR-002**: Output parquet schema: `day` (1-10), `stock_id` (1-5),
  `tick` (timestamp within day), 40 LOB features
  (`bid_price_1..bid_price_10`, `bid_volume_1..bid_volume_10`,
  `ask_price_1..ask_price_10`, `ask_volume_1..ask_volume_10`), 5 labels
  (`label_k10`, `label_k20`, `label_k30`, `label_k50`, `label_k100` —
  values in {0, 1, 2} = (down, stationary, up)). Per FI-2010 standard,
  features are z-score normalized (`NoAuction_DecPre_CF`).
- **FR-003**: `scripts/fetch_lob_fi2010.py --demo-only` MUST produce
  `data/lob_fi2010_demo.parquet` (≤ 20 MB) by subsampling 1 stock × 1 day
  from the test set. Required for the live HF Space.
- **FR-004**: The Modal training script MUST be invokable as
  `modal run src/training/train_deeplob.py --arch {DeepLOB|MLP|CNN1|CNN2|LSTM}`
  and as `python -m src.training.train_deeplob --arch DeepLOB --max-epochs 1`
  for the CPU smoke (Constitution Principle II).

**Models**:

- **FR-005**: `src/models/deeplob.py` MUST implement:
  - `DeepLOB(n_classes=3, lookback=100)` — CNN + Inception + LSTM →
    3-way softmax per paper §III
  - `LOBSimpleMLP(input_dim=4000, n_classes=3)` — baseline MLP on
    flattened features
  - `LOBCNN_I(n_classes=3, lookback=100)` — Tsantekidis et al. 2017 CNN-I
  - `LOBCNN_II(n_classes=3, lookback=100)` — Tsantekidis et al. 2017 CNN-II
  - `LOBLSTM(n_classes=3, lookback=100, hidden_size=64)` — LSTM baseline
- **FR-006**: All models device-agnostic (Principle II). All output a
  `(B, n_classes)` softmax vector that sums to 1 ± 1e-6.

**Classical baselines**:

- **FR-007**: `src/strategies/lob_classical.py` MUST implement
  `fit_lda(X_train, y_train, **kwargs)` and `predict_lda(model, X_test)`
  thin wrappers over `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`.
  Self-contained — paper-faithful but not paper-reimplemented.

**Training**:

- **FR-008**: `src/training/train_deeplob.py` is a Modal app per
  constitution v1.1.0. Reads `lob_fi2010.parquet` from
  `dl-research-data` Modal Volume. Per-arch dispatch via `--arch`.
- **FR-009**: Loss = `nn.CrossEntropyLoss()` on smoothed labels (paper
  Eq. 4 — labels already smoothed in the FI-2010 distribution; no
  re-smoothing needed). Optimizer = `Adam(lr=1e-3)`. Batch size 64.
- **FR-010**: Early stopping per canonical `src/early_stopper.py` on
  validation cross-entropy (~10% of train held out as val).
- **FR-011**: Five `.pt` files + JSON sidecars committed to
  `data/pretrained/`:
  `{deeplob, mlp, cnn1, cnn2, lstm}_fi2010_k10.{pt,json}`.

**Page rendering**:

- **FR-012**: `pages/3_📖_Order_Book.py` replaces Phase 0 placeholder.
  Sidebar: stock filter, prediction-horizon selector (k value), time-window
  slider over demo slice, classical-baseline picker for Tab 2.
- **FR-013**: Tab 1 — LOB snapshot at selected timestamp (Plotly stacked
  bars), class-balance summary, smoothed-label visualization (mid-price
  with up/down shading).
- **FR-014**: Tab 2 — pick a baseline, see its predictions on the demo
  slice, per-baseline metrics card.
- **FR-015**: Tab 3A — architecture diagram (markdown + ASCII art is
  acceptable; Plotly sankey optional). Tab 3B — interactive
  prediction-demo with sidebar time slider. Tab 3C — surface CLI command.
- **FR-016**: Tab 4 sub-tabs:
  - **4A** — Table II replication: 6 rows × 4 metric columns. Best F1
    bolded. Mix of "(reproduced here)" and "(paper-reported)" badges.
  - **4B** — Confusion matrices: 3×3 heatmaps for top 3 methods.
  - **4C** — Per-class F1 bar chart.
- **FR-017**: Math panel: mid-price formula, Eq. 4 smoothed-label
  convention, gated activation in first conv block, Inception module's
  parallel-filter design. Code panel:
  `inspect.getsource(DeepLOB)` + `inspect.getsource(LOBSimpleMLP)`.

**Tests**:

- **FR-018**: `tests/unit/test_deeplob_models.py` — forward-pass smoke
  for each architecture; output shape + softmax invariant; no NaN.
- **FR-019**: `tests/unit/test_smoothed_labels.py` — Eq. 4 formula on
  fixed-seed mid-price series; label values are in {0, 1, 2}.
- **FR-020**: `tests/unit/test_lob_checkpoint_smoke.py` — load each
  committed `.pt`; pytest.skip when absent.
- **FR-021**: `tests/unit/test_lob_train_smoke.py` — CPU one-epoch train
  on the demo slice; returns dict; writes `.pt`.
- **FR-022**: `tests/integration/test_order_book_page.py` — 4
  parametrised cases (demo_slice × checkpoints) = {absent, present}².

## Key Entities

- **LOBDataset** — long-format parquet at `lob_fi2010.parquet`. 40 LOB
  features per timestamp + 5 horizon labels. ~4M rows when fully unpacked.
- **LOBDemoSlice** — `data/lob_fi2010_demo.parquet` — 1-stock × 1-day
  subsample (~10 MB) shipped with the repo for the live HF Space.
- **DeepLOBCheckpoint** — `.pt` + JSON sidecar per `data-model.md` §E3.
  Five instances in Phase 3.
- **LOBMetricsPanel** — `data/backtests/lob_results.parquet` —
  per-(method, k) Accuracy/Precision/Recall/F1 metrics computed once
  offline.

## Success Criteria *(mandatory)*

- **SC-001**: Kaggle download + parquet build completes without error
  on the developer's machine.
- **SC-002**: All 4 tabs of `pages/3_📖_Order_Book.py` render in ≤ 3 s
  on HF CPU Basic.
- **SC-003**: DeepLOB Setup 2 k=10 F1 is qualitatively consistent with
  paper Table II — **DeepLOB MUST achieve F1 ≥ 60%** (paper reports
  77%; ours runs on the same FI-2010 dataset but may differ slightly
  due to random init / training hyperparams; ≥ 60% confirms the
  architecture is implemented correctly).
- **SC-004**: DeepLOB F1 > each of the 4 reproduced baselines (MLP /
  CNN-I / CNN-II / LSTM). Paper's qualitative ordering.
- **SC-005**: `pytest -v` exits 0 from a clean checkout.
- **SC-006**: Modal training run for DeepLOB ≤ 60 min on T4; total cost
  for all 5 models ≤ $1.50.

## Scope

**In scope**:
- FI-2010 dataset + Kaggle fetch script
- 5 PyTorch models (DeepLOB + 4 baselines)
- Modal training, 5 checkpoints + sidecars
- LDA classical baseline (sklearn, no Modal)
- Order Book page with 4 tabs
- Tab 4A Table II replication, 4B confusion matrices, 4C per-class F1
- Math + Code panels
- Tests (5 new modules)

**Out of scope** (Phase 4 stretch or future):
- Setup 1 (9-fold anchored-forward CV)
- Other prediction horizons (k ∈ {20, 30, 50, 100}) — paper-reported only
- Other baselines (SVM, BoF, TABL variants, MCSDA) — paper-reported only
- Tab 4E LIME sensitivity heatmaps
- Live in-page training (Tab 3C surfaces CLI command only)

## Critical References

- Paper: Zhang, Zohren, Roberts (2019). arXiv:1808.03668. IEEE TSP.
- Reference notebook: `notebooks/reference/02_predictive_signal_lob.ipynb`
  (note: original was regression with MSE on a sample CSV; Phase 3 is
  3-class classification with cross-entropy on FI-2010 — different task).
- FI-2010 paper: Ntakaris et al. (2017). Defines labeling protocol.
- Brief: `Project_brief.md` §§ 6 Page 3, 11 Phase 3, 4.1 (Kaggle fetch).
- Constitution: v1.1.0.
- Phase 1/2 lessons:
  - Per-day-aggregation pitfall doesn't apply here (classification, not Sharpe)
  - Tab 4 should default to test-set-only view
  - Backtest parquet must ship in git for live-app rendering
  - Pretrained .pt must ship in git (LFS-tracked)

## Open Decisions

- **OD-1** (settled in Clarifications): Demo slice strategy for HF Space.
- **OD-2** (informational): Setup 1 (9-fold CV) reserved as Phase 4
  stretch. If the developer wants it later, the training script can be
  extended with a `--setup 1 --fold N` flag.
