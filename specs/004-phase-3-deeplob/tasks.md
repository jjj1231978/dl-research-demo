---
description: "Task list for Phase 3 — DeepLOB Page (Zhang et al. 2019)"
---

# Tasks: Phase 3 — DeepLOB Page

## Phase 1: Setup

- [x] T001 Confirm `~/.kaggle/kaggle.json` works: `.venv/bin/kaggle datasets list --max-size 1` succeeds. *(Done in spec session.)*
- [x] T002 Confirm `kaggle` pip package is installed in `.venv`. *(Done.)*

## Phase 2: Foundational (Blocking)

- [x] T003 [P] Implement `src/models/deeplob.py` per `contracts/lob_models.md`: 5 classes (DeepLOB, LOBSimpleMLP, LOBCNN_I, LOBCNN_II, LOBLSTM). All device-agnostic, all 3-way softmax output.
- [x] T004 [P] Implement `src/strategies/lob_classical.py`: LDA wrapper over sklearn. Self-contained.
- [x] T005 Implement `scripts/fetch_lob_fi2010.py` per contract: Kaggle download → 4 .txt parse → parquet. Idempotent. Supports `--demo-only` for HF Space slice.
- [x] T006 Implement `src/training/train_deeplob.py` per contract: Modal app + device-agnostic `train()` body. Per-arch dispatch. Per-tick sliding window. `nn.CrossEntropyLoss()`.
- [x] T007 Extend `scripts/run_backtests.py` with `--lob` flag. Loads each `.pt`, predicts on test split, computes accuracy/precision/recall/F1 + 3×3 confusion matrix. Adds paper-reported rows from a hard-coded table.

## Phase 3: US1 — MVP

- [x] T008 [US1] Run `scripts/fetch_lob_fi2010.py` against Kaggle. Verify `lob_fi2010.parquet` + `lob_fi2010_demo.parquet` land.
- [x] T009 [US1] Run `scripts/run_backtests.py --lob` to seed the metrics parquet with paper-reported rows (deep rows fill in post-US2).
- [x] T010 [US1] Create `tests/integration/test_order_book_page.py`: 4 parametrised cases per contract; substrate disclosure + setup-instruction banner copy-text invariants.
- [x] T011 [US1] Rewrite `pages/3_📖_Order_Book.py` per contract: sidebar, header, 4 tabs, math + code panels.
- [x] T012 [US1] Tab 1 body — LOB snapshot Plotly chart, class-balance bar chart, smoothed-label visualization.
- [x] T013 [US1] Tab 2 body — baseline picker, predictions overlay, metrics card.
- [x] T014 [US1] Tab 3 body — 3 sub-tabs (architecture explainer, live prediction demo, train-your-own CLI).
- [x] T015 [US1] Tab 4 body — 3 sub-tabs (Table II, confusion matrices, per-class F1).
- [x] T016 [US1] Math + Code panels.
- [x] T017 [US1] Run `pytest tests/integration/test_order_book_page.py -v`. All pass.

## Phase 4: US2 — Modal training (developer-action)

- [x] T018 [US2] Upload FI-2010 parquet to Modal Volume.
- [x] T019 [US2] Train DeepLOB. *(F1=0.712, Acc=0.83, 53 epochs on T4.)*
- [x] T020 [US2] Train MLP, CNN-I, CNN-II, LSTM baselines.
- [x] T021 [US2] Pull all 5 checkpoints + sidecars.
- [x] T022 [US2] Inspect sidecars: confirm `arch`, `setup=2`, `k=10`, `trained_with` starts with "Modal ".
- [x] T023 [US2] Re-run `scripts/run_backtests.py --lob` to populate deep-method rows in the metrics panel.

## Phase 5: US3 — unit tests

- [x] T024 [P] [US3] `tests/unit/test_deeplob_models.py`: forward-pass smoke for each of 5 models.
- [x] T025 [P] [US3] `tests/unit/test_smoothed_labels.py`: Eq. 4 formula equality.
- [x] T026 [P] [US3] `tests/unit/test_lob_checkpoint_smoke.py`: load each `.pt`; skip-when-absent.
- [x] T027 [P] [US3] `tests/unit/test_lob_train_smoke.py`: CPU one-epoch training on demo slice.
- [x] T028 [US3] Run all 4 unit-test modules; verify all pass.

## Phase 6: Polish + Deploy

- [x] T029 Run full `pytest -v` — ~106 tests. *(115 passed, 0 skipped after US2 checkpoints landed.)*
- [x] T030 Merge `004-phase-3-deeplob` → main. *(Merge commit 43c1488; no-ff to preserve branch history.)*
- [x] T031 Push to GitHub + HF, wait for rebuild, verify Order Book card live.

## Dependencies

- Setup → Foundational → US1 (page MVP).
- US2 depends on Foundational T006 + manual operator.
- US3 depends on Foundational. Checkpoint smoke skips pre-US2.
- Polish after all three user stories.
