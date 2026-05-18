# Quickstart: Phase 3 — DeepLOB Page

**Audience**: developer running Phase 3 for the first time.
**Prereqs**: Phase 0+1+2 merged into main. `.venv` with `requirements.txt`.
Kaggle credentials at `~/.kaggle/kaggle.json` (Phase 3 spec session).
Modal CLI authenticated (Phase 1).

---

## US1 — Read the DeepLOB story end-to-end (P1, MVP)

### Fetch FI-2010 (one-time)

```bash
.venv/bin/python scripts/fetch_lob_fi2010.py -v
```

Downloads ~940 MB of raw .txt from Kaggle to `~/data_lake/fi2010/`,
parses into a single parquet at `${DEEP_FINANCE_DATA_DIR}/lob_fi2010.parquet`
(~300 MB compressed), and writes `data/lob_fi2010_demo.parquet` (~10 MB).
Idempotent — re-running skips Kaggle download if files exist.

### Pre-compute the metrics panel

```bash
.venv/bin/python scripts/run_backtests.py --lob -v
```

Produces `data/backtests/lob_results.parquet` with per-method metrics +
confusion matrices. Skips deep-method rows if checkpoints absent.

### Render the page

```bash
.venv/bin/streamlit run streamlit_app.py
# → click 📖 Order Book card
```

---

## US2 — Train the deep models on Modal (P2)

### Upload FI-2010 to the Modal Volume (one-time)

```bash
modal volume put dl-research-data data/lob_fi2010.parquet /lob_fi2010.parquet --force
```

### Train each model

```bash
modal run src/training/train_deeplob.py --arch DeepLOB   # ~60 min, $0.50
modal run src/training/train_deeplob.py --arch MLP       # ~10 min, $0.08
modal run src/training/train_deeplob.py --arch CNN1      # ~15 min, $0.12
modal run src/training/train_deeplob.py --arch CNN2      # ~15 min, $0.12
modal run src/training/train_deeplob.py --arch LSTM      # ~15 min, $0.12
```

### Pull all checkpoints

```bash
for arch in deeplob mlp cnn1 cnn2 lstm; do
  modal volume get dl-research-data /pretrained/${arch}_fi2010_k10.pt \
    ./data/pretrained/${arch}_fi2010_k10.pt --force
  modal volume get dl-research-data /pretrained/${arch}_fi2010_k10.json \
    ./data/pretrained/${arch}_fi2010_k10.json --force
done
git add data/pretrained/*_fi2010_k10.*
git commit -m "feat(phase-3): train 5 FI-2010 checkpoints on Modal T4"
```

### Regenerate metrics panel

```bash
.venv/bin/python scripts/run_backtests.py --lob -v
```

Now populates all 5 reproduced methods in `lob_results.parquet`.

---

## US3 — Reviewer audits paper-faithful replication (P3)

```bash
.venv/bin/python -m pytest tests/unit/test_deeplob_models.py            -v
.venv/bin/python -m pytest tests/unit/test_smoothed_labels.py           -v
.venv/bin/python -m pytest tests/unit/test_lob_checkpoint_smoke.py      -v
.venv/bin/python -m pytest tests/unit/test_lob_train_smoke.py           -v
.venv/bin/python -m pytest tests/integration/test_order_book_page.py    -v
```

Total Phase 3 tests: ~30. Full sweep `pytest -v` should show ~106 tests
(76 Phase 0+1+2 + ~30 Phase 3).

---

## Acceptance check

```bash
.venv/bin/python -m pytest -v
# ~106 tests pass

ls data/pretrained/{deeplob,mlp,cnn1,cnn2,lstm}_fi2010_k10.{pt,json} \
   data/backtests/lob_results.parquet data/lob_fi2010_demo.parquet
# all 13 paths should exist

.venv/bin/streamlit run streamlit_app.py
# walk through Order Book → all 4 tabs → switch baselines + slider
```

Push to both remotes when green; HF rebuilds in ~5 min.
