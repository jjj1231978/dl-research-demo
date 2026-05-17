# Quickstart: Phase 2 — Portfolio Page

**Audience**: developers picking up Phase 2 for the first time.
**Prereqs**: Phase 0 + Phase 1 merged into main. `.venv` with
`requirements.txt`. FMP_API_KEY in `.env`. Modal CLI authenticated
(`~/.modal.toml` populated). HF Space already provisioned (Phase 1).

---

## US1 — Read the Portfolio story end-to-end (P1, MVP)

### Fetch the parquets (one-time)

```bash
.venv/bin/python scripts/fetch_data.py --universe etf_basket --universe sp500_20 -v
```

Expected:
```
[fetch_data] etf_basket: 15459 rows, 4 symbols, 2011-01-03 → 2026-05-15 (shares_outstanding: yes)
[fetch_data] sp500_20: 100000 rows, 20 symbols, 2006-06-30 → 2026-05-15 (shares_outstanding: yes)
```

### Pre-compute the backtest panel

```bash
.venv/bin/python scripts/run_backtests.py --portfolio -v
```

Produces `data/backtests/portfolio_results.parquet` (~240 k rows).
Deep Portfolio rows only populate if the checkpoints exist; otherwise
warns and skips.

### Render the page

```bash
.venv/bin/streamlit run streamlit_app.py
# open http://localhost:7860 → click the 💼 Portfolio card
```

**Expected**:
- Header: paper citation + arXiv link
- Sidebar: 5 widgets + Phase 0 data-status component
- Tab 1: rolling-corr heatmap + universe metrics + substrate disclosure
- Tab 2: classical-benchmark overlay (Equal Weight, Min Var, Max Div, DWP
  if available) + weight-evolution heatmap
- Tab 3: model-card badge + Deep Portfolio equity curve vs best classical
- Tab 4: 3-panel Table 1 + 3-panel Figure 3 cumulative-return overlay
- Math + Code panels work

---

## US2 — Train the deep models on Modal (P2)

### Train ETF basket

```bash
modal run src/training/train_deep_portfolio.py --universe etfs
```

Expected: warm-image start ~30 s; training ~5-10 min on T4; ~$0.08 of
Modal credit.

### Train 20-stock

```bash
modal run src/training/train_deep_portfolio.py --universe 20stock
```

Expected: ~10-15 min wall-clock (larger model); ~$0.15 of Modal credit.

### Pull checkpoints back

```bash
modal volume get dl-research-data /pretrained/deep_portfolio_etfs.pt    ./data/pretrained/deep_portfolio_etfs.pt --force
modal volume get dl-research-data /pretrained/deep_portfolio_etfs.json  ./data/pretrained/deep_portfolio_etfs.json --force
modal volume get dl-research-data /pretrained/deep_portfolio_20stock.pt ./data/pretrained/deep_portfolio_20stock.pt --force
modal volume get dl-research-data /pretrained/deep_portfolio_20stock.json ./data/pretrained/deep_portfolio_20stock.json --force
git add data/pretrained/deep_portfolio_*.pt data/pretrained/deep_portfolio_*.json
git commit -m "feat(phase-2): train deep_portfolio_etfs + deep_portfolio_20stock (Modal T4)"
```

### Re-run backtests with the new checkpoints

```bash
.venv/bin/python scripts/run_backtests.py --portfolio -v
```

Tab 4 now shows all methods including Deep Portfolio.

---

## US3 — Reviewer verifies paper-faithful replication (P3)

```bash
.venv/bin/python -m pytest tests/unit/test_classical_portfolios.py     -v
.venv/bin/python -m pytest tests/unit/test_softmax_head.py             -v
.venv/bin/python -m pytest tests/unit/test_portfolio_checkpoint_smoke.py -v
.venv/bin/python -m pytest tests/unit/test_portfolio_train_smoke.py    -v
.venv/bin/python -m pytest tests/integration/test_portfolio_page.py    -v
```

Expected: all green. Full sweep: `pytest -v` should show ~70 tests
passing (49 Phase 0+1 + ~21 Phase 2).

---

## Acceptance check (run before declaring Phase 2 done)

```bash
.venv/bin/python -m pytest -v
# expected: ~70 tests pass

ls data/pretrained/deep_portfolio_etfs.pt data/pretrained/deep_portfolio_etfs.json \
   data/pretrained/deep_portfolio_20stock.pt data/pretrained/deep_portfolio_20stock.json \
   data/backtests/portfolio_results.parquet \
   data/etf_basket.parquet data/sp500_20.parquet
# all seven paths should exist

.venv/bin/streamlit run streamlit_app.py
# manually click Portfolio → walk through all four tabs → switch universe
# → toggle cost-rate / vol-scaling → verify Tab 4 panels populate
```

---

## Push to GitHub + HF

```bash
git push origin 003-phase-2-portfolio   # PR or merge to main
# (after merge to main:)
git checkout main && git pull
git push hf main
```

Wait ~3-5 min for HF Docker rebuild; verify live at
https://huggingface.co/spaces/JJ-JIN12345/dl-research-demo → click
💼 Portfolio card.

When all green, Phase 2 is done and Phase 3 (DeepLOB) can begin.
