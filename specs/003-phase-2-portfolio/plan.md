# Implementation Plan: Phase 2 — Portfolio Page (Zhang et al. 2020)

**Branch**: `003-phase-2-portfolio` | **Date**: 2026-05-17 | **Spec**: [spec.md](spec.md)

## Summary

Build the Portfolio page replicating Zhang/Zohren/Roberts (2020) Table 1 and
Figure 3 on two universes (ETF basket + sp500_20). Architecture: 4-classical-
benchmark library (Equal Weight, Min Variance, Max Diversification, DWP) +
DeepPortfolioMLP with Softmax output + Sharpe loss. Two Modal-trained
pretrained `.pt` checkpoints, Tab 4 with metrics table (4A) and 3-panel
cumulative-return overlay (4B). All Phase 1 plumbing reused: Modal app
pattern, per-day portfolio Sharpe-loss aggregation, date_range filtering,
`(reproduced here)` badging, bundled-CSV fallback.

## Technical Context

**Language / runtime**: Python 3.11 (pinned by `Dockerfile`).

**Production deps** (`requirements.txt`, already complete from Phase 0/1):
- streamlit, plotly, pandas, pyarrow, numpy, scikit-learn, requests,
  python-dotenv, pytest. **scikit-learn was already there** because Phase 0
  anticipated `LedoitWolf` covariance for the Portfolio page.
- torch (CPU index). No new dep needed.

**Training deps** (`requirements-train.txt`): torch (GPU index via Modal),
numpy, pandas, pyarrow. No new dep needed.

**Datasets** (FMP fetched, committed):
- `data/etf_basket.parquet`  (4 symbols, 15,459 rows, 2011-01-03 → today,
  shares_outstanding present → DWP works)
- `data/sp500_20.parquet`   (20 symbols, 100,000 rows, 2006-06-30 → today,
  shares_outstanding present)

**Modal Volume**: `dl-research-data` (already created in Phase 1).

**Test pretrained**: 2 new `.pt` + `.json` sidecars committed to
`data/pretrained/` joining the Phase 1 Momentum checkpoints.

**Project type**: single project (continuation of Phase 0/1 layout).

## Constitution Check

Reviewed against `.specify/memory/constitution.md` v1.1.0.

| Principle | Status | Notes |
|---|---|---|
| I. Paper-Faithful Replication | ✓ | Tab 4 replicates Table 1 + Fig 3 directly; substrate disclosure on Tab 1 explains universe vs paper's ETF-only-with-S&P; loss-scope-discipline (Sharpe only) is documented. |
| II. Device-Agnostic Torch (NON-NEGOTIABLE) | ✓ | `DeepPortfolioMLP` has no `.cuda()`/hardcoded device; `train_deep_portfolio.py` mirrors Phase 1's Modal-decorator-top / device-agnostic-body pattern. |
| III. Two-Compute-Environment Discipline | ✓ | Modal Volume + container image for training; production CPU image runs the page; "Train your own" expander surfaces CLI command only (Phase 1 precedent). |
| IV. Data-as-Artifact (parquet, not DB) | ✓ | Two input parquets + one backtest parquet (`portfolio_results.parquet`) + 2 .pt + 2 .json sidecars. No DB, no API backend. |
| V. Pre-existing Canon | ✓ | `src/losses.py:SharpeLoss/Neg_Sharpe`, `src/early_stopper.py:EarlyStopping`, `src/torch_data.py:MyDataset`, `src/metrics.py` all used as-is. `_DailyPortfolioDataset` from Phase 1's training script is the precedent for local dataset variants. |
| VI. Test Critical Paths | ✓ | 4 new test modules (FRs 019/020/021/022) cover classical methods, softmax invariant, checkpoint load, page render. |

**Result: 6/6 PASS — no waivers needed.**

## Project Structure

### Documentation (this feature)

```
specs/003-phase-2-portfolio/
├── spec.md
├── plan.md                ← this file
├── research.md            ← decisions: architecture, hyperparams, classical lib choice
├── data-model.md          ← 4 entities (PortfolioWeights, DeepPortfolioCheckpoint, PortfolioBacktestPanel, UniverseDataset)
├── quickstart.md          ← operator walkthrough
├── tasks.md               ← dependency-ordered task list (Setup → ... → Polish)
└── contracts/
    ├── deep_portfolio_model.md
    ├── train_deep_portfolio_cli.md
    ├── portfolio_page_ui.md
    └── classical_benchmarks_api.md
```

### Source Code (repository root) — new in Phase 2

```
src/
├── strategies/
│   └── portfolios.py             [+] 4 classical portfolio methods
├── models/
│   └── deep_portfolio.py         [+] DeepPortfolioMLP
└── training/
    └── train_deep_portfolio.py   [+] Modal app + device-agnostic train() body

pages/
└── 2_💼_Portfolio.py             [+] rewrite (replace Phase 0 placeholder)

scripts/
└── run_backtests.py              [+] add --portfolio flag

data/
├── etf_basket.parquet            [+] already committed (this commit)
├── sp500_20.parquet              [+] already committed
├── pretrained/
│   ├── deep_portfolio_etfs.pt    [+] from Modal training (US2)
│   ├── deep_portfolio_etfs.json
│   ├── deep_portfolio_20stock.pt
│   └── deep_portfolio_20stock.json
└── backtests/
    └── portfolio_results.parquet [+] from scripts/run_backtests.py --portfolio

tests/
├── unit/
│   ├── test_classical_portfolios.py    [+] FR-019
│   ├── test_softmax_head.py            [+] FR-020
│   ├── test_portfolio_checkpoint_smoke.py [+] FR-021
│   └── test_portfolio_train_smoke.py   [+] FR-021
└── integration/
    └── test_portfolio_page.py          [+] FR-022
```

## Primary Dependencies

No new dependencies. Phase 0's `requirements.txt` already includes
scikit-learn (for `LedoitWolf`/`ShrunkCovariance`) and torch.

## Testing strategy

- **Self-contained**: no test imports from external repos (Phase 1 precedent
  / Constitution VI).
- **Unit tests** (FR-019/020/021): synthetic fixtures with fixed seed, no
  network, no FMP, no Modal.
- **Integration tests** (FR-022): `streamlit.testing.v1.AppTest` parametrised
  across 4 (universe × parquet-present) combinations.
- **Smoke tests**: a single-epoch CPU train via the device-agnostic `train()`
  body. Phase 1 precedent in `test_train_smoke.py`.

## Post-Design Constitution Re-Check

After drafting the contracts, all 6 principles still hold. The classical
benchmarks add scikit-learn usage but no new principle interactions.

## Complexity Tracking

| Element | Estimated lines | Risk |
|---|---|---|
| `src/strategies/portfolios.py` | ~150 | Low (formulas in research.md R2) |
| `src/models/deep_portfolio.py` | ~60 | Low (mirrors `deep_momentum.py`) |
| `src/training/train_deep_portfolio.py` | ~350 | Medium (per-day aggregation reuses Phase 1's pattern; per-universe scaffolding adds dispatch logic) |
| `scripts/run_backtests.py` --portfolio | +100 | Low (mirror existing --momentum) |
| `pages/2_💼_Portfolio.py` | ~500 | Medium (4 tabs × 2 universes + cost sweep) |
| Tests (5 files) | ~300 | Low |

Total: ~1500 LOC. Smaller than Phase 1 (which was ~1800 LOC for the page
alone due to the box-plot section); Phase 2 has fewer charts but more
classical methods.
