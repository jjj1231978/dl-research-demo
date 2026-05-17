# Implementation Plan: Phase 0 — Skeleton & Data Foundation

**Branch**: `001-phase-0-skeleton-data` | **Date**: 2026-05-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-phase-0-skeleton-data/spec.md`

## Summary

Deliver the repository scaffold, data-loading layer, FMP fetch script, landing
page, and canonical-utility adoption that every later phase depends on. The
headline outcome is first-clone usability: `streamlit run streamlit_app.py`
must render the landing page on a clean checkout with no API key and no
parquet files, falling back transparently to the two bundled CSVs that ship in
the repo. Three user stories prioritize the work — landing page (P1), FMP
data refresh (P2), metrics extension with test coverage (P3) — and each is
independently shippable.

## Technical Context

**Language/Version**: Python 3.11 (pinned in the future Dockerfile; constitution-mandated)

**Primary Dependencies**:

- Production runtime (`requirements.txt`):
  - `streamlit` (latest) — UI framework, multi-page app
  - `torch` from `https://download.pytorch.org/whl/cpu` — CPU-only PyTorch
  - `pandas`, `pyarrow` — parquet I/O and DataFrames
  - `numpy` — numeric primitives (metrics module dependency)
  - `scikit-learn` — `ShrunkCovariance` (Phase 0 imports it for symmetry with later phases; not actively used until Phase 2)
  - `plotly` — interactive charts (landing page uses none; Phase 1+ consumes)
  - `requests` — FMP HTTP client (used only by `scripts/fetch_data.py`)
  - `python-dotenv` — local `.env` loading for `FMP_API_KEY`
- Training runtime (`requirements-train.txt`):
  - `torch` (default GPU index) — baked into the Modal container image via
    `modal.Image.pip_install_from_requirements("requirements-train.txt")`
    when later phases land their training scripts.
  - `modal` itself is installed locally (developer side) via `pip install modal`
    so `modal run` works from VS Code; not pinned into `requirements.txt`
    because the live HF app does not call Modal.
  - Phase 0 seeds the GPU torch line; later phases extend with training-only
    deps (wandb, etc., not in Phase 0).
- Test dependencies (kept inline in `requirements.txt` so `pytest` runs on the prod environment):
  - `pytest` — test runner
  - `pytest-cov` (optional) — coverage reporting

**Storage**: parquet files in the directory pointed to by
`DEEP_FINANCE_DATA_DIR` (default `./data/`; the developer typically sets it
to `~/data_lake/deep-finance/` to co-locate with their existing data lake at
`/data_lake/data_bento/cme/`) — produced by `scripts/fetch_data.py` and
consumed by `src/data.py`. Bundled CSV fallbacks at the fixed repo-relative
paths `data/aapl.csv` and `data/portfolio_data.csv` (these do NOT honour the
env var, so the override does not break fresh-clone usability). No database,
no cache server, no `.pt` checkpoints in Phase 0 (those land in Phases 1–3
via Modal — see constitution v1.1.0 §"Training workflow (Modal)"; a Modal
Volume `deep-finance-data` holds intermediate training state, the developer
pulls finals via `modal volume get` and commits to `data/pretrained/`).
Publish step: a `cp $DEEP_FINANCE_DATA_DIR/*.parquet ./data/` + commit ships
the developer's local parquets to HF Space (documented in `quickstart.md`).

**Testing**:

- Unit tests: `pytest` against `tests/unit/test_metrics.py`. Fixture is a
  fixed-seed `np.random.normal(0.0005, 0.01, 1000)` returns series; per-key
  formula cross-checks; regression assertions on the three original keys.
- Integration tests: `pytest` against `tests/integration/test_landing_page.py`
  using `streamlit.testing.v1.AppTest`. Parametrized across four
  `(parquet_present, fmp_key_set)` combinations using tmp_path fixtures to
  toggle parquet presence and monkeypatch to toggle the env var.

**Target Platform**: Linux container, Hugging Face Space CPU Basic (2 vCPU,
16 GB RAM). Same binaries run locally on macOS / WSL for development. Python
3.11 across all environments.

**Project Type**: Single project. Streamlit is the whole app — there is no
backend service to split out. The repo layout is dictated by the brief's §5
(authoritative; this plan does not propose alternatives).

**Performance Goals**:

- Landing page warm render: < 3 s (target inherited from constitution).
- Cold-start render: < 30 s (per SC-001).
- CSV-fallback fail-soft on missing parquet: silent and inside one re-render.
- `scripts/fetch_data.py` wall-clock for full three-universe pull on the
  developer's machine: best-effort; brief gives no SLA. Per-universe summary
  line at exit is required (FR-010, US2 acceptance scenario 1).

**Constraints**:

- HF CPU Basic: 16 GB RAM ceiling. Bundled CSVs and parquets must all fit
  comfortably under that ceiling at load time; the 100-stock universe at
  Premium FMP density (~25 years × ~252 trading days × 100 names × ~6 columns
  in float64) is roughly 30 MB raw — well within budget.
- No CUDA/GPU operations in `src/` (Constitution Principle II).
- No live training, no `optimizer.step()` reachable without user opt-in
  (Principle III) — non-issue in Phase 0 since there is no model training.
- `Neg_Sharpe` and `SharpeLoss` from the canonical `src/losses.py` MUST NOT be
  edited (Principle V); they ride along unchanged from `Utilis/`.

**Scale/Scope**:

- Five universe definitions; three parquet outputs from `scripts/fetch_data.py`.
- One landing page + three placeholder pages (page bodies land in Phases 1–3).
- One unit-test module (~10–15 assertions covering 9 metric keys + regression).
- One integration-test module (~4 parameterized test cases).
- Files touched: ~25 (per the §5 layout in the brief).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Evaluated against the six principles in `.specify/memory/constitution.md`
v1.0.0.

| Principle | Status | Notes |
|---|---|---|
| I. Paper-Faithful Replication | PASS | Phase 0 ships no paper content; per-paper page bodies are explicitly out of scope. Plan ensures placeholder pages do not pre-empt later-phase decisions (no premature math copy on the stubs; arXiv links only). |
| II. Device-Agnostic Torch (NON-NEGOTIABLE) | PASS | The only torch-touching files in Phase 0 are the canonical `losses.py` and `torch_data.py`, both already device-agnostic (no `.cuda()` calls; tensors created from numpy via `torch.from_numpy(...)`, which inherits CPU device by default). No new model code in Phase 0. |
| III. Two-Compute-Environment Discipline | PASS | Phase 0 establishes the `requirements.txt` (CPU torch, baked into the HF Space Docker image) / `requirements-train.txt` (GPU torch, baked into the Modal container image in Phases 1+) split that enforces this discipline downstream. No `optimizer.step()` anywhere in Phase 0 code; no Modal-decorated functions in Phase 0 (the training scripts land in Phases 1–3). |
| IV. Data-as-Artifact (parquet-not-DB) | PASS | `scripts/fetch_data.py` writes parquet under `data/`; loader reads parquet under `src/data.py`; bundled CSVs are the canonical fallback. No database, no API backend. |
| V. Pre-existing Canon | PASS | Four canonical files (`losses.py`, `early_stopper.py`, `torch_data.py`, `metrics.py`) copied verbatim from `Utilis/` to `src/`. The single allowed extension is `metrics.py` per brief §13.7, which preserves all original keys byte-for-byte (verified by a regression test under US3). |
| VI. Test Critical Paths | PASS | Unit tests for the new metric keys (per FR-020); integration test for landing-page render across four data states (per FR-021); regression test on the three original metric keys; no UI snapshot tests, no browser-driven tests. Matches the constitution's "test critical paths, not everywhere" mandate. |

**No violations. No Complexity Tracking entries required.**

The re-check after Phase 1 design (data-model + contracts) is recorded at the
end of this plan; if any design choice introduces friction with the
principles, it MUST be documented in the Complexity Tracking table below
with explicit justification.

## Project Structure

### Documentation (this feature)

```text
specs/001-phase-0-skeleton-data/
├── plan.md              # This file (/speckit-plan command output)
├── spec.md              # /speckit-specify output (already exists)
├── research.md          # Phase 0 output — research and stack-decisions consolidation
├── data-model.md        # Phase 1 output — Universe, DataSnapshot, MetricsReport entities
├── quickstart.md        # Phase 1 output — operator walkthrough for US1/US2/US3
├── contracts/           # Phase 1 output — interface contracts
│   ├── fetch_data_cli.md            # scripts/fetch_data.py contract
│   ├── data_loader_api.md           # src/data.py contract
│   ├── metrics_report_schema.md     # src/metrics.py report_metrics() return shape
│   └── landing_page_sidebar.md      # streamlit_app.py sidebar contract
├── checklists/
│   └── requirements.md   # /speckit-specify validation checklist (already exists)
└── tasks.md             # Phase 2 output (/speckit-tasks command — NOT created by /speckit-plan)
```

### Source Code (repository root)

The repository layout is authoritative per `Project_brief.md` §5. Phase 0
creates only the subset listed below; later phases fill in the remainder.

```text
deep-finance-showcase/
├── streamlit_app.py              # NEW — Phase 0: landing page (Page 0)
├── pages/                        # NEW — Phase 0: empty stubs (page bodies are Phases 1–3)
│   ├── 1_📈_Momentum.py          #   stub with "Coming in Phase 1" notice + arXiv link
│   ├── 2_💼_Portfolio.py         #   stub with "Coming in Phase 2" notice + arXiv link
│   └── 3_📖_Order_Book.py        #   stub with "Coming in Phase 3" notice + arXiv link
├── src/                          # NEW — Phase 0
│   ├── __init__.py               #   copied verbatim from Utilis/__init__.py
│   ├── losses.py                 #   copied verbatim from Utilis/loss.py (rename loss.py → losses.py per brief §13.0)
│   ├── early_stopper.py          #   copied verbatim from Utilis/early_stopper.py
│   ├── torch_data.py             #   copied verbatim from Utilis/torch_data.py
│   ├── metrics.py                #   EXTENDED per brief §13.7 (6 new keys, original 3 unchanged)
│   ├── data.py                   #   parquet-first loader + FMP HTTP client + CSV fallback
│   └── universes.py              #   five named universe definitions (ETF, 20-stock, 100-stock, CME, AAPL)
├── data/                         # NEW — Phase 0
│   ├── aapl.csv                  #   relocated from existing Data/aapl.csv
│   ├── portfolio_data.csv        #   relocated from existing Data/sample_portfolio_data.csv (rename)
│   ├── etf_basket.parquet        #   produced by scripts/fetch_data.py (committed once fetched)
│   ├── sp500_20.parquet          #   produced by scripts/fetch_data.py (committed once fetched)
│   └── sp500_100.parquet         #   produced by scripts/fetch_data.py AFTER OD-2 ticker confirmation
├── scripts/                      # NEW — Phase 0
│   └── fetch_data.py             #   one-shot FMP pull → three parquets
├── notebooks/                    # NEW — Phase 0 creates the directories with .gitkeep
│   └── reference/
│       └── data/
│           └── limit_order_book_data.csv    #   relocated from Data/sample_limit_order_book_data.csv (rename)
│                                  # Note: no notebooks/colab/ directory — Phases 1–3 use Modal (src/training/train_*.py with @app.local_entrypoint), not notebooks.
│                                  # See constitution v1.1.0 §"Training workflow (Modal)".
├── papers/                       # NEW — Phase 0 creates the directory with .gitkeep (PDFs added later)
├── tests/                        # NEW — Phase 0
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   └── test_metrics.py       #   covers 6 new keys + 3-key regression guard
│   └── integration/
│       ├── __init__.py
│       └── test_landing_page.py  #   AppTest × {parquet, fmp_key} 2×2 matrix
├── .github/                      # NEW — Phase 0 creates the directories with .gitkeep
│   └── workflows/                #   empty in Phase 0; Phase 4 adds sync-to-hf.yml
├── .streamlit/                   # NEW — Phase 0
│   ├── config.toml               #   port 7860, address 0.0.0.0, headless, usage-stats off
│   └── secrets.toml.example      #   FMP_API_KEY placeholder
├── .env.example                  # NEW — Phase 0: FMP_API_KEY=your_key_here
├── .gitignore                    # NEW — Phase 0: excludes .env, __pycache__/, .pt files in dev
├── .dockerignore                 # NEW — Phase 0: scaffold; full body in Phase 4
├── Dockerfile                    # NEW — Phase 0: minimal stub; full body in Phase 4
├── requirements.txt              # NEW — Phase 0: CPU torch + Streamlit + data deps + pytest
├── requirements-train.txt        # NEW — Phase 0: GPU torch seed (grows in later phases)
├── packages.txt                  # NEW — Phase 0: empty (no apt deps needed)
├── LICENSE                       # NEW — Phase 0: MIT (per brief assumption; user confirmed default)
└── README.md                     # NEW — Phase 0: minimal README; full body in Phase 4
```

**Structure Decision**: Single-project Streamlit app. The layout is fully
dictated by `Project_brief.md` §5; this plan inherits it verbatim. No
backend/frontend split, no monorepo, no library extraction — Streamlit is the
whole app and the source tree is read-mostly with one `src/` package, one
`scripts/` directory of one-shot CLIs, one `data/` directory of committed
artifacts, and one `tests/` directory. Files marked `NEW` ship in Phase 0;
all other files in §5 of the brief are created by later phases.

## Post-Design Constitution Re-Check

After producing `research.md`, `data-model.md`, `contracts/`, and
`quickstart.md`, re-evaluating the six principles:

| Principle | Status | Notes |
|---|---|---|
| I. Paper-Faithful Replication | PASS | Design surfaces no paper content. |
| II. Device-Agnostic Torch | PASS | Design surfaces no new torch code. |
| III. Two-Compute-Environment Discipline | PASS | `contracts/fetch_data_cli.md` documents CPU-only execution; `requirements-train.txt` boundary preserved. |
| IV. Data-as-Artifact | PASS | `data-model.md` entities map to parquet/CSV files on disk; no DB tables. |
| V. Pre-existing Canon | PASS | `contracts/metrics_report_schema.md` preserves the three original keys and names them as backward-compat invariants. |
| VI. Test Critical Paths | PASS | `quickstart.md` walkthrough exercises both pytest invocations (unit + integration); no UI snapshot or browser tests introduced. |

Re-check passes. No Complexity Tracking entries.

## Complexity Tracking

No Constitution Check violations in either gate. This table is intentionally
empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| _(none)_  | _(n/a)_    | _(n/a)_                             |
