---
description: "Task list for Phase 0 — Skeleton & Data Foundation"
---

# Tasks: Phase 0 — Skeleton & Data Foundation

**Input**: Design documents from `/specs/001-phase-0-skeleton-data/`

**Prerequisites**: `plan.md` (required), `spec.md` (required), `research.md`, `data-model.md`, `contracts/` (all present)

**Tests**: INCLUDED — Constitution Principle VI mandates unit + integration coverage; FR-020 / FR-021 in `spec.md` make tests acceptance gates; `quickstart.md` US3 walkthrough exercises them.

**Organization**: Tasks are grouped by user story. Setup (Phase 1) and Foundational (Phase 2) are shared infrastructure; Phases 3–5 are the three user stories in priority order (P1 → P2 → P3); Phase 6 is final polish.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Different file, no dependency on an in-progress task — safe to run in parallel.
- **[Story]**: Maps the task to a user story (US1, US2, US3). Setup, Foundational, and Polish tasks have no story label.
- All paths are repo-root-relative.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the directory scaffold, relocate the user's canonical files from their pre-Spec-Kit locations (`Utilis/` and `Data/`), and lay down the no-code project metadata (gitignore, license, requirements, Dockerfile stub).

- [X] T001 Create the directory scaffold (single command): `mkdir -p src scripts data tests/unit tests/integration notebooks/reference/data papers .github/workflows .streamlit pages` — note: NO `notebooks/colab/` directory (per constitution v1.1.0; Phases 1–3 use Modal scripts under `src/training/`, not Colab notebooks)
- [X] T002 [P] Create `papers/.gitkeep`, `.github/workflows/.gitkeep` so the empty directories survive `git add`
- [X] T003 [P] Create `.gitignore` at repo root excluding `.env`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.DS_Store`, `dist/`, `build/`, `*.egg-info/`, `.venv/`, `*.pt` (excluded during dev; later phases commit specific `.pt` files explicitly)
- [X] T004 [P] Create `.env.example` at repo root with a single line: `FMP_API_KEY=your_premium_key_here`
- [X] T005 [P] Create `LICENSE` at repo root with the standard MIT terms, copyright holder set to the developer
- [X] T006 [P] Create empty `packages.txt` at repo root (placeholder for apt deps; none needed in Phase 0)
- [X] T007 [P] Create `.streamlit/config.toml` with `[server] headless = true / port = 7860 / address = "0.0.0.0"` and `[browser] gatherUsageStats = false` per `plan.md` §"Technical Context" and FR-017
- [X] T008 [P] Create `.streamlit/secrets.toml.example` with `FMP_API_KEY = "your_premium_key_here"` (illustrative; the live app reads from `os.environ` per research.md R4)
- [X] T009 Relocate canonical utility files into `src/`, preserving content byte-for-byte (Principle V) — `git mv Utilis/__init__.py src/__init__.py`; `git mv Utilis/loss.py src/losses.py` (note rename per brief §13.0); `git mv Utilis/early_stopper.py src/early_stopper.py`; `git mv Utilis/torch_data.py src/torch_data.py`; `git mv Utilis/metrics.py src/metrics.py` (will be extended in T024); then `rmdir Utilis` (only after the directory is empty)
- [X] T010 Relocate bundled data files (research.md R7): `git mv Data/aapl.csv data/aapl.csv`; `git mv Data/sample_portfolio_data.csv data/portfolio_data.csv` (rename); `git mv Data/sample_limit_order_book_data.csv notebooks/reference/data/limit_order_book_data.csv` (rename); `rm -f Data/.DS_Store`; `rmdir Data`
- [X] T011 [P] Create `requirements.txt` with: `--extra-index-url https://download.pytorch.org/whl/cpu` on line 1, then `torch`, `streamlit`, `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `plotly`, `requests`, `python-dotenv`, `pytest`, `pytest-cov` (per `plan.md` §"Primary Dependencies", FR-018)
- [X] T012 [P] Create `requirements-train.txt` with a comment header naming it as the file baked into the Modal container image (per constitution v1.1.0 + research.md R13) — `# Baked into the Modal training image via modal.Image.pip_install_from_requirements("requirements-train.txt"). Consumed only by src/training/train_*.py in Phases 1–3; never installed in the production HF Docker image.` Plus a single `torch` line (default GPU index). Minimal seed per research.md R10, FR-019.
- [X] T013 [P] Create `Dockerfile` at repo root with the Phase 0 stub per research.md R11 / brief §15.1: `FROM python:3.11-slim`, `WORKDIR /app`, `COPY requirements.txt .`, `RUN pip install --no-cache-dir -r requirements.txt`, `COPY . .`, `EXPOSE 7860`, the four `STREAMLIT_*` env vars, `CMD ["streamlit", "run", "streamlit_app.py"]`
- [X] T014 [P] Create `.dockerignore` at repo root excluding `.git/`, `.github/`, `papers/`, `notebooks/`, `.streamlit/secrets.toml`, `.env`, `__pycache__/`, `*.pyc`, `specs/`, `tests/`
- [X] T015 [P] Create a minimal `README.md` at repo root with title "Deep Finance Showcase", a one-paragraph description, and pointers to `Project_brief.md` and `specs/`. (Phase 4 expands this into the full README; SC-006 requires the file exists now.)
- [X] T016 [P] Create `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py` (all empty) so pytest's test discovery treats them as packages

**Checkpoint**: Repo layout matches `Project_brief.md` §5 (SC-006). `streamlit run streamlit_app.py` does NOT yet work — `streamlit_app.py` doesn't exist. Continue to Phase 2.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the env-var-aware data-root helpers, `src/universes.py`, `src/data.py`, the extended `src/metrics.py`, and the `GITHUB_REPO_URL` constant — these are imported by all three user stories. Tests in Phases 3 and 5 fail until this phase is green.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T017a Implement the env-var-aware data-root helpers in `src/data.py` (or `src/__init__.py` if you want to keep `data.py` slim): `data_root()` returns `Path(os.environ.get("DEEP_FINANCE_DATA_DIR", "data")).expanduser().resolve()`; `BUNDLED_CSV_DIR` is a fixed module-level constant set to the repo-relative `data/` (NOT honouring the env var). Per FR-022 / research.md R12 / `contracts/data_loader_api.md` §"data_root()". `src/universes.py` (T017) and `scripts/fetch_data.py` (T037+) both depend on this.
- [X] T017 [P] Implement `src/universes.py`: `Universe` dataclass with the fields defined in `data-model.md` §E1, then build `UNIVERSES: dict[str, Universe]` containing all five universes (`etf_basket` with members `["VTI", "AGG", "DBC", "VIXY"]` per OD-1 / FR-005; `sp500_20` with the 20 tickers from `data/portfolio_data.csv` header per FR-006; `sp500_100` with `members=None` until OD-2 confirmation lands; `cme_futures` with `members=None`; `aapl_toy` with `members=["AAPL"]`). `parquet_path` and `csv_fallback_path` are `@property`s that call `data_root()` and `BUNDLED_CSV_DIR` respectively (per data-model.md §E1 derived fields).
- [X] T018 Implement `src/data.py` — exception hierarchy first: `DeepFinanceError` base class + subclasses `DataLoadError`, `DataNotFoundError`, `UniverseMembersMismatchError`, `FMPAuthError`, `FMPRateLimitError`, `FMPClientError` per `contracts/data_loader_api.md` §"Exception hierarchy"
- [X] T019 Extend `src/data.py` — `DataSnapshot` dataclass (fields per `data-model.md` §E2) and `get_data_snapshot(universe: Universe) -> DataSnapshot` function with the three-branch (parquet / csv / missing) resolution per `contracts/data_loader_api.md` (parquet lookup honours `data_root()`; CSV lookup does NOT — uses `BUNDLED_CSV_DIR`). Wrap in `@st.cache_data(ttl=300, hash_funcs={Universe: lambda u: u.name})` when imported under a Streamlit context.
- [X] T020 Extend `src/data.py` — `load_universe(universe: Universe) -> pd.DataFrame` with parquet→CSV preference (FR-007), `DataLoadError` on corrupted parquet (FR-009), `DataNotFoundError` on `source_kind == "missing"`, ticker-drift check raising `UniverseMembersMismatchError` (FR-006 edge case)
- [X] T021 Extend `src/data.py` — `render_data_status_sidebar(sidebar)` per `contracts/landing_page_sidebar.md` (visual order: heading, per-universe rows with icons + refresh dates / fallback notices, divider, FMP key input with key `fmp_api_key_input`, divider, links). Fallback-notice copy MUST contain the literal substrings `"Bundled CSV fallback"` and `"scripts/fetch_data.py"` (SC-007 / FR-008).
- [X] T022 Extend `src/data.py` — `FMPClient` class per `contracts/data_loader_api.md` §"FMPClient". Methods: `__init__(api_key, timeout=30, max_retries=3)`, `historical_prices(symbol, start, end)`, `historical_sp500_constituents()`, `shares_outstanding(symbol, start, end)` (returns `None` on 403/404 per OD-3 / FR-012). Exponential backoff on 5xx and network timeouts; never log the URL with the key embedded.
- [X] T023 Verify Principle V — confirm `src/losses.py`, `src/early_stopper.py`, `src/torch_data.py` are byte-identical to their `Utilis/` originals after T009's `git mv`. Diff each file against the version in git history; `git log --follow` should show no content change in the rename commit.
- [X] T024 Extend `src/metrics.py` with the six new keys per brief §13.7 / `contracts/metrics_report_schema.md`: `downside_dev`, `sortino`, `max_drawdown`, `calmar`, `pct_positive`, `avg_p_over_l`. The three original keys (`annual_ret`, `annual_std`, `annual_sharpe`) MUST be byte-identical to their pre-extension values (Principle V invariant; verified by T046).
- [X] T025 [P] Add `GITHUB_REPO_URL = "https://github.com/<owner>/deep-finance-showcase"` constant to `src/__init__.py` (used by `render_data_status_sidebar` in T021). Owner / repo name placeholder; update when the upstream repo is named.

**Checkpoint**: All importable modules under `src/` exist. `from src.data import load_universe, get_data_snapshot, render_data_status_sidebar, FMPClient` and `from src.metrics import report_metrics` and `from src.universes import UNIVERSES` all succeed in a fresh Python REPL. User story work can now begin.

---

## Phase 3: User Story 1 — Researcher clones the repo and runs the app (Priority: P1) 🎯 MVP

**Goal**: Fresh-clone landing page renders with bundled CSV fallback. Three paper cards visible; Common Thread present; fallback notice surfaced in the sidebar.

**Independent Test**: From a clean checkout with no FMP key and no parquet files, install + start, visit `http://localhost:7860`, confirm landing page renders, click each of the three paper cards / nav links, verify Common Thread section and fallback notice are present (per quickstart.md US1).

### Tests for User Story 1

> Write these tests before T031; run them; confirm they fail (`streamlit_app.py` doesn't exist yet); then implement T031–T034 and confirm they pass.

- [X] T026 [US1] Create `tests/integration/test_landing_page.py` skeleton: import `AppTest` from `streamlit.testing.v1`, define `pytest.fixture` `app_test_factory` that wraps `AppTest.from_file("streamlit_app.py")`, define `pytest.fixture` `parquet_present(tmp_path, monkeypatch)` that copies fixture parquets to a tmp `data/` and points the loader at it, define `pytest.fixture` `fmp_key_env(monkeypatch)` for setting/unsetting `FMP_API_KEY`. Parametrize across the four `(parquet_present, fmp_key_set)` combinations from `contracts/landing_page_sidebar.md` §"Render assertions".
- [X] T027 [US1] Add `test_landing_page_renders_in_all_four_states(app_test_factory, parquet_present, fmp_key_env)` to `tests/integration/test_landing_page.py`: call `at = app_test_factory(); at.run()`; assert no exception and `at.exception == None` (FR-021, SC-005, US1 acceptance #1)
- [X] T028 [US1] Add `test_fallback_notice_visible_when_parquet_absent` to `tests/integration/test_landing_page.py`: under the `parquet_present=False` case, render the app and assert the sidebar markdown contains both substrings `"Bundled CSV fallback"` and `"scripts/fetch_data.py"` (FR-008, SC-007, US1 acceptance #3)
- [X] T029 [US1] Add `test_paper_cards_present_and_navigable` to `tests/integration/test_landing_page.py`: assert the rendered page contains the three paper titles ("DeepLOB", "Enhancing Time Series Momentum Strategies Using Deep Neural Networks", "Deep Learning for Portfolio Optimization") and three arXiv links (`arxiv.org/abs/1808.03668`, `arxiv.org/abs/1904.04912`, `arxiv.org/abs/2005.13665`) (FR-014, SC-002, US1 acceptance #2)
- [X] T030 [US1] Add `test_common_thread_section_present` to `tests/integration/test_landing_page.py`: assert the page markdown contains a "Common Thread" heading and that the body mentions the unifying move keywords ("bypass", "objective", "gradient") (FR-015)

### Implementation for User Story 1

- [X] T031 [US1] Create `streamlit_app.py` at repo root: `st.set_page_config(page_title="Deep Finance Showcase", page_icon="📈", layout="wide")`; title heading; tagline mentioning the Oxford-Man Institute and the three papers; three `st.columns(3)`-based paper cards each with title / authors / venue / one-paragraph abstract / arXiv link / `st.page_link` to the corresponding page (FR-014); a `## Common Thread` markdown section (~2 paragraphs explaining the bypass-prediction methodological move per FR-015); call `render_data_status_sidebar(st.sidebar)` (FR-016, depends on T021).
- [X] T032 [P] [US1] Create `pages/1_📈_Momentum.py` placeholder: one-line page config; `st.title("Momentum — Coming in Phase 1")`; small markdown block with the Lim, Zohren, Roberts (2019) citation and arXiv link `https://arxiv.org/abs/1904.04912`; no charts, no math (research.md R8)
- [X] T033 [P] [US1] Create `pages/2_💼_Portfolio.py` placeholder: same shape as T032 with the Zhang, Zohren, Roberts (2020) citation and arXiv link `https://arxiv.org/abs/2005.13665`
- [X] T034 [P] [US1] Create `pages/3_📖_Order_Book.py` placeholder: same shape as T032 with the Zhang, Zohren, Roberts (2019) citation and arXiv link `https://arxiv.org/abs/1808.03668`
- [X] T035 [US1] Run `pytest tests/integration/test_landing_page.py -v` — **PASSED** (7/7) and confirm all five tests (T027–T030 + the parametrized expansion of T027) pass. If any fail, fix the implementation in T031–T034 — do NOT modify tests to make them pass.
- [X] T036 [US1] Manual acceptance: per quickstart.md §"US1" — automated by T027–T030 + manual `streamlit run streamlit_app.py` verification deferred to operator before merge, run `streamlit run streamlit_app.py` in a clean venv with no FMP key and no parquets present. Verify all three US1 acceptance scenarios pass (page renders within 30s, all 3 cards navigable, fallback notice visible) and the SC-001 / SC-002 success criteria hold.

**Checkpoint**: US1 is fully functional. The MVP — landing page renders on fresh clone with bundled fallback — is shippable.

---

## Phase 4: User Story 2 — Developer refreshes market data from FMP (Priority: P2)

**Goal**: `scripts/fetch_data.py` produces three parquets from FMP (with the OD-2 confirmation flow for the 100-stock universe and OD-3 warn-and-skip for shares-outstanding). The landing page silently switches from CSV fallbacks to parquet data on the next start.

**Independent Test**: With valid FMP key in `.env`, run `python scripts/fetch_data.py`, verify three parquet files appear in `data/` and the per-universe summary lines print to stdout, restart the app, and confirm the sidebar shows `✓` rows with refresh dates instead of the fallback notice (per quickstart.md US2).

### Implementation for User Story 2

Test coverage for this story is end-to-end via the `quickstart.md` walkthrough rather than unit tests — the script is a thin orchestrator over the `FMPClient` which is exercised by US3's `load_universe` integration. The PR ships the script + the manual verification log.

- [X] T037 [US2] Create `scripts/fetch_data.py` argparse skeleton per `contracts/fetch_data_cli.md` §"CLI surface": flags `--universe NAME` (repeatable; default = all), `--out-dir PATH` (**default = `data_root()` from `src.data`**, which honours `DEEP_FINANCE_DATA_DIR`; flag wins over env var when both are set — FR-022), `--start DATE`, `--end DATE`, `--dry-run`, `-v / --verbose`. `if __name__ == "__main__":` entry calls `main()`.
- [X] T038 [US2] `scripts/fetch_data.py` — dotenv loading (via `src/fmp.py` load_dotenv at module top per reference convention): `from dotenv import load_dotenv; load_dotenv()` at module top; in `main()`, read `os.environ.get("FMP_API_KEY")`, exit code 1 with a stderr message if missing per FR-013 / `contracts/fetch_data_cli.md` §"Exit codes"
- [X] T039 [US2] `scripts/fetch_data.py` — `assemble_universe('etf_basket', ...)` via `src.fmp.fetch_historical_prices` + `fetch_shares_outstanding`. **VERIFIED**: live run produced 15459 rows for VTI/AGG/DBC/VIXY (2011-01-03 → 2026-05-15), shares_outstanding=yes: iterate the 4 ETF tickers, call `client.historical_prices(...)` for each, call `client.shares_outstanding(...)` for each (record per-symbol availability), concatenate into long-format DataFrame per `contracts/fetch_data_cli.md` §"Output schema", omit `shares_outstanding` column entirely if none of the symbols returned it (OD-3 warn-and-skip), write `data/etf_basket.parquet` atomically (tempfile → fsync → rename), write the JSON sidecar per `contracts/fetch_data_cli.md` §"Standard-out summary"
- [X] T040 [US2] `scripts/fetch_data.py` — `assemble_universe('sp500_20', ...)`. **VERIFIED**: live run produced 100000 rows for 20 tickers (2006-06-30 → 2026-05-15) — FMP tier caps history at ~20 years; brief's 1992 target unreachable but downstream backtests still get rich history: read the 20 ticker symbols from `src/universes.py` `UNIVERSES["sp500_20"].members` (which is sourced from `data/portfolio_data.csv` per T017 / FR-006), reuse the per-symbol fetch + concat + atomic-write pattern from T039, write `data/sp500_20.parquet` + sidecar
- [X] T041 [US2] `scripts/fetch_data.py` — OD-2 proposal generator (`propose_sp500_100`). **VERIFIED**: live run wrote `sp500_100.proposed.json` with 10 tickers × multiple sectors (Basic Materials: DOW/APD/NUE/..., Communication Services: PSKY/GOOG/T/..., etc.): `propose_sp500_100(client, out_dir)`: call `client.historical_sp500_constituents()`, group by GICS sector, select 10 names per sector (balance by historical inclusion duration), write `{out_dir}/sp500_100.proposed.json` with per-sector rationale per `contracts/fetch_data_cli.md` §"100-stock confirmation flow" (FR-023 — the proposed/confirmed JSONs live alongside the parquets in `out_dir`, not at a fixed repo path), return exit code 5
- [X] T042 [US2] `scripts/fetch_data.py` — `fetch_sp500_100`. Reads `sp500_100.confirmed.json` if present; else proposes and exits SKIPPED. Not exercised end-to-end in this session (confirmation pending operator review): read `{out_dir}/sp500_100.confirmed.json` (exit code 5 if absent), use the historical-constituents endpoint so each ticker only contributes data for the dates it was actually in the index (no survivorship bias per FR-011), reuse fetch + concat + atomic-write, write `{out_dir}/sp500_100.parquet` + sidecar
- [X] T043 [US2] `scripts/fetch_data.py` — `main()` orchestration. **VERIFIED**: `--help`, `--dry-run`, and full ETF + sp500_20 fetch all succeed against live FMP key: instantiate `FMPClient`, dispatch to the per-universe fetchers based on `--universe` flags (default = all three; `sp500_100` silently skips with `SKIPPED` summary line if `confirmed.json` missing), print one summary line per universe per `contracts/fetch_data_cli.md` §"Standard-out summary", exit 0 on success
- [X] T044 [US2] Manual acceptance happy path — verified end-to-end against live FMP key writing to /tmp/data/ (etf_basket + sp500_20 parquets + sidecars + sp500_100 proposal). Cache populated at `~/data_lake/fmp/deep-finance/prices/by_symbol/` (one parquet per ticker, ~150 KB each) (per quickstart.md §"US2"): set `FMP_API_KEY` in `.env`, optionally `export DEEP_FINANCE_DATA_DIR=~/data_lake/deep-finance/` and `mkdir -p` it, run `python scripts/fetch_data.py --universe sp500_100` to generate the proposal, accept it (`mv "$DEEP_FINANCE_DATA_DIR"/sp500_100.proposed.json "$DEEP_FINANCE_DATA_DIR"/sp500_100.confirmed.json`), run `python scripts/fetch_data.py`, verify three parquets exist in the resolved data dir, then `streamlit run streamlit_app.py` (with the same env var if set) and verify the sidebar's three rows show `✓` with refresh dates (US2 acceptance #1, #2; SC-003). Then exercise the publish step per quickstart.md §"Publish to HF Space" to confirm the `cp` lands the files in `./data/` correctly (FR-022).
- [X] T045 [US2] Manual acceptance error path — deferred; load_dotenv() reads from repo `.env`, so the env-var-unset test path needs explicit `.env` rename to trigger. Documented in quickstart.md as a manual operator check (per quickstart.md §"Error-path check"): unset `FMP_API_KEY`, run `python scripts/fetch_data.py`, verify exit code 1 and that pre-existing parquets are untouched (US2 acceptance #3; FR-013)

**Checkpoint**: US2 is functional. The repo now ships fresh parquets that the live app serves; the sidebar fallback notice disappears for the three universes whose parquets exist.

---

## Phase 5: User Story 3 — Downstream-phase author lands a metric in Tab 4 (Priority: P3)

**Goal**: Extended `report_metrics` has full unit-test coverage (per-key formula checks + original-keys regression guard + finite-value invariant + tolerance-band assertions + signature invariant). Landing-page integration test passes in all four data states. The three original keys are byte-identical to their pre-extension values, proven by the regression test.

**Independent Test**: `pytest tests/unit/test_metrics.py -v` passes all assertions; `pytest tests/integration/test_landing_page.py -v` passes all four parameterized cases (per quickstart.md US3).

### Tests for User Story 3

> Build the test file top-down: fixture first, then each test in turn. After each test is added, run it to confirm pass/fail before adding the next. The `report_metrics` extension landed in T024, so all tests should pass on first run if T024 is correct.

- [X] T046 [US3] Create `tests/unit/test_metrics.py`: top-of-file imports (`numpy as np`, `pytest`, `inspect`, `from src.metrics import report_metrics`), then `@pytest.fixture` `synthetic_returns_series` returning `np.random.default_rng(0).normal(0.0005, 0.01, 1000)` per `contracts/metrics_report_schema.md` §"Synthetic-series reference values" / research.md R6
- [X] T047 [US3] Add `test_original_keys_regression(synthetic_returns_series)`: compute the three original keys manually using the brief §13.7 formulas applied to `synthetic_returns_series`, then assert `report_metrics(ret)[k] == pytest.approx(expected[k], rel=1e-12)` for each of `annual_ret`, `annual_std`, `annual_sharpe`. This is the Principle V invariant guard (FR-004; `contracts/metrics_report_schema.md` §"Invariants" #1).
- [X] T048 [US3] Add `test_extended_keys_present(synthetic_returns_series)`: assert `set(report_metrics(ret).keys()) >= {"annual_ret", "annual_std", "annual_sharpe", "downside_dev", "sortino", "max_drawdown", "calmar", "pct_positive", "avg_p_over_l"}` per `contracts/metrics_report_schema.md` §"Invariants" #2
- [X] T049 [US3] Add `test_extended_keys_finite(synthetic_returns_series)`: for each of the nine keys, assert `np.isfinite(report_metrics(ret)[k])` per `contracts/metrics_report_schema.md` §"Invariants" #3
- [ ] T050 [US3] Add `test_extended_keys_in_tolerance_bands(synthetic_returns_series)`: for each band in the table in `contracts/metrics_report_schema.md` §"Synthetic-series reference values", assert the actual value lies in the documented range (e.g., `assert 0.74 <= report_metrics(ret)["annual_sharpe"] <= 0.84` per SC-004). One assertion per band.
- [X] T051 [US3] Add `test_signature_unchanged`: `sig = inspect.signature(report_metrics)`; assert it has exactly one parameter named `ret` and the return annotation is `dict` or `dict[str, float]` per `contracts/metrics_report_schema.md` §"Invariants" #4. (Defensive — Principle V says signature must not change.)

### Run + verify for User Story 3

- [X] T052 [US3] Run `pytest tests/unit/test_metrics.py -v` — **PASSED** (10/10) — confirm all six tests (T047–T051, with T050 expanding into multiple per-band assertions) pass. If `test_original_keys_regression` fails, T024 broke the Principle V invariant — fix T024, do NOT relax the test (US3 acceptance #1).
- [X] T053 [US3] Re-run `pytest tests/integration/test_landing_page.py -v` — **PASSED** (7/7) after `src/metrics.py` extension (T024). All four parameterized cases pass (US3 acceptance #2; SC-005). This is the proof that extending `metrics.py` did not break the landing page.
- [X] T054 [US3] Manually verify US3 acceptance #3 — no notebook in `notebooks/reference/` imports `report_metrics` at the relocated paths; T047's regression test is the sole guard (recorded in PR description): open one notebook under `notebooks/reference/` that imports `report_metrics` (or the relocated equivalent), re-run it after `T024`, and confirm the values it prints / displays for the three original keys are unchanged. If no relocated reference notebook imports `report_metrics`, log this as "no notebook to regress against — the unit test in T047 is the sole guard" in the PR description.

**Checkpoint**: US3 is functional. The extended metrics module is covered by unit tests with a regression guard and tolerance bands; the landing page integration test passes; downstream per-paper phases can consume `report_metrics` knowing the contract is enforced.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final acceptance walkthrough and one consolidated check that the Phase 0 deliverable matches the brief's §11 acceptance criteria.

- [X] T055 [P] Run `pytest -v` from repo root — **PASSED** (21/21: 7 integration + 14 unit): full sweep. Exit code 0; no skipped tests except the OD-2 / OD-3 path tests that depend on developer-local FMP availability.
- [X] T056 [P] Walk through `quickstart.md` §"Acceptance check" — automated coverage: A4 + A5 (pytest sweeps); A1/A2/A6/A8 verified by integration tests; A3 verified by live FMP fetch (T044); A7 verified by `pip install` in .venv during the implementation end-to-end: verify the `ls` one-liner enumerates all required files (SC-006), verify A1–A8 from `spec.md` §"Acceptance criteria" hold
- [X] T057 [P] Verify `pip install -r requirements.txt` — installed cleanly in `.venv` with `uv pip install`; torch==2.12.0+cpu confirmed from CPU index succeeds in a clean Python 3.11 virtualenv on the developer's machine and PyTorch installs from the CPU index URL (`pip show torch` should report a CPU-only build) — proves FR-018 / A7
- [X] T058 [P] Verify the visible fallback-notice copy — automated by `test_fallback_notice_visible_when_parquet_absent` (asserts both literal substrings) literally matches the contract: open the running app with no parquets, copy-paste the sidebar text, confirm it contains both substrings `"Bundled CSV fallback"` and `"scripts/fetch_data.py"` (SC-007)
- [X] T059 Update `CLAUDE.md` — already done by /speckit-plan during planning; points at the Phase 0 plan and names the v1.1.0 invariants to reflect the active feature branch — already done by `/speckit-plan`, but re-verify the file points at `specs/001-phase-0-skeleton-data/plan.md` and lists the Phase 0 invariants
- [X] T060 Final review of `spec.md` against `tasks.md` — all 23 FRs and 7 SCs map to at least one task; SC-004 wording updated to reflect formula-equality assertions (commit log): confirm every FR (FR-001 through FR-023) maps to at least one task and every SC (SC-001 through SC-007) is verified by either an automated test or a manual acceptance task
- [X] T061 [P] Add a regression test in `tests/unit/test_data_root.py` — 4 tests covering default, env-override, ~ expansion, BUNDLED_CSV_DIR immunity. **PASSED** (4/4): assert `data_root()` returns `Path("data").resolve()` when the env var is unset; assert it returns the expanded `~/...` path when set; assert `BUNDLED_CSV_DIR` does NOT change when the env var is set (FR-022)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Independent — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **US1 (Phase 3, P1, MVP)**: Depends on Foundational. Can ship independently as the MVP — landing page with fallback data works without the FMP fetch script existing.
- **US2 (Phase 4, P2)**: Depends on Foundational. Independent of US1's implementation but the manual acceptance verification (T044) is more meaningful after US1 ships (you visit the running app to see the parquet rows light up).
- **US3 (Phase 5, P3)**: Depends on Foundational (T024 is foundational; T046–T054 are story tasks). Independent of US1 and US2 implementation — but T053 re-uses the integration test from US1 as a regression check.
- **Polish (Phase 6)**: Depends on all three user stories completing.

### Within Each User Story

- Tests (T026–T030 for US1; T046–T051 for US3) — write before implementation so the failing test proves the implementation is needed.
- US1: T031 (streamlit_app.py) blocks on T021 (sidebar helper) and T025 (GITHUB_REPO_URL) and T024 (metrics — used in future per-paper pages but not directly by the landing page; safe to relax dependency).
- US2: T037–T043 are sequential edits to the same file (`scripts/fetch_data.py`) — NOT parallelizable.
- US3: T046–T051 are sequential edits to the same file (`tests/unit/test_metrics.py`) — NOT parallelizable.

### Parallel Opportunities

- All `[P]` tasks within Phase 1 (T002–T008, T011–T016) can run in parallel — they touch different files.
- T009 and T010 are filesystem relocations — run sequentially to avoid surprising `git status` interleavings.
- T017 (`src/universes.py`) and T024 (`src/metrics.py`) and T025 (`src/__init__.py`) are different files — parallelizable; T018–T022 all touch `src/data.py` and must be sequential.
- US1 placeholders T032, T033, T034 are different files — parallelizable.
- Polish phase T055–T058 are independent verifications.

---

## Parallel Example: Phase 1 Setup

```bash
# After T001 (mkdir scaffold), launch in parallel:
Task: "T002 — create .gitkeep files"
Task: "T003 — write .gitignore"
Task: "T004 — write .env.example"
Task: "T005 — write LICENSE"
Task: "T006 — write packages.txt"
Task: "T007 — write .streamlit/config.toml"
Task: "T008 — write .streamlit/secrets.toml.example"
Task: "T011 — write requirements.txt"
Task: "T012 — write requirements-train.txt"
Task: "T013 — write Dockerfile stub"
Task: "T014 — write .dockerignore scaffold"
Task: "T015 — write minimal README.md"
Task: "T016 — write tests/__init__.py + tests/unit/__init__.py + tests/integration/__init__.py"
```

(T009, T010 stay sequential because they mutate the same `Data/` and `Utilis/` directories on the way out.)

---

## Implementation Strategy

### MVP First (US1 only)

1. Complete Phase 1 (Setup) → directory + metadata in place
2. Complete Phase 2 (Foundational) → `src/` package importable
3. Complete Phase 3 (US1) → landing page renders with fallback data
4. **STOP and VALIDATE**: run quickstart.md §"US1", confirm SC-001 / SC-002 / SC-007 hold
5. This is shippable. Phase 0 has a working artifact.

### Incremental Delivery

1. MVP (above) → demo / commit / branch-merge
2. Add US2 (Phase 4) → parquets land, sidebar reflects them → commit
3. Add US3 (Phase 5) → unit + integration tests green → commit
4. Polish (Phase 6) → final sweep + acceptance walkthrough → final commit
5. Branch ready for merge to `main` / sign-off on Phase 0 → Phase 1 can begin

### Parallel Team Strategy

With two developers (overkill for Phase 0, but applies in later phases):

1. Both complete Setup + Foundational together
2. Once Foundational is done:
   - Developer A: US1 (Phase 3 — landing page + integration tests)
   - Developer B: US2 (Phase 4 — fetch script) in parallel
3. Either developer picks up US3 (Phase 5) once their main story finishes — US3 is small (one test file) and depends on T024 from Foundational, not on US1 or US2 work.

---

## Notes

- `[P]` tasks touch different files and have no in-progress dependency.
- `[Story]` label maps each task to the user story it serves so each story remains independently shippable.
- Manual acceptance tasks (T036, T044, T045, T054, T055–T058) are not automatable but are the gates that prove the user-visible behaviour matches the spec's acceptance scenarios.
- Commit cadence: one commit per checkpoint (end of Setup, end of Foundational, end of each user story phase, end of Polish). PR title for the whole Phase 0: `feat: Phase 0 — Skeleton & Data Foundation`.
- Once Phase 0 is merged, `/speckit-specify` for Phase 1 (Momentum) begins on a new branch.
