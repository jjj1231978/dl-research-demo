# Feature Specification: Phase 0 — Skeleton & Data Foundation

**Feature Branch**: `001-phase-0-skeleton-data`

**Created**: 2026-05-16

**Status**: Draft

**Input**: User description (paraphrased): Deliver the repository scaffold, data
layer, landing page, and canonical utility module adoption that every later
phase (Momentum, Portfolio, Order Book, Deployment) depends on. First-clone
usability is the headline outcome: `streamlit run streamlit_app.py` must work
with no API key and no parquet files, falling back to the bundled CSVs that
ship in the repo.

## Clarifications

### Session 2026-05-16

- Q: Which ETF should stand in for the paper's non-tradable VIX in the ETF basket? → A: **VIXY** (ProShares VIX Short-Term Futures ETF). The ETF basket is therefore the 4-ticker set `{VTI, AGG, DBC, VIXY}`. Inception of VIXY (2011) bounds the basket's earliest usable date.
- Q: How should the 100-stock S&P 500 universe be constructed? → A: **Claude Code proposes a sector-balanced list of 100 historical S&P 500 constituents (10 names × 10 GICS sectors, drawn from FMP's historical constituents endpoint to avoid survivorship bias); the developer reviews and confirms the proposal before `scripts/fetch_data.py` is run.** Confirmation is a Phase-0 milestone and gates the third parquet output.
- Q: If FMP's shares-outstanding endpoint is unavailable on the developer's tier, how should `scripts/fetch_data.py` behave? → A: **Warn-and-skip**. The script logs a clear "shares-outstanding endpoint unavailable — DWP benchmark will be unavailable in Phase 2" notice, omits the `shares_outstanding` column from the parquets, and exits zero. The Phase 2 Portfolio page renders DWP as N/A with the same notice.

### Session 2026-05-17

- Q: Where should `scripts/fetch_data.py` write its parquet output, and where should the live app read parquets from? → A: **Configurable via a `DEEP_FINANCE_DATA_DIR` environment variable. Default `./data/` (so HF Space and fresh clones work with no configuration). On the developer's machine, the variable is set to `~/data_lake/deep-finance/` so fetched parquets co-locate with the developer's existing data lake (e.g., the CME raw source at `/data_lake/data_bento/cme/`). Before deploying, the developer copies the relevant parquets into `./data/` (a one-liner documented in `quickstart.md`) and commits them so HF Space serves real data.** The bundled CSV fallbacks (`data/aapl.csv`, `data/portfolio_data.csv`) remain at fixed repo-relative paths regardless of `DEEP_FINANCE_DATA_DIR`, because they ship in git and are part of first-clone usability.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Researcher clones the repo and runs the app (Priority: P1)

A researcher who arrived from one of the three papers' arXiv pages clones the
GitHub repo, follows the README quickstart, and within minutes sees a working
landing page that summarises the three papers and offers a navigation entry
into each. They have no API key, no GPU, and have never seen this project
before. The app must boot end-to-end on their laptop using only the data that
ships in the repo, and must make the data provenance unambiguous (a visible
notice tells them they are viewing fallback data and how to upgrade).

**Why this priority**: Without first-clone usability, the entire showcase
narrative is unreachable. The landing page is the artifact every later page is
navigated from; the fallback-data path is the demo that runs on every fresh
clone and in CI. It is the project's MVP — even with no per-paper pages built
yet, this story alone proves the skeleton, the data layer, and the multi-page
shell all work.

**Independent Test**: From a clean checkout with no FMP key and no parquet
files, install the production dependencies, start the Streamlit app, and
confirm the landing page renders, the three paper cards are visible and
navigable, the Common Thread section is present, and the data-status sidebar
shows a notice that bundled-CSV fallback data is in use.

**Acceptance Scenarios**:

1. **Given** a fresh clone of the repository with bundled CSVs present but no
   parquet files and no `FMP_API_KEY` environment variable, **When** the
   researcher runs the production install and starts the Streamlit app,
   **Then** the landing page renders without error within 30 seconds of the
   first request, displays the title and tagline, and shows three clickable
   paper cards.
2. **Given** the running landing page, **When** the researcher clicks any of
   the three paper cards or selects a page from the multi-page navigation,
   **Then** the corresponding placeholder page loads without error (page
   content may be a stub — that is later phases' deliverable).
3. **Given** the running landing page in fallback-data mode, **When** the
   researcher looks at the sidebar, **Then** they see a clearly worded notice
   that bundled-CSV fallback data is in use along with instructions on how to
   obtain live parquet data.

---

### User Story 2 — Developer refreshes market data from FMP (Priority: P2)

The developer has an FMP Premium subscription. They run the one-shot fetch
script locally, which pulls historical EOD prices for three universes (an ETF
basket, the existing 20-stock list, and a sector-balanced 100-stock S&P 500
selection drawn from historical constituents to avoid survivorship bias) and
writes three named parquet files into `data/`. The next time they start the
app, the landing page silently switches from fallback CSVs to the parquets and
the data-status sidebar reflects the new refresh date.

**Why this priority**: Phase 0 ends with parquets committed to the repo so the
deployed Hugging Face Space serves real data, not bundled CSVs. The fetch
script is also what the Portfolio page in Phase 2 will consume; getting it
right here de-risks the Phase 2 acceptance criteria.

**Independent Test**: With a valid FMP key in `.env`, run the fetch script and
verify three named parquets appear in `data/`. Restart the app; confirm the
data-status sidebar shows the new refresh date and no fallback notice.

**Acceptance Scenarios**:

1. **Given** a valid `FMP_API_KEY` in `.env`, **When** the developer runs the
   fetch script, **Then** three parquet files (one per universe) appear in
   `data/` and the script exits cleanly with a summary line per universe
   (universe name, row count, date range covered).
2. **Given** parquet files exist in `data/`, **When** the app is restarted,
   **Then** the landing page's data-status sidebar shows the parquet refresh
   date and the fallback notice is absent.
3. **Given** the FMP key is missing or invalid, **When** the developer runs
   the fetch script, **Then** the script exits with a clear error message
   identifying which credential is missing or rejected; no partial parquet
   files are written.

---

### User Story 3 — Downstream-phase author lands a metric in Tab 4 (Priority: P3)

An author working on a per-paper page (Momentum, Portfolio, or Order Book in
Phases 1–3) needs to populate a "Tab 4" master metrics table with values like
Sortino ratio, maximum drawdown, and Calmar ratio that are not in the
developer's starter `metrics.py`. Phase 0 extends the existing `report_metrics`
function so those values are available — and verified — before any later phase
depends on them. A unit test against a synthetic returns series proves each
new metric matches its mathematical definition; an integration test proves the
landing page still renders after the extension.

**Why this priority**: Every per-paper page in Phases 1–3 will consume these
metric values in its headline Tab 4 table. Shipping them without unit-test
backing risks landing bad numbers on the public showcase; shipping them
without preserving the existing keys risks breaking the reference notebooks
that already import them. This story locks both gates in place before any
page-builder needs the function.

**Independent Test**: Run the unit-test suite against the synthetic returns
series specified in the brief and confirm every new metric key produces a
value consistent with its formula and within the tolerance bands the brief
documents; confirm the existing keys are unchanged. Run the landing-page
integration test against `streamlit.testing.v1.AppTest` to confirm the
extended module does not break the landing-page render.

**Acceptance Scenarios**:

1. **Given** the extended `report_metrics` function and a synthetic returns
   series generated as `np.random.normal(0.0005, 0.01, 1000)` with a fixed
   seed, **When** the unit test runs, **Then** the function returns a
   dictionary containing all original keys (`annual_ret`, `annual_std`,
   `annual_sharpe`) unchanged plus all new keys (`downside_dev`, `sortino`,
   `max_drawdown`, `calmar`, `pct_positive`, `avg_p_over_l`) with values
   matching their formulas to floating-point tolerance.
2. **Given** the landing page is rendered under `AppTest` in each of the four
   combinations of {parquet present, parquet absent} × {FMP key set, FMP key
   unset}, **When** the integration test asserts no exception was raised,
   **Then** all four combinations pass.
3. **Given** a reference notebook under `notebooks/reference/` that imports
   `report_metrics` and reads the original three keys, **When** the notebook
   runs after the extension, **Then** the same numeric values are returned for
   those original keys as before the extension (regression check).

---

### Edge Cases

- **No internet during install**: `pip install -r requirements.txt` must
  succeed offline once the wheel cache exists; the production install pulls
  PyTorch from the CPU-only wheel index, not the default GPU-augmented index.
- **Parquet present but corrupted**: the loader must surface a clear error
  rather than silently re-fall-back to CSVs (otherwise a corrupted parquet
  would be invisible to the operator).
- **Bundled CSV missing or corrupted**: the loader must surface a clear error
  before the page attempts to render charts on missing data; the app should
  not crash inside a chart-rendering call.
- **FMP key set but rate-limited mid-fetch**: the fetch script must report the
  symbol that failed, leave previously-written parquet files in place, and
  exit non-zero so the operator can re-run only the missing universes.
- **Researcher mistakenly drops a non-parquet file into `data/` with a
  parquet name**: the loader must give a clear "expected parquet, got X"
  message rather than a low-level binary-format stack trace.
- **20-stock CSV ticker list drifts from the universe definition**: a
  consistency check at load time should detect the mismatch and fail loudly.

## Requirements *(mandatory)*

### Functional Requirements

**Repository scaffold**

- **FR-001**: The repository MUST contain the directory and file layout
  specified in §5 of `Project_brief.md` at the root level; every directory
  listed there MUST exist (placeholder content is acceptable for files whose
  body is delivered by later phases).
- **FR-002**: A multi-page Streamlit application MUST be installable from the
  repository root; the landing page entry point and three per-paper page
  placeholders MUST be reachable through Streamlit's native multi-page
  navigation.

**Canonical utility adoption**

- **FR-003**: The developer's four canonical files (Sharpe loss module, early
  stopper, PyTorch dataset wrapper, metrics) MUST be copied verbatim from the
  existing `Utilis/` directory into `src/` with identifier names preserved
  exactly (`Neg_Sharpe`, `SharpeLoss`, `EarlyStopping`, `MyDataset`,
  `report_metrics`).
- **FR-004**: The `report_metrics(ret)` function MUST be extended with six
  additional keys (`downside_dev`, `sortino`, `max_drawdown`, `calmar`,
  `pct_positive`, `avg_p_over_l`) per the formulas in §13.7 of the brief,
  while preserving the three original keys (`annual_ret`, `annual_std`,
  `annual_sharpe`) byte-for-byte against a fixed-seed synthetic returns series.

**Universe definitions**

- **FR-005**: The system MUST expose machine-readable definitions for five
  named universes — CME futures, toy single-asset (AAPL), ETF basket,
  20-stock, 100-stock S&P 500 — each carrying at minimum a human-readable
  name, the list of member identifiers (where statically known), the data
  source name, and which page(s) consume it. The ETF basket's member list is
  `{VTI, AGG, DBC, VIXY}` per the OD-1 clarification.
- **FR-006**: The 20-stock universe ticker list MUST be drawn directly from
  the bundled `portfolio_data.csv` header to guarantee parity with the
  existing data file.

**Data-loading layer**

- **FR-007**: The data-loading layer MUST attempt to load a universe's parquet
  file first and MUST fall back to the corresponding bundled CSV when the
  parquet is absent.
- **FR-008**: When falling back from parquet to CSV, the data-loading layer
  MUST render a visible notice in the Streamlit UI naming the universe, the
  reason for the fallback, and the operator action required to upgrade
  (running the fetch script).
- **FR-009**: When a parquet file is present but unreadable (corrupted,
  schema-mismatched, wrong file type), the loader MUST surface a clear error
  to the operator rather than silently re-falling-back to a CSV.

**FMP fetch script**

- **FR-010**: A one-shot fetch script MUST produce three named parquet files
  in `data/` (ETF basket, 20-stock, 100-stock S&P 500) using the developer's
  FMP credential read from a local environment file. The 100-stock parquet
  MUST be written only after the developer has confirmed the
  Claude-Code-proposed 100-ticker list (per OD-2); until then, the script
  MUST treat the 100-stock universe as unconfigured and skip that parquet
  with a clear "100-stock universe pending confirmation" notice.
- **FR-011**: The fetch script MUST use historical S&P 500 constituent data
  when assembling the 100-stock universe so a ticker present today but absent
  in 2010 contributes no data points to a 2010-era backtest (no survivorship
  bias).
- **FR-012**: The fetch script MUST attempt to pull each universe's per-symbol
  shares-outstanding history (needed for the Diversity-Weighted Portfolio
  benchmark in Phase 2). If the FMP tier in use does not expose this endpoint
  (per OD-3), the script MUST log a clear "shares-outstanding endpoint
  unavailable — DWP benchmark will be unavailable in Phase 2" notice, omit
  the `shares_outstanding` column from the affected parquets, and exit zero.
  The presence or absence of the `shares_outstanding` column MUST be
  detectable downstream so Phase 2 can render DWP as N/A with the same notice.
- **FR-013**: The fetch script MUST exit non-zero with a clear, operator-
  actionable error message if the FMP credential is missing or rejected; it
  MUST NOT write a partial parquet file in this case. (Note: a missing
  shares-outstanding endpoint is **not** a credential failure and MUST NOT
  trigger this path; see FR-012.)

**Landing page**

- **FR-014**: The landing page MUST display the application title, a tagline
  mentioning the Oxford-Man Institute and the three papers, and three large
  paper cards (one per paper) with the paper title, authors, year, venue,
  arXiv link, a one-paragraph abstract, and a navigation link to the
  corresponding per-paper page.
- **FR-015**: The landing page MUST include a "Common Thread" section
  (approximately two short paragraphs) explaining the unifying methodological
  move across the three papers: bypass the prediction step, output the
  decision directly, optimise the financial objective end-to-end via gradient
  descent.
- **FR-016**: The landing page sidebar MUST display the parquet refresh date
  (or a fallback-mode indicator when no parquets exist), a link to the GitHub
  repository, and an optional FMP-key text input for the per-session
  "Refresh data" workflow.

**Streamlit runtime**

- **FR-017**: The Streamlit server MUST be configured to bind to port 7860 and
  address `0.0.0.0` in headless mode with usage-stats gathering disabled (so
  the same configuration runs locally and on Hugging Face Spaces without
  changes).

**Dependency management**

- **FR-018**: The production dependency file MUST install PyTorch from the
  CPU-only wheel index so the production runtime never pulls GPU CUDA
  libraries.
- **FR-019**: A separate training-only dependency file MUST exist; its
  contents are baked into the GPU training-container image by later phases
  (per constitution v1.1.0 the named GPU host is Modal serverless containers,
  consuming `requirements-train.txt` via
  `modal.Image.pip_install_from_requirements(...)`). Phase 0 ships a minimal
  seed file with GPU-enabled PyTorch; later phases extend it with training-
  only dependencies as needed.

**Tests**

- **FR-020**: A unit-test suite MUST cover the extended `report_metrics`
  function: each new key MUST be cross-checked against its formula on a
  fixed-seed synthetic returns series, and the three original keys MUST be
  unchanged against the same fixture (regression guard).
- **FR-021**: An integration-test suite MUST verify the landing page renders
  without exception under `streamlit.testing.v1.AppTest` in all four
  combinations of {parquet present, parquet absent} × {FMP key set, FMP key
  unset}.

**Configurable data root**

- **FR-022**: A single environment variable `DEEP_FINANCE_DATA_DIR` MUST
  control the directory where `scripts/fetch_data.py` writes parquet output
  and where `src/data.py` looks for parquet input. Default value (when the
  variable is unset) is `./data/` relative to the repository root. The
  bundled CSV fallbacks (`data/aapl.csv`, `data/portfolio_data.csv`) MUST
  always resolve to their fixed repo-relative paths and MUST NOT honour the
  override — they ship in git as part of first-clone usability and would
  otherwise disappear when the developer points the override at a directory
  that has parquets but no CSVs.
- **FR-023**: The OD-2 confirmation files (`sp500_100.proposed.json`,
  `sp500_100.confirmed.json`) MUST live alongside the parquets, i.e. under
  `${DEEP_FINANCE_DATA_DIR}/`. Otherwise the developer's data-lake copy
  diverges from the repo copy and the confirmation flow becomes ambiguous.

### Key Entities

- **Universe**: Named collection of tradeable instruments. Attributes: name
  (e.g. "etf_basket"), member identifiers (where statically known), data
  source (FMP / CME data lake / Kaggle / local CSV), human-readable label,
  list of pages that consume it.
- **DataSnapshot**: A point-in-time materialisation of a universe to disk.
  Attributes: universe reference, on-disk file path, source kind (parquet vs
  CSV), refresh timestamp, row count, date-range covered. Surfaced in the
  landing page sidebar.
- **MetricsReport**: The dictionary returned by `report_metrics(ret)`.
  Attributes: nine numeric keys — three original (`annual_ret`, `annual_std`,
  `annual_sharpe`) and six new (`downside_dev`, `sortino`, `max_drawdown`,
  `calmar`, `pct_positive`, `avg_p_over_l`). Backward-compatibility of the
  three original keys is a non-negotiable invariant.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher with a fresh clone, Python 3.11, and no API
  credentials can go from `git clone` to a rendered landing page in their
  browser in under 5 minutes (one install command, one start command, page
  renders within 30 seconds of the first request).
- **SC-002**: 100% of the three paper cards on the landing page are visible
  and route a click to the correct per-paper page placeholder; the Common
  Thread section is present and contains the unifying-move explanation.
- **SC-003**: After running the FMP fetch script with a valid credential, the
  landing-page data-status sidebar reflects the new refresh date on the next
  app start without any further operator action; the fallback notice
  disappears for every universe whose parquet was produced. The 100-stock
  universe is exempt until OD-2 confirmation is in hand; until then its row
  in the sidebar continues to show "pending confirmation" without blocking
  the rest of the page.
- **SC-004**: The extended metrics function passes its unit tests against a
  fixed-seed `np.random.default_rng(0).normal(0.0005, 0.01, 1000)` series
  using formula-equality assertions (one per metric key) plus key-set,
  finiteness, and signature-invariant checks. The brief's §13.7 tolerance
  bands ("Sharpe ≈ 0.79") were the THEORETICAL values for the population —
  empirical convergence at N=1000 is not tight enough for those bands; the
  test instead asserts each metric matches its formula re-computed inline
  with `pytest.approx(rel=1e-12)`. This is a stronger guard against formula
  drift (Principle V).
- **SC-005**: 100% of the four (parquet × FMP-key) combinations pass the
  landing-page integration test under `streamlit.testing.v1.AppTest`.
- **SC-006**: A second engineer reviewing the repository can locate any file
  listed in §5 of the brief at its specified path without searching — the
  scaffold is layout-conformant.
- **SC-007**: When the data loader falls back to CSV, the visible fallback
  notice names the universe and the operator action required to upgrade,
  verified by a copy-text inspection of the rendered UI.

## Assumptions

- Python 3.11 is the target interpreter (per the project constitution); no
  Python 3.10 or 3.12 compatibility is required at this phase.
- The developer's existing `Utilis/` files are authoritative and will be moved
  to `src/` without modification beyond the explicit extension of
  `metrics.py`.
- The 20-stock ticker list AAPL/ABT/AEP/AXP/BAC/CI/GD/GE/HON/MMM/MO/MRK/NEM/
  NKE/NSC/PFE/PG/PTC/SNA/SO is final and need not be re-confirmed (per brief
  §12 item 4).
- The bundled CSV at `Data/sample_portfolio_data.csv` is the source of truth
  for the 20-stock ticker header and will be relocated to
  `data/portfolio_data.csv` (and `Data/aapl.csv` to `data/aapl.csv`) during
  the scaffold step; existing filename casing differences are intentional and
  to be normalised.
- The FMP Premium tier in use supports bulk historical EOD prices and the
  historical S&P 500 constituents endpoint. Shares-outstanding access is
  tier-dependent; per the OD-3 clarification, the fetch script treats a
  missing shares-outstanding endpoint as a warn-and-skip event (not a
  failure) and omits the column from the affected parquets.
- Phase 0 cannot ship the 100-stock parquet until the developer confirms the
  Claude-Code-proposed sector-balanced ticker list (OD-2). Until confirmation
  is in hand, `scripts/fetch_data.py` skips that parquet with a clear notice.
- The developer maintains an existing data lake on their machine (e.g., CME
  raw source at `/data_lake/data_bento/cme/`) and prefers to co-locate the
  FMP-fetched parquets there. Per the OD-4 clarification (2026-05-17), the
  app and fetch script honour a `DEEP_FINANCE_DATA_DIR` env var that defaults
  to `./data/` (so HF Space works with no config) and is overridden to
  `~/data_lake/deep-finance/` on the developer's machine. A `cp` one-liner
  (documented in `quickstart.md` §US2) is the publish step that puts the
  current parquets back into `./data/` before a deploy.
- Hugging Face Spaces deployment configuration and the GitHub→HF sync workflow
  are out of scope for Phase 0; the `.streamlit/config.toml` written now will
  be reused by Phase 4 without further changes.
- The 100-stock universe is materialised as committed parquet data (no live
  refresh on the production runtime) once Phase 0 ships; subsequent refreshes
  are a developer-local operation.

## Out of Scope

- Per-paper page implementations (Tabs 2/3/4, models, backtests, math/code
  panels) — these are Phases 1–3 deliverables.
- The CME futures fetch script (`scripts/fetch_futures.py`) and the LOB
  Kaggle fetch script (`scripts/fetch_lob_fi2010.py`) — depend on
  developer-local data sources and ship in Phases 1 and 3 respectively.
- Pretrained `.pt` checkpoints — produced on Modal serverless GPU containers during Phases 1–3 per constitution v1.1.0 (Principle III).
- Dockerfile correctness verification, the GitHub→HF Space sync workflow,
  README polish, hero screenshot — Phase 4 deliverables.
- The `run_backtests.py` script body — extended page by page, starts empty in
  Phase 0.
- Hyperparameter sweeps, model selection, or any retraining of the existing
  notebooks' models in this phase.

## Open Decisions

All three Phase-0-blocking decisions identified during specification are
resolved in the `## Clarifications` section above (OD-1, OD-2, OD-3,
session 2026-05-16). The brief's §12 lists eight further open decisions that
do not affect Phase 0 scope; they will be re-surfaced as `[NEEDS
CLARIFICATION]` markers in the specs for Phases 1–4 where they are blocking.
