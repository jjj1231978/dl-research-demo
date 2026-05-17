<!--
SYNC IMPACT REPORT
==================
Version change: 1.0.0 → 1.1.0
Rationale: Replaced the named GPU host (Google Colab Pro → Modal serverless
GPU containers) across Principle II's rationale, Principle III's rules, the
Tech Stack section, and the Training workflow section. The underlying
principles (device-agnostic torch, no live training in prod, GPU host
discipline) are unchanged — only the tool reference moves. Treated as MINOR
because no principle was removed or redefined; the substitution is materially
expanded guidance (Modal patterns differ from Colab patterns).

Prior history:
  v1.0.0 (2026-05-16) — initial ratification. Six principles defined.

Principles (6 total, all retained):
  I.   Paper-Faithful Replication
  II.  Device-Agnostic Torch (NON-NEGOTIABLE) — GPU host changed Colab → Modal
  III. Two-Compute-Environment Discipline — Colab references → Modal
  IV.  Data-as-Artifact (parquet-not-DB)
  V.   Pre-existing Canon
  VI.  Test Critical Paths

Sections changed:
  - Principle II rules: dropped "Colab orchestrator notebooks under notebooks/colab/
    are the only place GPU-specific glue belongs" → replaced with Modal-specific
    glue guidance.
  - Principle III rules: replaced "Colab Pro" with "Modal" in checkpoint
    provenance rule.
  - "Development & Deployment Workflow → Training workflow" subsection: fully
    rewritten for Modal (no notebooks; Volume + decorator pattern; modal run
    invocation; Modal Secrets for credentials).
  - "Tech Stack & Runtime Constraints → ML" entry: GPU build target named
    as Modal container image.

Templates / artifacts requiring follow-up:
  ✅ .specify/templates/* — no template edit required; principles propagate
      at /speckit-plan run time.
  ⚠  specs/001-phase-0-skeleton-data/spec.md — FR-019 wording references Colab;
      will be amended in lockstep with this constitution change.
  ⚠  specs/001-phase-0-skeleton-data/plan.md — File tree contains notebooks/colab/
      directory; will be removed in lockstep.
  ⚠  specs/001-phase-0-skeleton-data/tasks.md — T002 creates
      notebooks/colab/.gitkeep; will be retargeted at a Modal-related path or
      dropped in lockstep.
  ⚠  specs/001-phase-0-skeleton-data/research.md — R10 (requirements-train.txt
      seed) references Colab notebooks; will be updated in lockstep.
  ⚠  specs/001-phase-0-skeleton-data/quickstart.md — no training section yet;
      no edit required for Phase 0 itself.
  ⚠  CLAUDE.md — references the Phase 0 plan only; no constitution-version
      reference; no edit required.
  ⚠  Project_brief.md — written extensively around Colab Pro (entire §2.2,
      §7.2, §7.4). Author (user) owns this document; constitution is now the
      authoritative GPU-host reference. Brief will be reconciled when later
      phases are specified.

Follow-up TODOs: none.
-->

# Deep Finance Showcase Constitution

## Core Principles

### I. Paper-Faithful Replication

Every page replicates **named** exhibits, tables, or figures from its source paper.
For the Momentum page that is Lim, Zohren, Roberts (2019) Exhibits 2 / 3 / 4 / 5;
for the Portfolio page, Zhang, Zohren, Roberts (2020) Table 1, Figure 3, and
(stretch) Figure 6; for the Order Book page, Zhang, Zohren, Roberts (2019)
Tables I and II on the FI-2010 benchmark.

Rules:

- Every chart, table, and metric MUST cite the paper section it reproduces.
- Reproduced cells MUST be visually distinguished from paper-cited cells (e.g.
  a `(reproduced here)` vs `(paper-reported)` badge in Tab 4 master tables).
- Math, signal construction, loss functions, and labeling conventions MUST
  follow the paper exactly; no silent "improvements" or convention swaps.
- Where the data substrate differs from the paper (e.g. CME futures vs the
  paper's Pinnacle CLC; FI-2010 z-score variant vs raw), the page MUST disclose
  the substitution and the expected qualitative consistency claim (e.g.
  "ordering preserved: LSTM > MLP > MACD > Sgn(Returns) > Long Only").

Rationale: The app is a research showcase, not a textbook walkthrough. Its
credibility depends entirely on each visitor being able to map what they see
back to the paper they came from. Silent deviation forfeits that.

### II. Device-Agnostic Torch (NON-NEGOTIABLE)

Every PyTorch model and training loop in `src/` MUST detect its device via
`torch.device('cuda' if torch.cuda.is_available() else 'cpu')` and MUST execute
correctly on both. Production runs on CPU (Hugging Face Space, CPU Basic).
Training runs on GPU (Modal serverless containers — default; another GPU host
is acceptable if Modal is unavailable, provided the device-agnostic invariant
holds). The same Python module serves both.

Rules:

- No `.cuda()` calls, no hardcoded `device='cuda'`, no GPU-only ops (e.g.
  `torch.cuda.synchronize`) inside the model definitions or training-loop
  body. The Modal `@app.function(gpu=...)` decorator at the top of each
  `src/training/train_*.py` is the only place GPU-specific glue (image
  definition, Volume mounts, runtime assertions) belongs.
- Every `Tensor` allocation that the user controls MUST take `device` from a
  parameter or from the model's parameter device — never assume CPU/GPU.
- Unit tests for models and training loops MUST be runnable on CPU only.
  `python -m src.training.train_deep_momentum` MUST work locally on CPU for a
  reduced-data smoke run; `modal run src/training/train_deep_momentum.py`
  invokes the same code on Modal's GPU container.

Rationale: The two-environment split (§III) collapses without this. Every hour
saved by skipping device-agnosticism is paid back tenfold by debugging a
checkpoint that only loads on the machine that trained it. Modal's
local-and-remote symmetry (same Python file runs both ways) only works if the
code below the decorator is device-agnostic.

### III. Two-Compute-Environment Discipline

Production and training run in different environments with different rules.
No live training on the production path; no production CPU code reaches the
Modal training container. The boundary is enforced through file layout
(`src/training/` for training, `pages/` + `streamlit_app.py` for production)
and the `requirements.txt` / `requirements-train.txt` split, not convention.

Rules:

- All committed `.pt` checkpoints in `data/pretrained/` MUST be produced
  offline on Modal (or another GPU host) and accompanied by a `<name>.json`
  sidecar per the §7.3 schema (train date, data range, hyperparameters,
  final metrics, git commit). The sidecar's `trained_with` field MUST name
  the GPU host and image hash (e.g., `"Modal A10G, image deep-finance-train:abc1234"`).
- The production Streamlit app MUST NOT call `optimizer.step()` in any code path
  reached without an explicit user opt-in ("Train your own (tiny subsample)").
- "Train your own" toggles MUST have a hard time budget of ≤ 30 seconds on
  HF CPU Basic, run on a deliberately reduced subsample, and be labeled
  "illustrative only" in the UI.
- `requirements.txt` (production) MUST install CPU PyTorch
  (`torch --index-url https://download.pytorch.org/whl/cpu`).
  `requirements-train.txt` (Modal image) is a separate file, baked into the
  Modal container image via `Image.pip_install_from_requirements(...)`, and
  MUST NOT be installed by the production Docker image.
- Modal credentials (`modal token new`) and Modal Secrets (e.g., `fmp-key`
  if a fetch step runs on Modal) live in the developer's Modal account, NEVER
  in the repo or in `.env`. The repo references Modal Secrets by name only.

Rationale: HF CPU Basic is 2 vCPU / 16 GB RAM and sleeps after 48h idle.
Attempting real training there would either OOM or wall-clock out. Modal's
serverless model means training spins up on demand and tears down when done,
so there's no persistent training infrastructure to drift out of sync with
production — but only if the boundary is policed by the file layout and
requirements split.

### IV. Data-as-Artifact (parquet-not-DB)

All persistent state in the project is files. No database, no API backend, no
cache server. The Streamlit app is the whole app.

Rules:

- Market data: parquet files under `data/` (one per universe per page),
  produced by one-shot scripts under `scripts/` and committed to the repo.
- Model weights: `.pt` files under `data/pretrained/` with adjacent `.json`
  metadata sidecars.
- Backtest results: parquet files under `data/backtests/`, produced by
  `scripts/run_backtests.py` and committed to the repo.
- Fallbacks for first-clone usability: bundled CSVs (`data/aapl.csv`,
  `data/portfolio_data.csv`) MUST exist so `streamlit run streamlit_app.py`
  works with no FMP key and no parquet files present.
- Loaders MUST prefer parquet, fall back to CSV, and emit a Streamlit notice
  when falling back so the data provenance is never ambiguous to the visitor.

Rationale: A research showcase is read-mostly. Parquet gives columnar I/O,
schema preservation, and free version control via git. A database would buy
nothing and would block the "fork and run" usability that makes the repo
appealing to researchers.

### V. Pre-existing Canon

Four files were authored by the developer before this project began and are
canonical. Claude Code and contributors MUST use them as-is (or extend per
§13.7 of the brief in the case of `metrics.py`); they MUST NOT be rewritten,
renamed, or refactored "for cleanliness".

Canonical files:

- `src/losses.py` — `Neg_Sharpe(portfolio)` function and `SharpeLoss(nn.Module)`
  class. Names: capital `N`, capital `S` — preserved exactly.
- `src/early_stopper.py` — `EarlyStopping` class. `savepath` is the first
  positional argument; saves `model.state_dict()` on improvement.
- `src/torch_data.py` — `MyDataset(data.Dataset)` numpy→torch wrapper. Generic
  name preserved because the source notebooks reference it directly.
- `src/metrics.py` — starter `report_metrics(ret)`. Extended per §13.7 of the
  brief; existing keys (`annual_ret`, `annual_std`, `annual_sharpe`) MUST stay
  backward-compatible.

Rules:

- For multi-asset uses of `SharpeLoss` (Portfolio page), the caller MUST
  pre-aggregate `outputs * future_rets` into a portfolio-return series before
  calling the loss. Do not modify the loss signature.
- `Neg_Sharpe` deliberately omits an epsilon-for-numerical-stability term —
  this matches the source notebooks and is treated as a feature, not a bug.

Rationale: These four files anchor the project's notebook→app provenance.
Renaming or restyling them would silently break parity with the reference
notebooks under `notebooks/reference/` and force every downstream contributor
to mentally translate between two naming conventions.

### VI. Test Critical Paths

Testing rigor scales with consequence. Not TDD-everywhere, but no shipping a
page whose headline numbers can't be cross-checked.

Required test surfaces (per feature, where applicable):

- **Unit tests**: every formula added to `src/metrics.py` (Sortino, Calmar,
  max drawdown, hit rate, avg P / avg L); every signal function in
  `src/strategies/` (Sgn, MACD ensemble, classical portfolio weights).
- **Integration tests**: each `pages/*.py` MUST render without exception under
  `streamlit.testing.v1.AppTest` with both parquet-present and parquet-absent
  data states.
- **Data-pipeline parity**: classical methods that have a reference
  implementation in `notebooks/reference/*.ipynb` MUST match notebook outputs
  within numerical tolerance documented in the feature's `quickstart.md`.
- **Checkpoint smoke tests**: every committed `.pt` MUST load on CPU into its
  corresponding `nn.Module` and produce a non-NaN forward pass on a unit input.

Out of scope unless the spec explicitly requests them:

- UI snapshot tests, browser-driven Selenium/Playwright tests, ML model
  performance tests beyond the qualitative ordering claim from Principle I.

Rationale: The Tab 4 master tables are the showcase's reason to exist. Any
number that surfaces there without a unit or parity check behind it is a
liability. Conversely, exhaustive UI-tier testing for a research demo is a
poor ROI use of contributor time.

## Tech Stack & Runtime Constraints

The repo's stack is fixed by the brief and the deployment target.

- **Language**: Python 3.11 (pinned in the Dockerfile).
- **App framework**: Streamlit (latest), multi-page layout via `pages/`.
- **ML**: PyTorch — CPU build in production (`requirements.txt`); GPU build inside the Modal container image (`requirements-train.txt`, baked via `modal.Image.pip_install_from_requirements`).
- **GPU host**: Modal serverless GPU containers (T4 / A10G / A100 selected per training script via the `@app.function(gpu=...)` decorator). Modal CLI authenticated once via `modal token new`. Persistent data and intermediate checkpoints live on a Modal Volume; the developer pulls finals via `modal volume get` and commits to `data/pretrained/`.
- **Data**: pandas, pyarrow (parquet I/O), numpy.
- **Classical stats**: scikit-learn (`ShrunkCovariance` for portfolio benchmarks).
- **Charts**: Plotly (interactive hover, log-scale supported, web-friendly).
- **Math rendering**: KaTeX via `st.latex`.
- **Market data**: FMP Premium API (one-shot fetch only) for equities/ETFs;
  developer-local CME data lake for futures; Kaggle FI-2010 mirror for LOB.

Runtime budget (HF CPU Basic: 2 vCPU, 16 GB RAM):

- Cold-start page load: < 30 seconds.
- Warm page load: < 3 seconds.
- Tab switch within a page: < 500 ms.
- "Train your own" subsample run: ≤ 30 seconds wall-clock.

Repository structure: §5 of `Project_brief.md` is authoritative. Feature plans
MUST reference it; they MUST NOT propose alternative layouts without a
constitutional amendment.

## Development & Deployment Workflow

Two homes, one direction of flow:

- **GitHub `main`** is the canonical source of truth. Issues, PRs, README,
  citations, contributor traffic all live here.
- **Hugging Face Space** is the production runtime. It receives code via a
  GitHub Action that force-pushes `main` to the Space's `main` branch on every
  push to GitHub.

Rules:

- Developers push to GitHub only. The Space is downstream and rebuilds itself.
- Force-push is intentional (see brief §15.3): the Space's auto-generated initial
  commit is overwritten on first sync. Do not weaken to a fast-forward push.
- Secrets live in three stores with three purposes (brief §15.4): `HF_TOKEN` in
  GitHub Secrets, `HF_USERNAME` / `HF_SPACE_NAME` in GitHub Variables,
  `FMP_API_KEY` in HF Space Secrets, and `.env` for local dev only.
- A push to GitHub `main` MUST yield a successful Space rebuild within 5 minutes;
  the workflow under `.github/workflows/sync-to-hf.yml` is the single source of
  the sync contract.

Training workflow (Modal):

- Training source code lives under `src/models/` and `src/training/` and is
  device-agnostic per Principle II.
- Each `src/training/train_*.py` file is BOTH a plain Python module (callable
  locally on CPU for unit tests and reduced-data smoke runs) AND a Modal
  app: the top of the file declares the Modal `App`, the container `Image`
  (with `pip_install_from_requirements("requirements-train.txt")`), the
  shared `Volume` for data + checkpoints, the `@app.function(gpu=...)`
  decorator on the training function, and an `@app.local_entrypoint()` that
  parses CLI args and dispatches.
- `modal run src/training/train_deep_momentum.py --arch MLP` runs the
  training on Modal's GPU container. `python -m src.training.train_deep_momentum
  --arch MLP` runs the same logic locally on CPU (reduced data, smoke-test
  size).
- Training MUST checkpoint to the Modal Volume every N epochs and accept a
  `resume_from` argument so a container kill / preemption / quota interruption
  does not destroy progress. Modal containers are ephemeral; only the Volume
  persists.
- The developer pulls the final `.pt` + JSON sidecar from the Volume via
  `modal volume get dl-research-data /pretrained ./data/pretrained/`,
  commits them, pushes; the GitHub Action syncs to HF Space which serves
  them. The repo is the integration point between the Modal GPU environment
  and the live app.
- Modal Volume lifecycle: create once with `modal volume create dl-research-data`,
  populate input data once with `modal volume put dl-research-data
  ~/data_lake/deep-finance/ /data`, re-populate when parquets are refreshed.

## Governance

**Authority**: This constitution supersedes ad-hoc preferences expressed in
chat or PR comments. Where a feature request conflicts with a principle, the
principle wins unless the constitution is amended first.

**Amendment procedure**:

1. Propose the amendment in a PR that touches only `.specify/memory/constitution.md`
   and any templates that must change in lockstep.
2. The PR description MUST include a Sync Impact Report (rendered as the HTML
   comment at the top of this file) with: version delta, principles changed,
   templates updated, follow-up TODOs.
3. Version policy (semver):
   - **MAJOR**: a principle is removed or its meaning is materially
     incompatible with prior usage (e.g. dropping device-agnosticism).
   - **MINOR**: a new principle or governance section is added; an existing
     principle is materially expanded.
   - **PATCH**: clarifications, typo fixes, non-semantic wording changes.

**Compliance review**: every `/speckit-plan` invocation runs a Constitution
Check gate before research and re-checks it after Phase 1 design. Violations
must be either resolved or recorded in the plan's Complexity Tracking table
with explicit justification.

**Runtime guidance**: agents (Claude Code in particular) and contributors
should treat `CLAUDE.md` as the entry point. Once the first feature plan
exists, `CLAUDE.md` MUST be expanded to surface the principles above and
point at the active feature plan.

**Version**: 1.1.0 | **Ratified**: 2026-05-16 | **Last Amended**: 2026-05-17
