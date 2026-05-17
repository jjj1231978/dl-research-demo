# Research: Phase 0 — Skeleton & Data Foundation

**Branch**: `001-phase-0-skeleton-data`
**Date**: 2026-05-16
**Inputs**: `spec.md`, `plan.md`, `Project_brief.md` §§4–6, 13

This document records the small set of stack/library decisions reached during
Phase 0 planning. All open-decision questions from the spec (OD-1, OD-2, OD-3)
were resolved during `/speckit-clarify` and are captured in `spec.md`'s
Clarifications section; they are not re-litigated here.

---

## R1 — Multi-page navigation approach

**Decision**: Use Streamlit's filesystem-based `pages/` directory convention.

**Rationale**:
- Brief §5 explicitly names `pages/1_📈_Momentum.py`, `pages/2_💼_Portfolio.py`,
  `pages/3_📖_Order_Book.py` as the page entry points. This is the
  filesystem-based convention.
- It is the convention assumed by `streamlit.testing.v1.AppTest`, which the
  integration tests rely on.
- Page ordering and labelling come for free from the filename pattern
  (`<number>_<emoji>_<title>.py`); no separate router config is needed.

**Alternatives considered**:
- `st.navigation(...)` with programmatic page registration (newer Streamlit
  feature). Rejected because: (a) the brief locks in the filesystem layout,
  (b) AppTest's filesystem discovery still works best with `pages/`, (c) no
  feature advantage for a 4-page app.

---

## R2 — Parquet engine

**Decision**: `pyarrow` as the parquet engine (used both for writing in
`scripts/fetch_data.py` and for reading in `src/data.py`).

**Rationale**:
- Already in the brief's tech-stack table (§3) and `requirements.txt`.
- Pandas defaults to `pyarrow` when both `pyarrow` and `fastparquet` are
  installed; locking it in explicitly avoids drift.
- Better schema preservation (column dtypes, timestamps) than `fastparquet`,
  which matters for the FMP date columns.

**Alternatives considered**:
- `fastparquet`. Rejected: smaller community, weaker schema handling for
  nullable integer types that show up in FMP's shares-outstanding column.

---

## R3 — FMP API client

**Decision**: Raw `requests` against the FMP REST endpoints, with a small
wrapper class in `src/data.py` that handles auth, retries, and rate-limit
backoff. No third-party FMP SDK.

**Rationale**:
- The set of endpoints we call is tiny — historical EOD, historical S&P 500
  constituents, key-metrics (for shares-outstanding). A 100-line wrapper is
  cheaper than auditing a third-party SDK.
- Lets us hold the `requests` dependency that's already in `requirements.txt`
  rather than adding a new one.
- Keeps the surface area small for security review; FMP keys are sensitive.

**Alternatives considered**:
- `fmpsdk` (third-party PyPI package). Rejected: another dependency to track
  for security updates, and its abstraction does not save meaningful code for
  three endpoints.

---

## R4 — `.env` loading

**Decision**: `python-dotenv` loaded at the top of `scripts/fetch_data.py`.
The live Streamlit app does **not** call `dotenv.load_dotenv()` — it reads
`os.environ.get("FMP_API_KEY")` directly, which on HF Spaces is populated by
the platform's secrets mechanism and locally is populated by the developer's
shell.

**Rationale**:
- Brief §15.4 mandates `os.environ`-based reads for local/HF symmetry.
- `python-dotenv` is needed by the fetch script (developer-local convenience)
  but is intentionally not in the app's hot path.
- Avoids the "loaded twice" footgun where the app inherits a stale `.env`
  in a hot-reload scenario.

**Alternatives considered**:
- `pydantic-settings`. Rejected: overkill for a single env var.
- Loading `.env` in `streamlit_app.py` for local dev convenience. Rejected:
  conflicts with the brief's local/HF symmetry requirement.

---

## R5 — `streamlit.testing.v1.AppTest` patterns

**Decision**: Use `AppTest.from_file(...)` (parametrized via `pytest.mark.parametrize`)
to render the landing page across the four data states. Use `monkeypatch`
fixtures to toggle `FMP_API_KEY`, and `tmp_path` fixtures to toggle parquet
presence (the data loader takes its base directory from a module-level
constant that can be patched per-test).

**Rationale**:
- `AppTest` is Streamlit's officially supported testing surface and lives in
  the same package as the runtime, so versions stay in sync.
- The parametrize × monkeypatch × tmp_path pattern is idiomatic pytest and
  scales to the four (parquet × FMP-key) combinations without test
  duplication.
- Avoids the heaviness of a browser-driven test (Selenium, Playwright) for a
  render-only assertion.

**Alternatives considered**:
- `seleniumbase` / Playwright. Rejected: too heavyweight for the assertion
  ("page rendered without exception").
- Snapshot testing of the rendered HTML. Rejected: brittle, no value beyond
  the exception-free render assertion.

---

## R6 — `tests/unit/test_metrics.py` fixture

**Decision**: A module-level pytest fixture `synthetic_returns_series` that
returns `np.random.default_rng(0).normal(0.0005, 0.01, 1000)`. All test
functions consume this fixture; expected values are hard-coded in the test
constants, derived once on the developer's machine and matched against the
tolerance bands in brief §13.7 (Sharpe ≈ 0.79 ± 0.05, max_drawdown ∈
[0.05, 0.20], hit_rate ≈ 0.52 ± 0.05).

**Rationale**:
- `default_rng(0)` is the modern numpy idiom and is reproducible across numpy
  versions ≥ 1.17 (well below our 3.11 ceiling).
- A single shared fixture keeps the test file small and prevents per-test
  seed drift.
- Hard-coding expected values gives the regression check teeth: any change
  to the metric formulas surfaces immediately.

**Alternatives considered**:
- `np.random.seed(0); np.random.normal(...)` (legacy API). Rejected: legacy,
  not preferred in modern numpy.
- Deriving expected values inside each test via the formula. Rejected: tests
  would tautologically pass any implementation that uses the same formula.

---

## R7 — Bundled CSV relocation

**Decision**: Three relocations from existing `Data/` (capital D) and existing
`Utilis/` to the brief's prescribed paths:

| From | To |
|---|---|
| `Data/aapl.csv` | `data/aapl.csv` |
| `Data/sample_portfolio_data.csv` | `data/portfolio_data.csv` (rename) |
| `Data/sample_limit_order_book_data.csv` | `notebooks/reference/data/limit_order_book_data.csv` (rename) |
| `Utilis/*.py` | `src/*.py` (and rename `loss.py` → `losses.py` per brief §13.0) |

The original `Data/` and `Utilis/` directories are removed once the
relocations are complete, to avoid two-source confusion. The `.DS_Store` file
under `Data/` is dropped during relocation.

**Rationale**:
- Brief §5 lists `data/` (lowercase) and `src/` as the authoritative paths.
- Brief §13.0 names `losses.py` (plural) as the canonical filename.
- Two-source confusion (the same file at two paths) is a Principle V
  violation waiting to happen.

**Alternatives considered**:
- Leave `Data/` and `Utilis/` in place and add `data/` / `src/` as the new
  authoritative paths via symlinks. Rejected: complicates packaging, complicates
  `.gitignore`, complicates the Docker build.

---

## R8 — Placeholder page bodies

**Decision**: Each `pages/<n>_<title>.py` placeholder displays the paper
title, authors, arXiv link, and a one-line "Coming in Phase N" notice. No
math, no charts, no model code — those are the respective phase's
deliverable.

**Rationale**:
- Constitution Principle I (Paper-Faithful Replication) forbids "improvising
  on the math". A placeholder that pre-empts later-phase decisions risks
  bleeding paper-specific content into the wrong revision history.
- The arXiv link + author citation is enough to satisfy the navigation
  test (US1 acceptance scenario 2) without pretending the page is more
  complete than it is.

**Alternatives considered**:
- Empty pages (no content). Rejected: produces an undefined-behaviour render
  ("Coming soon" placeholder is friendlier UX).
- Full math + diagrams that get rewritten in Phase N. Rejected: per-phase
  ownership is the point of phase slicing.

---

## R9 — Sidebar data-status component

**Decision**: A small reusable function in `src/data.py` (call it
`render_data_status_sidebar(st_sidebar)`) that scans `data/` for known
parquet filenames, looks up their mtimes, and renders one row per universe
with either the refresh date or the fallback notice. The landing page calls
it; future per-paper pages can call it too.

**Rationale**:
- Constitution Principle IV (Data-as-Artifact) makes the parquet files the
  source of truth for refresh status; their mtime is the natural signal.
- Centralising the rendering in one helper keeps the contract crisp (see
  `contracts/landing_page_sidebar.md`) and avoids drift between the landing
  page and future per-paper page sidebars.

**Alternatives considered**:
- Sidecar JSON files recording refresh timestamps. Rejected: extra files to
  maintain when filesystem mtime is sufficient.
- Database/cache. Rejected: forbidden by Principle IV.

---

## R10 — `requirements-train.txt` Phase 0 seed

**Decision**: In Phase 0, `requirements-train.txt` contains only the
GPU-enabled torch line (no version pin; later phases pin once they validate
on Modal) plus a one-line comment naming the file's purpose: it is baked
into the Modal container image via `Image.pip_install_from_requirements(...)`
when training scripts land in Phases 1–3. It is deliberately minimal — later
phases add their training-only dependencies (e.g., wandb).

**Rationale**:
- Constitution Principle III makes the file's existence non-negotiable in
  Phase 0 (the production/training boundary is established now), but its
  body is mostly empty since Phase 0 has no training pipeline.
- A trivial seed file ensures the file exists in the repo from day one so
  later phases' git diffs only add content.

**Alternatives considered**:
- Defer creating `requirements-train.txt` until Phase 1. Rejected: would
  break the principle of establishing the production/training boundary now.
- Inline the Modal `Image` definition in the seed file. Rejected: Phase 0
  has no Modal-decorated functions to consume it; landing the image
  definition without a consumer is dead code.

---

## R13 — Modal as the GPU host (Phase 0 forward reference)

**Decision**: Phases 1–3 will use **Modal** serverless GPU containers (not
Google Colab Pro as originally written in `Project_brief.md` §2.2 and §7).
Phase 0 has no training code, so the only Phase-0 footprint of this decision
is: (a) no `notebooks/colab/` directory in the file tree; (b) the
`requirements-train.txt` seed file is described as the source of the Modal
container image rather than a Colab `pip install` target.

**Rationale**:
- The developer's primary development surface is VS Code. Modal's CLI-first
  workflow (`modal run src/training/train_*.py`) lives natively in the VS
  Code terminal and avoids the round-trip through `colab.research.google.com`
  on every iteration.
- Modal's `@app.function` decorator pattern lets the same `.py` file run
  locally on CPU (for unit tests and smoke runs) and on Modal's GPU (for the
  real training run) without a notebook wrapper. This is a stronger fit with
  Constitution Principle II (device-agnostic torch) than the Colab "thin
  notebook → import module" pattern, because there is no notebook at all.
- Modal Volumes give first-class persistent storage (training data input,
  intermediate checkpoints, final `.pt` outputs) without the `drive.mount()`
  + manual download dance the Colab pattern required.
- Per-second billing on Modal works out cheaper than Colab Pro's flat
  $10/month for the project's expected training budget (~$10 of Modal
  compute covers the Phase 1–3 training runs estimated in brief §7.1).
- Constitution v1.1.0 names Modal as the default GPU host but explicitly
  allows "another GPU host if Modal is unavailable, provided the device-
  agnostic invariant holds" — so the principle survives the tool swap.

**Forward-looking pattern for Phases 1–3** (illustrative; no code in Phase 0):

```python
# src/training/train_deep_momentum.py — landed in Phase 1
import modal

app = modal.App("deep-finance-train")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-train.txt")
    .add_local_dir("src", "/root/src")
)
volume = modal.Volume.from_name("deep-finance-data", create_if_missing=True)

@app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=3600)
def train_remote(arch: str = "MLP") -> bytes:
    from src.training.train_deep_momentum import train  # device-agnostic
    state_dict, metrics = train(data_dir="/data", arch=arch, device="cuda")
    # ... save to /data/pretrained/...
    volume.commit()
    return state_dict_bytes

@app.local_entrypoint()
def main(arch: str = "MLP"):
    train_remote.remote(arch=arch)
    # then `modal volume get deep-finance-data /pretrained ./data/pretrained/`
```

Local CPU smoke run: `python -m src.training.train_deep_momentum --arch MLP`
(unit-test sized, no `modal` dependency in the import path because the
decorator block is at module top and only imports `modal` when needed by
defining it in a try/except or behind a `if __name__ == "__modal__"` guard).

**Alternatives considered**:

- **Google Colab Pro** (original brief design). Rejected per the user's
  Modal commitment. Trade-offs: flat-fee billing, browser-only UX, manual
  Drive download for checkpoints, intermittent session disconnects.
- **Paid GPU VM via VS Code Remote-SSH** (Lambda Labs / Paperspace / vast.ai).
  Rejected as default because of higher ops burden (provision, SSH key
  management, manual teardown). Acceptable Plan B if Modal becomes
  unavailable.
- **Kaggle Notebooks**. Rejected because of weekly GPU quota and 30 GB
  session storage limit; FI-2010 (Phase 3) alone is ~200 MB raw.
- **Hugging Face Spaces with GPU upgrade**. Rejected because of the
  one-environment risk (training and serving on the same Space conflates
  Constitution Principle III's boundary).

## R12 — Configurable data root via `DEEP_FINANCE_DATA_DIR` env var

**Decision**: One env var, `DEEP_FINANCE_DATA_DIR`, controls the resolved
parquet directory. Default `./data/` (repo-relative). Developer sets it to
`~/data_lake/deep-finance/` on their machine. The bundled CSV fallbacks stay
at fixed repo-relative paths regardless of the override.

**Resolution helper** (lives in `src/data.py`):

```python
def data_root() -> Path:
    return Path(os.environ.get("DEEP_FINANCE_DATA_DIR", "data")).expanduser().resolve()
```

Both `scripts/fetch_data.py` and `src/data.py:get_data_snapshot` consult
`data_root()` to find / write parquet files. The `Universe.parquet_path`
field on the dataclass becomes a `@property` rather than a stored attribute,
re-resolved per call so changing the env var mid-process is visible.

**Publish step** (documented in `quickstart.md` §US2): when the developer
wants the freshly-fetched parquets to flow to HF Space, they run a one-line
`cp` and commit:

```bash
cp -v $DEEP_FINANCE_DATA_DIR/*.parquet ./data/
git add data/*.parquet && git commit -m "data: refresh parquets <date>"
git push
```

**Rationale**:

- Co-locates fetched market data with the developer's existing data lake
  (e.g., CME raw at `/data_lake/data_bento/cme/`) so all market data lives
  under one tree on their machine.
- Keeps the HF deployment story intact: HF clones the repo, finds parquets
  at `./data/`, serves real data. The env var is unset on HF, so the
  default resolves correctly.
- Keeps first-clone usability intact: a researcher who pulls the repo gets
  CSV-fallback behaviour from `./data/aapl.csv` and `./data/portfolio_data.csv`
  with no setup. The CSV paths are not overridable, deliberately.
- Adds exactly one moving part (one env var, one helper function); the
  publish step is a `cp` that can be turned into a Makefile target later
  with no spec change.

**Alternatives considered**:

- Always write to `./data/`. Rejected: developer wanted co-location with
  their existing data lake.
- Always write to `~/data_lake/deep-finance/`. Rejected: HF Space would
  only ever serve fallback CSVs — the showcase loses its "real data" claim.
- Two-stage `~/data_lake/raw/ → ~/data_lake/processed/ → repo/data/`.
  Rejected: more moving parts; the simpler one-var design already gives the
  developer the co-location they wanted.
- Use a config file (e.g., `.deep_finance_config.toml`). Rejected: env vars
  are simpler, work the same on HF as locally, and don't need a parser.

## R11 — `Dockerfile` Phase 0 stub

**Decision**: Phase 0 commits a minimal Dockerfile that captures only what
the brief §15.1 mandates (Python 3.11-slim base, install requirements.txt,
copy app, expose port 7860, set Streamlit env vars, CMD streamlit run). Full
hardening (`.dockerignore`, apt deps if needed, multi-stage build) is Phase 4.

**Rationale**:
- Brief §5 lists `Dockerfile` at the root; not creating it would violate the
  layout-conformance success criterion SC-006.
- The minimal Dockerfile is functional — pushed to HF today it would build —
  but Phase 4 owns deployment correctness verification.

**Alternatives considered**:
- Skip the Dockerfile in Phase 0. Rejected: violates SC-006.
- Ship the full Phase 4 Dockerfile now. Rejected: deployment verification is
  explicitly a Phase 4 acceptance criterion; shipping it untested in Phase 0
  blurs the phase boundary.
