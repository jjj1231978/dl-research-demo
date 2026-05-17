# Project Brief — Deep Learning for Quantitative Finance: Interactive Research Showcase

## 1. Mission

Build a public, interactive Streamlit application that demonstrates three influential papers from the Oxford-Man Institute of Quantitative Finance applying deep learning to three canonical quant-finance problems. The app is framed as a **research showcase**, not a textbook walkthrough — each page foregrounds the paper and its core contribution, with interactive demos that let visitors explore the methods on real market data.

The three papers:

1. **DeepLOB** — Zhang, Zohren, Roberts. *"DeepLOB: Deep Convolutional Neural Networks for Limit Order Books."* IEEE Transactions on Signal Processing, 2019. [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)

2. **Deep Momentum Networks** — Lim, Zohren, Roberts. *"Enhancing Time Series Momentum Strategies Using Deep Neural Networks."* Journal of Financial Data Science, 2019. [arXiv:1904.04912](https://arxiv.org/abs/1904.04912)

3. **Deep Portfolio Optimization** — Zhang, Zohren, Roberts. *"Deep Learning for Portfolio Optimization."* 2020. [arXiv:2005.13665](https://arxiv.org/abs/2005.13665)

Each paper shares the same conceptual move that the app should make explicit: **bypass the prediction step and optimize the financial objective end-to-end** (Sharpe ratio, classification accuracy, or downstream P&L) via deep neural networks with appropriate output activations.

## 2. Deployment Target & Constraints

This project has **two distinct compute environments**: the production hosting target and the offline training environment. Conflating them is a common source of wasted work — every model and training loop must run in both, so device-agnosticism is non-negotiable.

### 2.1 Production hosting — Hugging Face Space (primary) + GitHub (canonical source)

The app has **two homes** with different roles:

- **GitHub** — canonical source of truth. Public, viewable, where the README lives, where issues and PRs go, where the BibTeX citation is found, where researchers discover the code.
- **Hugging Face Space** — production runtime. Where the app actually serves users. Docker SDK (Streamlit is no longer a top-level HF SDK; it lives as a template under Docker).

Code flows one direction: **GitHub `main` → HF Space `main`**, automated via a GitHub Action. The developer pushes to GitHub only; the HF Space updates itself. Full deployment mechanics in §15.

- **Compute (HF free CPU Basic):** 2 vCPU, 16 GB RAM, sleeps after ~48h idle. Upgrade to paid CPU (~$22/month) from Space Settings → Hardware to disable sleep when ready for public launch.
- **Implication 1:** Use CPU-only PyTorch in `requirements.txt` (`torch --index-url https://download.pytorch.org/whl/cpu`)
- **Implication 2:** No live training in the production path — all models pretrained offline on Modal serverless GPU containers (see §2.2) and committed as `.pt` files. "Train your own" toggles run only on tiny subsamples with a hard time budget for illustration.
- **Implication 3:** Data is pre-fetched once via a script and committed as parquet; the app never calls FMP unless the user explicitly opts in with their own key (which is stored as an HF Space secret, not in the repo).
- **Python:** 3.11 (pinned in the Dockerfile)
- **Optional secondary deployment:** Streamlit Community Cloud can also be pointed at the same GitHub repo as a free fallback URL. Costs nothing to maintain in parallel.

### 2.2 Training environment — Modal serverless GPU containers

> **Note (2026-05-17 amendment, see constitution v1.1.0)** — The original draft of this brief named Google Colab Pro as the GPU host. The decision was changed to **Modal** to give a VS-Code-native workflow with no browser round-trip. The constitution at `.specify/memory/constitution.md` is now the authoritative GPU-host reference; this section has been rewritten to reflect that. Wherever the rest of this brief still says "Colab", read it as "Modal" until that section is revised.

All model training that produces the checkpoints in `data/pretrained/` is performed on **Modal** serverless GPU containers. Training is **never** done locally on the developer's machine (beyond CPU smoke tests) and **never** on Streamlit Cloud or HF Space.

- **Why Modal:** Pay-per-second GPU billing on T4 / A10G / A100 (~$0.50 / $1.50 / $3.50 per hour respectively). CLI-first workflow (`modal run src/training/train_*.py`) integrates natively with VS Code's terminal. No browser tab, no Drive mount, no manual checkpoint download — `modal volume get` pulls finals back to the local repo. Total Phase 1–3 training compute is estimated under $10 if you do not retrain.
- **Workflow:**
  1. Training source code lives in `src/models/*.py` and `src/training/*.py` as device-agnostic Python modules — testable locally on CPU with `python -m src.training.train_deep_momentum`, executable on Modal GPU with `modal run src/training/train_deep_momentum.py`.
  2. Each `src/training/train_*.py` file declares its Modal `App`, `Image`, `Volume`, and `@app.function(gpu="T4")` decorator at the top of the file. There is **no separate notebook layer** — the decorator replaces the Colab notebook.
  3. The developer pulls the final `.pt` + JSON sidecar from the Modal Volume via `modal volume get deep-finance-data /pretrained ./data/pretrained/`, commits to the repo, and pushes; the GitHub Action syncs to HF Space which serves them.
- **Device-agnosticism is mandatory:** Every model and every training loop must detect the device (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`) and work in both. The source notebooks already do this — preserve the pattern. The Modal decorator runs the function inside a CUDA-enabled container, so `torch.cuda.is_available()` returns `True` there and `False` locally.
- **Session reliability:** Modal containers are ephemeral — preemption, image rebuild, or timeout can interrupt a long run. Training functions must checkpoint to the **Modal Volume** every N epochs and accept a `resume_from` argument so a restart picks up the latest checkpoint without losing progress. The Volume persists across container instances; only it survives.
- **Separate requirements file:** `requirements-train.txt` contains GPU torch + training-only deps (e.g. wandb if desired); it is baked into the Modal container image via `modal.Image.pip_install_from_requirements("requirements-train.txt")`. The production `requirements.txt` stays CPU-lean and is **not** consumed by Modal.
- **Modal authentication:** one-time `modal token new` from the developer's terminal (writes to `~/.modal.toml`). Modal Secrets (e.g., `fmp-key` if a data step ever runs on Modal) live in the developer's Modal account and are referenced by name in the decorator — never committed to the repo.

## 3. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Frontend | Streamlit (latest) | Multi-page app via `pages/` |
| ML | PyTorch (CPU) | Mirrors the source notebooks |
| Data | pandas, pyarrow, numpy | Parquet for cached data |
| Stats | scikit-learn | `ShrunkCovariance` for classical portfolio |
| Charts | Plotly | Cleaner than matplotlib for web; interactive hover |
| Math | KaTeX via `st.latex` | For equation panels |
| Data source | FMP Premium API | Used once to populate parquet; optional at runtime |

## 4. Data Strategy

### 4.1 Data fetch scripts (one-shot, local)

Two separate fetch scripts produce the parquet snapshots that ship with the repo. Neither runs in production — they're invoked by the developer locally, the outputs are committed.

**`scripts/fetch_data.py`** — FMP source for Portfolio page data.

Pulls equity/ETF data via FMP Premium (key from `.env` or `st.secrets`), writes parquet files, exits. **Produces three parquets**, one per Portfolio universe:

- `data/etf_basket.parquet` — VTI, AGG, DBC, VIXY (paper-faithful)
- `data/sp500_20.parquet` — the 20 tickers from existing `portfolio_data.csv` (AAPL, ABT, AEP, AXP, BAC, CI, GD, GE, HON, MMM, MO, MRK, NEM, NKE, NSC, PFE, PG, PTC, SNA, SO), with history pulled fresh from FMP through today
- `data/sp500_100.parquet` — 100 sector-balanced names, using historical S&P 500 constituents to avoid survivorship bias

FMP endpoints to use:
- Historical S&P 500 constituents — to select the 100-stock universe without survivorship bias
- Bulk EOD historical prices — for all three universes (Premium+ feature)
- Per-symbol historical-price-full — fallback if bulk unavailable
- Company profile — for sector tagging
- **Shares-outstanding (`enterprise-values` or `key-metrics` endpoint)** — required for Diversity-Weighted Portfolio (DWP) market-cap weighting. If unavailable on user's FMP tier, DWP benchmark is skipped with a warning.

**Time range:** Each universe uses **whatever history is available** rather than a forced common start date. Per-universe targets:
- `etf_basket.parquet` — limited by youngest ETF inception (VIXY launched 2011, so ETF basket starts ~2011)
- `sp500_20.parquet` — these 20 names all trade back to 1990s; pull from earliest available (~1992, matches existing CSV)
- `sp500_100.parquet` — historical constituents endpoint handles index membership over time; tickers naturally enter/exit the dataset as they enter/exit the S&P 500. Earliest available coverage starts ~1995–2000 depending on names selected.

Backtests align on the **intersection of valid dates** per universe (no forced common start across universes), and per-strategy results report their own effective date range. This is normal practice; the brief doesn't try to force a contrived "all three universes from 2014" alignment.

**`scripts/fetch_futures.py`** — CME futures source for Momentum page data.

Reads from the developer's local data lake at `/data_lake/data_bento/cme`, normalizes contracts (ratio-adjusted continuous futures per Pinnacle Data Corp CLC conventions used in Lim et al. 2019), winsorizes outliers, writes `data/cme_futures.parquet`.

- **Data location:** `/data_lake/data_bento/cme` — developer's machine only; **not available to Claude Code or the HF Space**. The fetch script runs once locally and produces the committed parquet.
- **Reference codebase:** `/QIS_Commodities` — developer's existing repo for futures data loading and base trend-following strategy construction. Claude Code should mirror the data-loading conventions from this codebase (column names, contract metadata, roll handling, winsorization thresholds). **Claude Code needs read access to this path to follow the patterns faithfully** — see §4.5.
- **Universe:** ~88 ratio-adjusted continuous futures contracts spanning commodities, fixed income, equities, FX (matching the Lim et al. 2019 dataset shape).
- **Time range:** Aim for as much history as available, minimum 1990–today. The paper uses 1990–2015 with 1995–2015 as the test window; we extend through today to include recent regimes.

**`scripts/fetch_lob_fi2010.py`** — Kaggle-hosted FI-2010 benchmark for the Order Book page.

The Kaggle dataset at [praanj/limit-orderbook-data](https://www.kaggle.com/datasets/praanj/limit-orderbook-data) is a redistribution of the FI-2010 benchmark (Ntakaris et al. 2017), which is exactly the dataset used in Setups 1 and 2 of the DeepLOB paper's experiments. Original source: [Etsin / Fairdata.fi](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) (Helsinki/Aalto research data).

- **Contents:** ~4M time-series samples from 5 stocks on Nasdaq Nordic, 10 consecutive days (June 2010), 10 LOB levels each side (40 features per timestamp), pre-normalized in three flavors (z-score, min-max, decimal precision), with smoothed labels at 5 prediction horizons (10, 20, 30, 50, 100 ticks).
- **Why this matters:** the existing `limit_order_book_data.csv` from the GitHub author is a small handcrafted slice — fine for a demo, but FI-2010 is the canonical benchmark every LOB paper compares against. Having it gives the Order Book page legitimate research-grade backing.
- **Authentication:** Kaggle requires a `~/.kaggle/kaggle.json` API token (developer creates at `kaggle.com/settings`). The fetch script reads it from there or from `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars.
- **Output:** `data/lob_fi2010.parquet` — single parquet with all 10 days, z-score normalization (matches DeepLOB paper's primary setup), labels for horizon k=10 (the paper's default). Schema: `(day, timestamp, *40_features, label_k10)`.
- **Implementation:** uses the `kaggle` Python package (`pip install kaggle`) inside `scripts/fetch_lob_fi2010.py`. Downloads, unzips, processes the original `.txt` layout into parquet, exits.
- **Size estimate:** raw download ~200 MB, processed parquet ~50–100 MB compressed. Fits in a public GitHub repo without LFS.
- **License:** original dataset is research-use; Claude Code should add a clear attribution in the README and a `data/LOB_DATA_ATTRIBUTION.md` citing Ntakaris et al. 2017 and the Kaggle/Fairdata sources.

### 4.2 Universes (all derived from one parquet)

| Universe | Members | Used by |
|---|---|---|
| CME futures | ~88 ratio-adjusted continuous futures contracts (commodities, fixed income, equities, FX) | **Momentum page (primary)** |
| Toy single-asset | AAPL daily prices | Momentum page (fallback when CME parquet absent) |
| **ETF basket (paper-faithful)** | VTI, AGG, DBC, VIXY (proxy for VIX index) | Portfolio page (default mode) |
| **20-stock (FMP-extended)** | AAPL, ABT, AEP, AXP, BAC, CI, GD, GE, HON, MMM, MO, MRK, NEM, NKE, NSC, PFE, PG, PTC, SNA, SO — same as `portfolio_data.csv`, history extended via FMP | Portfolio page (stocks mode) |
| **100-stock S&P 500** | 100 sector-balanced names, historical constituents (no survivorship bias) | Portfolio page (large mode) |
| **LOB benchmark (FI-2010)** | 5 stocks × 10 days from Nasdaq Nordic, 40 LOB features per timestamp, ~4M samples (Ntakaris et al. 2017 via Kaggle) | Order Book page (canonical, no fallback) |

### 4.3 Optional power-user refresh

Sidebar exposes an optional FMP API key input (also readable from `st.secrets["FMP_API_KEY"]` for the live deployment). When present, a "Refresh data" button re-runs `scripts/fetch_data.py` (Portfolio page data only). The futures parquet has no live-refresh path on HF — it requires re-running `scripts/fetch_futures.py` on the developer's machine, which has access to the data lake.

### 4.4 Bundled quickstart data (CSVs)

Two CSV files ship in the repo so the app's Momentum and Portfolio pages work on first clone — useful for visitors and for local development before the fetch scripts have run:

| File | Purpose | Used by | Superseded by |
|---|---|---|---|
| `data/aapl.csv` | Single-asset daily prices for AAPL | Momentum page (toy-mode fallback when CME parquet absent) | `data/cme_futures.parquet` once fetched |
| `data/portfolio_data.csv` | 20-stock daily prices, sector-balanced | Portfolio page (fallback when FMP parquets absent) | `data/sp500_20.parquet`, `data/sp500_100.parquet`, `data/etf_basket.parquet` once fetched |

The data-loading layer (`src/data.py`) checks for parquet first, falls back to CSV if absent, and logs a notice in the Streamlit UI.

**Note on the LOB sample CSV:** the small `limit_order_book_data.csv` from the original GitHub author is **not** a runtime input to the Streamlit app. It was a smoke-test sample for the reference notebook (`02_predictive_signal_lob.ipynb`) and ships in `notebooks/reference/data/limit_order_book_data.csv` for anyone running that notebook locally. The Order Book page reads only from `data/lob_fi2010.parquet`; if that's absent, the page shows a setup-instruction banner rather than degrading to a toy mode.

In Momentum-page toy-mode, the Exhibit-style tables and box plots will be sparse (only AAPL, not 88 futures) and the page should display a prominent banner: *"Showing single-asset toy mode. Full Exhibits 2/3/4/5 replication requires `data/cme_futures.parquet` — run `scripts/fetch_futures.py` locally."*

### 4.5 External reference codebases

Two paths on the developer's machine that Claude Code may need to read in order to faithfully follow data and strategy conventions:

| Path | Purpose | Access required by Claude Code |
|---|---|---|
| `/data_lake/data_bento/cme` | Source CME futures data (DataBento format) consumed by `scripts/fetch_futures.py` | **Read access required** to write the fetch script |
| `/QIS_Commodities` | Developer's existing repo for CME data loading and base trend-following strategy construction. The fetch script's data-loading conventions (column names, contract metadata, roll handling, winsorization) and `src/strategies/tsmom.py`'s signal construction should mirror what this codebase already does. | **Read access required** to mirror conventions |
| Kaggle API token (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars) | Developer's Kaggle credentials, consumed by `scripts/fetch_lob_fi2010.py` to download the FI-2010 benchmark dataset | Required to run the LOB fetch script; not needed by the live app or by Claude Code during development |

If Claude Code does not have access to these paths, it should:
1. Stop and ask the developer to either grant access or copy the relevant excerpts (data-loading function, base strategy definitions, contract metadata) into the project as committed files
2. Not guess or invent the conventions — the goal is parity with the developer's existing infrastructure

## 5. Repository Structure

```
deep-finance-showcase/
├── streamlit_app.py                  # landing page (Page 0)
├── pages/
│   ├── 1_📈_Momentum.py              # Lim, Zohren, Roberts 2019 + classical baselines
│   ├── 2_💼_Portfolio.py             # Zhang, Zohren, Roberts 2020 + classical baselines
│   └── 3_📖_Order_Book.py            # Zhang, Zohren, Roberts 2019 + classical baselines
├── src/
│   ├── __init__.py
│   ├── data.py                       # FMP client + parquet loader
│   ├── universes.py                  # universe definitions
│   ├── metrics.py                    # ⚑ EXISTS (user) — extend per §13
│   ├── losses.py                     # ⚑ EXISTS (user) — use as-is, Neg_Sharpe + SharpeLoss
│   ├── early_stopper.py              # ⚑ EXISTS (user) — use as-is
│   ├── torch_data.py                 # ⚑ EXISTS (user) — PyTorch Dataset wrapper, use as-is
│   ├── strategies/
│   │   ├── tsmom.py                  # TSMOM, SMA, MACD signals
│   │   ├── cross_sectional.py        # long-short ranking (Portfolio page optional)
│   │   ├── vol_targeting.py
│   │   ├── classical_portfolio.py    # min-var, max-div
│   │   └── lob_baselines.py          # last-tick, OBI EWMA, linear-regression baseline
│   ├── models/
│   │   ├── deep_momentum.py          # MLP for Sharpe-optimized signal
│   │   ├── deep_portfolio.py         # MLP with softmax output
│   │   └── deeplob.py                # CNN + Inception + LSTM
│   ├── training/                     # device-agnostic training loops
│   │   ├── train_deep_momentum.py    # parameterized by arch (MLP/LSTM); Sharpe loss only
│   │   ├── train_deep_portfolio.py
│   │   └── train_deeplob.py
│   └── components/
│       ├── metric_card.py            # reusable Streamlit metric display
│       ├── equity_curve.py
│       ├── math_panel.py             # collapsible LaTeX block
│       └── paper_citation.py
├── data/
│   ├── aapl.csv                     # ⚑ BUNDLED Momentum-page toy-mode fallback
│   ├── portfolio_data.csv           # ⚑ BUNDLED Portfolio-page fallback
│   ├── LOB_DATA_ATTRIBUTION.md      # Ntakaris et al. 2017 citation, Kaggle/Fairdata sources
│   ├── cme_futures.parquet          # produced by fetch_futures.py; primary Momentum data
│   ├── etf_basket.parquet           # produced by fetch_data.py; Portfolio ETF mode
│   ├── sp500_20.parquet             # produced by fetch_data.py; Portfolio 20-stock mode
│   ├── sp500_100.parquet            # produced by fetch_data.py; Portfolio 100-stock mode
│   ├── lob_fi2010.parquet           # produced by fetch_lob_fi2010.py; primary (only) Order Book data
│   ├── pretrained/
│   │   ├── mlp_sharpe.pt            # Momentum: MLP × Sharpe loss
│   │   ├── lstm_sharpe.pt           # Momentum: LSTM × Sharpe loss
│   │   ├── deep_portfolio_etfs.pt        # Portfolio: ETF basket
│   │   ├── deep_portfolio_20stock.pt     # Portfolio: 20-stock universe
│   │   ├── deep_portfolio_100stock.pt    # Portfolio: 100-stock universe
│   │   └── deeplob_fi2010.pt        # Order Book: trained on FI-2010 benchmark
│   └── backtests/
│       ├── momentum_results.parquet     # consumed by Momentum page Tab 4
│       ├── portfolio_results.parquet    # consumed by Portfolio page Tab 4
│       └── order_book_results.parquet   # consumed by Order Book page Tab 4
├── scripts/
│   ├── fetch_data.py                 # local: one-shot FMP pull → parquet (Portfolio data)
│   ├── fetch_futures.py              # local: CME data_lake → parquet (Momentum data)
│   ├── fetch_lob_fi2010.py           # local: Kaggle FI-2010 download → parquet (Order Book data)
│   └── run_backtests.py              # local: produces data/backtests/*.parquet
│                                     # (training lives in src/training/train_*.py, runnable both locally on CPU and on Modal GPU)
├── notebooks/
│   └── reference/                    # ORIGINAL source notebooks (read-only; reference only)
│       ├── 01_time_series_momentum.ipynb
│       ├── 02_cross_sectional_momentum.ipynb
│       ├── 03_deep_momentum_strategy.ipynb
│       ├── 01_classical_portfolio_optimization.ipynb
│       ├── 02_deep_portfolio_optimization.ipynb
│       ├── 02_predictive_signal_lob.ipynb
│       └── data/
│           └── limit_order_book_data.csv   # smoke-test sample for the reference notebook only
│                                     # Note: no notebooks/colab/ directory — Phases 1–3 use Modal scripts
│                                     # (src/training/train_*.py with @app.local_entrypoint()), not notebooks.
├── papers/                           # reference PDFs
│   ├── DeepLOB.pdf
│   ├── DeepMomentum.pdf
│   └── DL_Portfolio_Optimization.pdf
├── .github/
│   └── workflows/
│       └── sync-to-hf.yml            # GitHub → HF Space auto-deploy
├── Dockerfile                        # HF Space runtime (Python 3.11 + Streamlit)
├── requirements.txt                  # CPU-only torch, runtime deps
├── requirements-train.txt            # GPU torch + training-only deps; baked into the Modal container image
├── packages.txt                      # apt deps if needed (likely empty)
├── .streamlit/
│   ├── config.toml                   # theme, server port, address
│   └── secrets.toml.example
├── .env.example                      # local-dev env vars (FMP_API_KEY=...)
├── .gitignore                        # excludes .env, __pycache__, model checkpoints under dev
├── .dockerignore                     # excludes papers/, notebooks/, .git/ from Docker build
├── LICENSE                           # MIT
└── README.md                         # YAML front matter for HF, body for GitHub
```

## 6. Page-by-Page Specifications

The app is **three pages, one per paper**. Each page is a self-contained research story: problem framing → classical baselines → deep method → side-by-side comparison. This shape mirrors the showcase narrative (classical vs deep, repeated across three domains) and keeps navigation minimal — visitors landing from any of the three papers find their paper's content in one place.

Every page follows the same skeleton:
1. **Header:** Paper title, authors, year, venue, arXiv link, one-line elevator pitch
2. **Sidebar:** Interactive controls scoped to that page (universe, hyperparameters, dates, method toggles)
3. **Main area structured as tabs:**
   - **Tab 1 — Problem & Data:** What problem the paper solves, what data the page is using, key summary statistics (one chart, one table)
   - **Tab 2 — Classical Benchmarks:** All classical/non-DL methods relevant to the problem, with toggles, overlaid equity curves (or accuracy plots for LOB), metrics table
   - **Tab 3 — Deep Method:** The paper's deep approach, with controls, pretrained model results, "train your own" opt-in
   - **Tab 4 — Comparison:** Overlay everything from Tabs 2 + 3 on one chart, master metrics table, key observations callout
4. **Collapsible "Method" panel** below the tabs: LaTeX equations + 2-paragraph plain-English explanation
5. **Collapsible "Code" panel:** Relevant model/loss snippet, syntax-highlighted
6. **Footer:** arXiv link, GitHub source for the page, citation BibTeX (copy button)

### Page 0 — Landing (`streamlit_app.py`)

- App title: *"Deep Learning for Quantitative Finance — Interactive Showcase"*
- Tagline mentioning the Oxford-Man Institute and the three papers
- **Three large cards, one per paper**, each linking directly to its corresponding page. Card content: paper title, authors, year, venue, arXiv link, one-paragraph abstract, "Explore →" button.
- Below the cards: a "Common Thread" section (~2 short paragraphs) explaining the unifying methodological move across all three papers: **bypass the prediction step, output the decision directly, optimize the financial objective end-to-end via gradient descent**. This is what makes the showcase cohesive.
- Sidebar: data status (last refresh date for the parquet snapshot), GitHub repo link, optional FMP key input for the "refresh data" feature.

---

### Page 1 — Momentum

**Paper:** Lim, Zohren, Roberts (2019). *"Enhancing Time Series Momentum Strategies Using Deep Neural Networks."* Journal of Financial Data Science. [arXiv:1904.04912](https://arxiv.org/abs/1904.04912)

**Elevator pitch:** *"Instead of forecasting returns then sizing positions, train a neural network end-to-end to directly output position sizes that maximize Sharpe ratio."*

**Scope:** This page replicates **Exhibits 2, 3, 4, and 5** from the paper as faithfully as possible given available data, using the developer's CME futures dataset rather than the paper's Pinnacle CLC dataset. The page is the most rigorous of the three (depth of replication > Portfolio and Order Book pages).

**Deep-learning objective: Sharpe loss only.** The paper compares four loss functions (Sharpe, Average Returns, MSE, Binary cross-entropy) and concludes Sharpe wins. This project **does not replicate that comparison** — both MLP and LSTM are trained only with Sharpe loss, because the showcase narrative is risk-adjusted return, not raw return-maximization or classification accuracy. Practical consequences:
- 2 pretrained checkpoints (MLP-Sharpe, LSTM-Sharpe), not 8
- No loss-function picker in the UI
- Exhibit-4 collapses from a 2×2 grid (one panel per loss) to a single panel (cumulative returns, all Sharpe-loss strategies overlaid — corresponds to panel (a) of the paper's Exhibit 4)
- Exhibits 2 and 3 retain their structure but with fewer rows (Reference Models + 2 deep models, not 11 total)
- No need to extend `src/losses.py` beyond the existing `Neg_Sharpe` / `SharpeLoss`

**Data:** `data/cme_futures.parquet` — ~88 ratio-adjusted continuous futures contracts spanning commodities, fixed income, equities, FX. Produced by `scripts/fetch_futures.py` (developer-local, see §4.1). When the parquet is absent, page degrades gracefully to single-asset toy mode using `data/aapl.csv`.

**Sidebar controls:**
- Asset-set picker: **All futures** (full Exhibit replication, default) | **Subset by class** (commodities only / FX only / etc.) | **Single asset** (deep-dive on one contract)
- Volatility scaling toggle (default: **scaled to σ_target = 15%**, controlling the switch between Exhibit 2 view and Exhibit 3 view, matching the paper's value)
- Volatility lookback (EWMA span, default 60 days)
- Backtest date range slider (default: matches paper's 1995–2015 test window, extendable to today)
- Deep-model selector: **MLP** (default) | **LSTM** | **Both overlaid**

#### Tab 1 — Problem & Data

- One paragraph: the time-series momentum problem (Moskowitz et al. 2012) and what the Lim et al. paper contributes (end-to-end position sizing, Sharpe-optimized loss)
- Universe summary: count of contracts by asset class, average inter-contract correlation, coverage timeline (Gantt-style chart showing which contracts have data when)
- Selected-asset price chart and return distribution (when single-asset mode is selected)

#### Tab 2 — Reference Models

The "reference" baselines from Exhibit 2, computed without training:

| Signal | Definition | Source |
|---|---|---|
| **Long Only** | X_t = 1 (always long, vol-scaled per σ_target) | Buy-and-hold benchmark |
| **Sgn(Returns)** | X_t = sgn(r_{t-252,t}) — sign of past-year return | Moskowitz, Ooi, Pedersen (2012) |
| **MACD** | Volatility-normalized MACD ensemble across (8/24, 16/48, 32/96) timescales, averaged | Baz et al. (2015) |

**Implementation note for Claude Code:** Mirror the signal construction from `/QIS_Commodities` (see §4.5) where the developer's base trend-following strategies already exist. Do not reinvent the calculation; reuse conventions.

**Outputs:**
- Overlaid cumulative-return curves on log scale, with and without σ_target = 15% volatility scaling (toggle from sidebar)
- Metrics summary inline (full table is in Tab 4)

#### Tab 3 — Deep Method (Lim et al. 2019)

The two deep architectures from the paper (WaveNet excluded per developer scope), **both trained with Sharpe loss only**, yielding **2 pretrained checkpoints** in `data/pretrained/`:

| Architecture | Loss | Checkpoint file |
|---|---|---|
| **MLP** (2-layer, hidden size ~20) | Sharpe | `mlp_sharpe.pt` |
| **LSTM** (single-layer, hidden size ~10) | Sharpe | `lstm_sharpe.pt` |

**Inputs (per asset, time t):**
- Normalized returns over past 1d, 1m, 3m, 6m, 1y (5 features)
- MACD indicators at three timescales (3 features)
- → 8 features per asset per timestep, fed in via the user's existing `MyDataset` (§13.0)

**Loss function:** Sharpe only. Use `src/losses.py` `SharpeLoss` as-is (§13.0). The paper's other losses (Average Returns, MSE, Binary) are intentionally not implemented — see "Deep-learning objective" note at the top of this page.

**Controls in this tab:**
- Architecture selector (MLP / LSTM / Both overlaid) — drives which checkpoint is loaded
- "Train your own (tiny subsample)" opt-in button — HF CPU budget ~30 sec, illustrative only
- Model-card badge with training metadata (date, dataset range, σ_target used, achieved test Sharpe)

**Outputs:**
- Equity curve for the selected architecture, with vs without vol-scaling toggle
- Position-over-time plot for a selected futures contract (shows the continuous position-sizing that distinguishes Sharpe-loss DMNs from sign-based reference signals)

#### Tab 4 — Exhibits 2 / 3 / 4 / 5 Replication

The headline tab. Each sub-tab is one exhibit.

**Sub-tab 4A — Exhibit 2 (Raw Signal Outputs):**
A performance metrics table with the same shape as Exhibit 2 of the paper:
- Rows: Long Only, Sgn(Returns), MACD, MLP-Sharpe, LSTM-Sharpe (5 rows — paper's other-loss rows omitted per scope)
- Columns: E[Return], Vol, Downside Deviation, MDD, Sharpe, Sortino, Calmar, % +ve Returns, Ave. P / Ave. L
- All metrics annualized
- Bold the best value per column
- Computed **without** volatility scaling
- Use Streamlit's `st.dataframe` with `column_config` for number formatting

**Sub-tab 4B — Exhibit 3 (Rescaled to σ_target = 15%):**
Same 5-row table as 4A but all strategies rescaled to the target volatility at the portfolio level. This is what enables apples-to-apples comparison since raw signals have very different volatility levels.

**Sub-tab 4C — Exhibit 4 (Cumulative Returns):**
A single Plotly cumulative-return chart on log scale, rescaled to σ_target, overlaying all 5 strategies: Long Only, Sgn(Returns), MACD, MLP-Sharpe, LSTM-Sharpe. Corresponds to panel (a) of the paper's Exhibit 4 (Sharpe Ratio loss). The paper's other three panels (b/c/d for Avg Returns, MSE, Binary losses) are not reproduced — they require training the other loss variants, which is out of scope.

**Sub-tab 4D — Exhibit 5 (Performance Across Individual Assets):**
Three Plotly box plots, computed by running each strategy on each of the ~88 futures contracts individually and aggregating:
- Sharpe ratio per asset, by strategy
- Average returns per asset, by strategy
- Volatility per asset, by strategy

X-axis: strategy name (5 boxes per chart: Long Only, Sgn(Returns), MACD, MLP-Sharpe, LSTM-Sharpe). Y-axis: per-asset metric. This makes the rigor of the deep methods (tighter IQR, fewer outliers) visually unmistakable — the headline finding of the paper.

**Sub-tab 4E (optional, Phase 4 stretch) — Transaction Cost Sensitivity:**
Reproduce Exhibit 7 of the paper: bar chart of Sharpe ratio as a function of cost rate (0, 0.5, 1, 2, 3, 4, 5 bps) for each of the 5 strategies. Highlights the breakeven cost level where DL strategies still outperform reference.

**Math panel:** Sharpe-loss derivation (already in `src/losses.py`); tanh / Softsign output mapping to (-1, 1) positions; volatility-scaling formula (Eq. 1 of the paper); brief description of why direct outputs beat regression + sign (the paper's argument).

**Code panel:** Show `class MLP` and `class LSTM` from `src/models/deep_momentum.py` (to be ported); `SharpeLoss` reference from `src/losses.py`.

**Source notebooks and references:**
- `notebooks/reference/01_time_series_momentum.ipynb` — reference signal construction
- `notebooks/reference/03_deep_momentum_strategy.ipynb` — reference MLP, training loop, feature construction
- `/QIS_Commodities` — developer's existing futures-data-loading and base-strategy conventions to mirror
- Lim et al. 2019 paper — full mathematical reference for losses, vol scaling, and exhibits


---

### Page 2 — Portfolio

**Paper:** Zhang, Zohren, Roberts (2020). *"Deep Learning for Portfolio Optimization."* [arXiv:2005.13665](https://arxiv.org/abs/2005.13665)

**Elevator pitch:** *"A long-only portfolio with softmax output, trained by gradient ascent to maximize Sharpe directly — no covariance matrix, no expected-return forecast."*

**Scope:** This page replicates **Table 1** and **Figure 3** of the paper as primary deliverables, plus a stretch attempt at **Figure 6** (gradient-based feature sensitivity analysis). All three are run across three universes to demonstrate the method's robustness.

**Deep-learning objective:** Sharpe loss (negative Sharpe minimized via Adam), same as the Momentum page. Same `src/losses.py` is reused.

**Universes (sidebar picker):**

| Universe | Members | Source |
|---|---|---|
| **ETF basket** (default, paper-faithful) | VTI, AGG, DBC, VIXY | FMP-fetched parquet (VIXY substituted for non-tradable VIX index per common practice) |
| **20-stock** | The 20 tickers from existing `portfolio_data.csv` — **AAPL, ABT, AEP, AXP, BAC, CI, GD, GE, HON, MMM, MO, MRK, NEM, NKE, NSC, PFE, PG, PTC, SNA, SO** — sector-spread across tech, healthcare, utilities, financials, industrials, materials, consumer staples, consumer discretionary, energy, communications. Existing CSV covers 1992-11-25 through 2022-11-23 with Close/Volume/Return columns. **FMP fetch extends the history through today** so the test window includes 2023–2025 regimes. | FMP-fetched parquet, ticker list pulled directly from CSV header |
| **100-stock** | 100 sector-balanced S&P 500 names, historical constituents to avoid survivorship bias | FMP-fetched parquet |

**Sidebar controls:**
- Universe picker (3 options above)
- Transaction-cost rate selector: `C ∈ {0.01%, 0.10%}` — controls which row of Table 1 is being viewed
- Volatility scaling toggle (default ON, σ_target = 10% per the paper's Table 1)
- Rolling estimation window for classical methods (default 50 days, matches paper §4.2)
- Backtest date range slider

#### Tab 1 — Problem & Data

- One paragraph: portfolio optimization problem (Markowitz 1952) and what Zhang et al. 2020 contributes (end-to-end softmax-output network, gradient ascent on Sharpe, no return forecasting)
- **Rolling-correlation heatmap** of the selected universe — cf. Figure 1 of the paper for the ETF case; analogous heatmap for 20-stock and 100-stock universes
- Summary stats: returns, vols, pairwise correlation distribution, sector breakdown (for stock universes)

#### Tab 2 — Classical Benchmarks

The full benchmark set from Table 1 of the paper. Which subset is available depends on the universe:

| Method | ETF basket | 20-stock | 100-stock |
|---|---|---|---|
| **Allocation 1** (25/25/25/25, shares/bonds/commodities/vol) | ✓ | — | — |
| **Allocation 2** (50/10/20/20) | ✓ | — | — |
| **Allocation 3** (10/50/20/20) | ✓ | — | — |
| **Allocation 4** (40/40/10/10) | ✓ | — | — |
| **Equal Weight** | ✓ | ✓ | ✓ |
| **Minimum Variance** (Markowitz 1952) — `ShrunkCovariance` per existing notebook | ✓ | ✓ | ✓ |
| **Maximum Diversification** (Choueifaty & Coignard 2008) | ✓ | ✓ | ✓ |
| **Diversity-Weighted Portfolio (DWP)** — Stochastic Portfolio Theory (Samo & Vervuurt 2016) | ✓ | ✓ | ✓ |

Fixed allocations 1–4 only apply to the 4-ETF basket since they are pre-specified per asset class.

**DWP implementation note:** requires market-cap data (`sharesOutstanding × close`). FMP provides both per-symbol. `scripts/fetch_data.py` must pull shares-outstanding alongside prices to support DWP. If unavailable, the brief flags DWP as skip-with-warning rather than failing the build.

**Outputs:**
- Multi-select toggle for which methods to display
- Overlaid cumulative-return curves (controlled by sidebar's vol-scaling toggle and cost-rate selector)
- Weight-evolution heatmap for the selected classical method (assets × time)
- Metrics summary inline (full table is in Tab 4)

#### Tab 3 — Deep Method (Zhang et al. 2020)

The paper's deep model: MLP with softmax output, trained to maximize Sharpe directly.

**Three pretrained checkpoints** (one per universe):

| Checkpoint | Universe | Approx. params |
|---|---|---|
| `deep_portfolio_etfs.pt` | ETF basket | ~5k |
| `deep_portfolio_20stock.pt` | 20-stock | ~10k |
| `deep_portfolio_100stock.pt` | 100-stock | ~25k |

**Inputs (per asset, time t):** close price + daily return, with T-day lookback (default T=50 per paper §4.3). For N assets the input shape is (T, 2N) — concatenated across all assets.

**Architecture:** MLP with softmax output (matches `02_deep_portfolio_optimization.ipynb`). LSTM variant (paper's actual choice) deferred to Phase 4 stretch — the MLP from the GitHub notebook is sufficient for Table 1 / Figure 3 replication.

**Loss:** `Neg_Sharpe` from `src/losses.py` (§13.0) used as-is.

**Controls in this tab:**
- Pretrained model loaded for the selected universe; model-card badge shown
- "Train your own (tiny subsample)" opt-in button — HF CPU budget ~30 sec, illustrative only

**Outputs:**
- Deep Portfolio equity curve overlaid with the best classical benchmark
- Weight-over-time heatmap (same axes as Tab 2 for side-by-side visual comparison)

#### Tab 4 — Table 1 / Figure 3 / Figure 6 Replication

The headline tab.

**Sub-tab 4A — Table 1 (Performance Metrics):**

A performance metrics table with the same shape as Table 1 of the paper:
- Rows: all in-scope benchmark methods for the selected universe + Deep Portfolio (DLS in paper notation)
- Columns: `E(R)`, `Std(R)`, `Sharpe`, `DD(R)` (downside deviation), `Sortino`, `MDD`, `% of +Ret`, `Ave. P / Ave. L`
- All metrics annualized
- Bold the best value per column
- **Three rows of the table** (governed by sidebar's cost-rate and vol-scaling toggles):
  1. No volatility scaling, C = 0.01%
  2. Vol scaling (σ_target = 0.10), C = 0.01%
  3. Vol scaling (σ_target = 0.10), C = 0.10%

Use Streamlit's `st.dataframe` with `column_config`. For the ETF basket, the table should be visually similar to the paper's Table 1.

**Sub-tab 4B — Figure 3 (Cumulative Returns, 3 panels):**

Three Plotly cumulative-return charts on log scale, side-by-side, matching the paper's Figure 3:
- **Left**: no vol scaling, C = 0.01%
- **Middle**: vol scaling (σ_target = 0.10), C = 0.01%
- **Right**: vol scaling (σ_target = 0.10), C = 0.10%

Overlay all in-scope methods. The cost selector and vol-scaling toggle in the sidebar control which panel is "active" / highlighted; the user can see all three panels simultaneously for the at-a-glance comparison the paper emphasizes.

**Sub-tab 4C — Figure 6 (Sensitivity Analysis) [STRETCH — Phase 4]:**

Reproduce the gradient-based feature sensitivity heatmap from §4.6 of the paper:

$$S_i = \frac{|\partial L / \partial x_i|}{\max_j |\partial L / \partial x_j|}$$

For each input feature `x_i` and each time step, compute the absolute normalized sensitivity of the Sharpe loss w.r.t. that feature, plot as a 2D heatmap (features × time). For the ETF basket with T=50 lookback and 2 features per asset, there are 4 × 50 × 2 = 400 features per input window.

This requires:
1. Forward pass through the trained Deep Portfolio model
2. Backward pass to compute input gradients (`torch.autograd.grad`)
3. Normalize by max absolute gradient
4. Aggregate across the test set, plot heatmap with features on Y-axis grouped by asset class

**Implementation note:** non-trivial but contained. Recommend deferring to Phase 4 to keep Phase 2 acceptance criteria tight.

**Sub-tab 4D — COVID Stress Test [STRETCH — Phase 4, optional]:**

Reproduce Figure 5 of the paper — Deep Portfolio's allocations during Jan–Apr 2020 across the four asset classes (stocks/bonds/commodities/volatility). Demonstrates the model's behavior during crisis: rotation into bonds before the crash, position-shrinking from vol scaling during the bond drop. Only applicable to the ETF basket.

**Was previously specified in scope; demoted to stretch in the current revision** because the user's latest spec focused on Table 1, Figure 3, and Figure 6. Easy to keep in Phase 2 if desired — flag in §12.

**Math panel:** Softmax output for long-only weight constraint (`w_i = exp(w̃_i) / Σ exp(w̃_j)`, satisfies `w_i ∈ [0,1]` and `Σ w_i = 1`); gradient ascent on the differentiable Sharpe loss; why no expected-return forecast is needed; volatility scaling formula from §4.4 of the paper (Eq. 7).

**Code panel:** `class MLP` and `Neg_Sharpe` from `02_deep_portfolio_optimization.ipynb`; `max_diversification_weights` and `min_variance_weights` from `01_classical_portfolio_optimization.ipynb`.

**Source notebooks and references:**
- `notebooks/reference/01_classical_portfolio_optimization.ipynb` — reference classical methods
- `notebooks/reference/02_deep_portfolio_optimization.ipynb` — reference MLP, softmax, Neg_Sharpe
- Zhang, Zohren, Roberts (2020) paper — Table 1, Figures 1/3/5/6 are direct replication targets

---

### Page 3 — Order Book Prediction

**Paper:** Zhang, Zohren, Roberts (2019). *"DeepLOB: Deep Convolutional Neural Networks for Limit Order Books."* IEEE Transactions on Signal Processing. [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)

**Elevator pitch:** *"A CNN + Inception + LSTM architecture that learns universal features of limit order book microstructure, transferable across instruments."*

**Scope:** This page replicates **Table I (Setup 1)** and **Table II (Setup 2)** of the DeepLOB paper, both computed on the FI-2010 benchmark dataset. These are the paper's headline results — accuracy, precision, recall, and F1 for the 3-class mid-price-movement classification task at multiple prediction horizons. This page is paper-replication-grade, on the same footing as the Momentum and Portfolio pages.

**Data:** `data/lob_fi2010.parquet` — the canonical Ntakaris et al. (2017) benchmark, downloaded from Kaggle (`praanj/limit-orderbook-data`) via `scripts/fetch_lob_fi2010.py` (see §4.1). ~4M samples, 5 Nasdaq Nordic stocks, 10 trading days (June 2010), 10 LOB levels each side, z-score normalized, pre-labeled with 3-class targets at horizons k ∈ {10, 20, 30, 50, 100} ticks.

**The bundled `data/limit_order_book_data.csv` is NOT used by the live app.** It was a smoke-test sample for the original GitHub notebook; FI-2010 supersedes it entirely. The CSV stays in `notebooks/reference/data/` (next to the reference notebook) for anyone running the original notebook locally, but the Streamlit page reads only from `lob_fi2010.parquet`. If the parquet is absent, the page shows a clear setup instruction rather than degrading to a toy mode.

**Task framing:** 3-class classification (down / stationary / up), matching the paper exactly. Cross-entropy loss, smoothed labels per Eq. 4 of the paper, T=100 lookback, k=10 prediction horizon by default (extendable to other k via sidebar).

**Neither FMP nor CME is involved on this page.**

**Sidebar controls:**
- Evaluation setup: **Setup 1 (anchored forward, 9 folds)** | **Setup 2 (7-day train / 3-day test)** — drives which of Table I or Table II is being viewed
- Prediction horizon k ∈ {10, 20, 30, 50, 100} ticks — drives which sub-row of the relevant table is highlighted
- Time-window slider over the test set (for the live demo)
- Stock filter (which of the 5 Nasdaq Nordic stocks to inspect in Tab 1)

#### Tab 1 — Problem & Data
- One paragraph: what a limit order book is, why mid-price prediction matters, what FI-2010 is and why it's the canonical benchmark
- LOB snapshot visualization at the selected timestamp: 10 levels each side, prices on x-axis, volumes as bar heights, colored bid (blue) vs ask (orange) — mirrors Figure 1 of the paper
- Class-balance summary: count of {down, stationary, up} labels per stock and per prediction horizon, shown as a stacked bar chart so the user can see how class balance shifts with k
- Smoothed-label visualization mirroring Figure 2 of the paper: mid-price overlaid with green (+1) and red (-1) shaded regions per Eq. 4

#### Tab 2 — Classical Benchmarks

The paper's published baselines from Tables I and II:
- **Ridge Regression (RR)** — Ntakaris et al. 2018 baseline
- **Single-Layer Feedforward Network (SLFN)** — Ntakaris et al. 2018 baseline
- **Linear Discriminant Analysis (LDA)** — Tran et al. 2017
- **MDA / MTR / WMTR** — Tran et al. tensor methods
- **MCSDA** — Tran et al. multilinear class-specific
- **Bag-of-Features (BoF / N-BoF)** — Passalis et al.
- **(TABL) variants — B(TABL), C(TABL)** — Tran et al. attention-augmented bilinear
- **SVM** — Setup 2 only (Tsantekidis et al.)
- **MLP** — Setup 2 only
- **CNN-I / CNN-II** — Setup 2 only (Tsantekidis et al.)
- **LSTM** — Setup 2 only

**Implementation note:** Many of these baselines are reported only in the paper, not re-implemented. The brief covers two practical paths:
1. **Implement a representative subset** locally (e.g., LDA, MLP, plain CNN) and run them on FI-2010 to get fresh numbers
2. **Cite paper-reported values** for the rest (RR, SLFN, BoF, TABL, MCSDA), with a clearly-marked "(paper-reported)" badge in the table

Recommended: implement 3–4 representative baselines, cite the rest from the paper. Don't try to reimplement every baseline — that's a separate project. The Tab 4 master tables clearly distinguish "(reproduced here)" from "(paper-reported)" cells.

**Outputs:**
- Pick a baseline from a dropdown, see its predictions on the live time slider
- Per-baseline metrics card (accuracy, F1, etc.)

#### Tab 3 — Deep Method (Zhang et al. 2019)

Three sub-tabs:

**Sub-tab A: Architecture Explorer**
- Interactive diagram of the model: Conv blocks → Inception module → LSTM → FC → 3-way softmax
- Click each block to see its config (kernel sizes, channels, gated tanh-sigmoid activation in the first conv block, 1×1 / 3×1 / 5×1 / max-pool branches in the Inception module)
- Pure visualization, no training

**Sub-tab B: Live Prediction Demo**
- Pretrained DeepLOB checkpoint loaded (`deeplob_fi2010.pt`)
- Scrub through the test set with the sidebar time slider
- At each timestamp t:
  - LOB snapshot (top): 10 levels each side
  - Predicted class probabilities {down, stationary, up} as a 3-bar chart
  - Realized class label
  - Running accuracy / precision / recall / F1 displayed at the bottom
- Model-card badge with training metadata (Setup 1 vs Setup 2, k value, train days, achieved test F1)

**Sub-tab C: Train Your Own (advanced)**
- Sliders for T, batch size, learning rate, patience, k (prediction horizon)
- Opt-in training button on a tiny subsample (~30 sec budget on HF CPU)
- Live loss curve

#### Tab 4 — Table I / Table II Replication

The headline tab.

**Sub-tab 4A — Table I (Setup 1: Anchored Forward, 9 Folds):**

Setup 1 is the paper's chronological cross-validation: train on first i days, test on day (i+1), for i = 1 ... 9. Mean of accuracy / precision / recall / F1 across the 9 folds.

- Rows: all in-scope baselines (reproduced + paper-cited) + DeepLOB
- Columns: Accuracy %, Precision %, Recall %, F1 %
- Three horizon blocks: k=10, k=50, k=100 (matching paper's Table I structure)
- "(reproduced here)" vs "(paper-reported)" badge per row
- Bold the best F1 per horizon block

**Sub-tab 4B — Table II (Setup 2: 7-Day Train / 3-Day Test):**

Setup 2 uses the first 7 days as training, last 3 days as test (a single split, not cross-validation). This is the more conventional deep-learning eval and the paper reports stronger numbers here.

- Same columns as Table I
- Rows include the Setup-2-only baselines (SVM, MLP, CNN-I, CNN-II, LSTM, TABL variants) plus DeepLOB
- Three horizon blocks: k=10, k=20, k=50
- Same badging and bolding conventions as 4A

**Sub-tab 4C — Confusion Matrices:**

3×3 confusion matrices side-by-side for the top 3–4 methods (per the user's selection). Helps visualize where errors concentrate — class imbalance and the difficulty of the stationary class are visible immediately.

**Sub-tab 4D — Per-Class Performance Curves:**

Precision-recall curves and per-class F1 vs k (prediction horizon) for DeepLOB and the top classical baselines. Shows how the gap between DeepLOB and baselines widens or narrows as the horizon increases.

**Sub-tab 4E (Phase 4 stretch) — LIME Sensitivity:**

Reproduce Figure 9 of the paper: LIME-based explanations showing which LOB regions drive specific predictions. Compares DeepLOB to the simpler CNN-I baseline; demonstrates that DeepLOB uses more of the input.

**Key observation callout (above Tab 4):**

"DeepLOB achieves F1 = [X]% on Setup 2 (k=10), versus [Y]% for the best paper-reported baseline (C(TABL)). Reproduced here from FI-2010, matching Table II of Zhang et al. 2019 within [Z] percentage points."

**Math panel:** Mid-price formula (Eq. 1); the smoothed labeling convention from Eq. 4 (used here, not Eq. 3); the gated activation in the first conv block; the Inception module's parallel-filter design and Network-in-Network 1×1 convolutions; why volume-imbalance features emerge naturally from the early conv layers (cf. Eq. 7 of the paper, the micro-price interpretation).

**Code panel:** `class deeplob` from `02_predictive_signal_lob.ipynb`, ported to `src/models/deeplob.py` and adapted from regression head to classification head (3-way softmax output).

**Source notebook and references:**
- `notebooks/reference/02_predictive_signal_lob.ipynb` — reference architecture (note: original notebook was regression with MSE loss on a sample CSV; this page uses classification with cross-entropy on FI-2010, which is the paper's actual task)
- Zhang, Zohren, Roberts (2019) — Tables I and II are direct replication targets
- Ntakaris et al. (2017) — original FI-2010 paper, defines the labeling protocol and the benchmark itself

---

### Page Footer (every page)

- Original paper citation in formatted form
- arXiv badge linking to the PDF
- Copy-to-clipboard BibTeX block
- "Source code for this page →" GitHub link (deep-linked to the relevant `pages/*.py` file)
- "Reference notebook →" GitHub link (deep-linked to the relevant `notebooks/reference/*.ipynb`)

## 7. Models to Pretrain

All training happens on **Modal** serverless GPU containers (see §2.2). The repo splits training concerns into two layers — there is no third "notebook orchestrator" layer because the Modal decorator at the top of each training script does that job:

1. **Model definitions** — `src/models/*.py`. Device-agnostic, pure PyTorch `nn.Module`s.
2. **Training scripts** — `src/training/train_*.py`. The bottom of each file is plain device-agnostic Python (testable locally on CPU with `python -m src.training.train_*`). The top of each file declares the Modal `App`, container `Image`, `Volume`, and `@app.function(gpu=...)` decorator so the same file runs on Modal with `modal run src/training/train_*.py`.

### 7.1 Required checkpoints

| Checkpoint | Page | Source ref | Trained on | Params | Est. wall-clock (Modal T4) | Est. compute cost |
|---|---|---|---|---|---|---|
| `mlp_sharpe.pt` | Momentum | `03_deep_momentum_strategy.ipynb` + `/QIS_Commodities` | CME futures, 1990–2010 train, 2011–today test | ~10k | ~10 min | ~$0.10 |
| `lstm_sharpe.pt` | Momentum | same | same | ~5k | ~15 min | ~$0.15 |
| `deep_portfolio_etfs.pt` | Portfolio | `02_deep_portfolio_optimization.ipynb` | ETF basket (VTI/AGG/DBC/VIXY), whatever history available, 60/20/20 chronological split | ~5k | ~3 min | ~$0.03 |
| `deep_portfolio_20stock.pt` | Portfolio | `02_deep_portfolio_optimization.ipynb` | 20-stock universe, whatever history available, 60/20/20 split | ~10k | ~5 min | ~$0.05 |
| `deep_portfolio_100stock.pt` | Portfolio | `02_deep_portfolio_optimization.ipynb` | 100-stock universe, whatever history available, 60/20/20 split | ~25k | ~10 min | ~$0.10 |
| `deeplob_fi2010.pt` | Order Book | `02_predictive_signal_lob.ipynb` + DeepLOB paper setup 2 | FI-2010 benchmark, 7 days train / 3 days test (paper's Setup 2) | ~30–60 min on T4; ~15 min on A10G | ~$0.50 (T4) or ~$0.40 (A10G) | |

**Total compute budget:** under **$2 of Modal credit** for a clean training of all six checkpoints. Modal bills per second and the container tears down on exit; you only pay for the wall-clock the training function runs.

**Single-script multi-checkpoint:** `src/training/train_deep_momentum.py` produces both `mlp_sharpe.pt` and `lstm_sharpe.pt` via a `--arch ∈ {MLP, LSTM}` CLI flag; `src/training/train_deep_portfolio.py` produces all three Portfolio checkpoints via a `--universe ∈ {etfs, 20stock, 100stock}` flag. Total ~3 distinct training scripts under `src/training/`.

### 7.2 Modal script template

Every training script in `src/training/` follows the same structure. The decorator block at the top is the Modal-specific glue; everything below `def train(...)` is plain device-agnostic Python:

```python
# src/training/train_deep_momentum.py
import os
from pathlib import Path
import modal

# ---- Modal scaffolding (top of file) ----
app = modal.App("deep-finance-train-momentum")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-train.txt")
    .add_local_python_source("src")  # mounts src/ into the container at /root/src
)

volume = modal.Volume.from_name("deep-finance-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",                                # or "A10G" / "A100"
    volumes={"/data": volume},
    timeout=3600,                            # 1-hour cap
    secrets=[modal.Secret.from_name("deep-finance-train-env")],  # optional
)
def train_remote(arch: str = "MLP", resume_from: str | None = None) -> dict:
    import torch
    from src.training.train_deep_momentum import train  # the device-agnostic body
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, metrics = train(
        data_dir=Path("/data"),
        arch=arch,
        device=device,
        checkpoint_dir=Path("/data/pretrained"),
        resume_from=resume_from,
    )
    volume.commit()                          # persist the .pt + JSON sidecar to the Volume
    return metrics

@app.local_entrypoint()
def main(arch: str = "MLP", resume_from: str | None = None):
    metrics = train_remote.remote(arch=arch, resume_from=resume_from)
    print(f"[train_deep_momentum] arch={arch} metrics={metrics}")
    print(f"To pull the checkpoint locally:")
    print(f"  modal volume get deep-finance-data /pretrained/{arch.lower()}_sharpe.pt "
          f"./data/pretrained/{arch.lower()}_sharpe.pt")

# ---- Device-agnostic training body (bottom of file) ----
def train(data_dir, arch, device, checkpoint_dir, resume_from=None):
    """Plain Python training loop. Importable + callable on CPU for unit tests."""
    # ... model construction, dataloader, optimizer.step() loop ...
    # ... periodic checkpoint to checkpoint_dir, resume_from support ...
    return state_dict, metrics
```

**Two invocations of the same file**:

```bash
# Local CPU smoke test (small subset of data, fast)
python -m src.training.train_deep_momentum --arch MLP

# Real training on Modal GPU
modal run src/training/train_deep_momentum.py --arch MLP
```

**One-time Modal setup** (developer's machine):

```bash
pip install modal                           # add to your dev environment, NOT to requirements.txt
modal token new                             # opens browser once, writes ~/.modal.toml
modal volume create deep-finance-data       # one-time Volume creation
modal volume put deep-finance-data \
       ~/data_lake/deep-finance/ /          # seed the Volume with current parquets
modal secret create deep-finance-train-env \
       FMP_API_KEY=$FMP_API_KEY             # only if any training step needs FMP at runtime (rare)
```

**Pulling checkpoints back to the repo**:

```bash
modal volume get deep-finance-data /pretrained ./data/pretrained/
git add data/pretrained/*.pt data/pretrained/*.json
git commit -m "feat: train mlp_sharpe + lstm_sharpe (Modal T4)"
git push                                    # GitHub Action syncs to HF Space
```

Key conventions:
- **Resumability:** Training loops save intermediate `<name>.pt` to the Modal Volume every N epochs and accept a `resume_from` argument so a container preemption doesn't lose work. The Volume persists across container instances; only it survives.
- **Reproducibility:** Set seeds at the top of every training script (numpy, torch, python `random`).
- **Data on the Modal Volume, not re-fetched:** Upload `etf_basket.parquet`, `sp500_20.parquet`, `sp500_100.parquet`, `cme_futures.parquet`, `lob_fi2010.parquet` to the `deep-finance-data` Volume once (with `modal volume put`). Training functions read from `/data/` (the Volume mount point) — they do not call FMP / Kaggle / read from the developer's local data lake.
- **Image bake:** the `Image` definition pins Python 3.11 and installs `requirements-train.txt`. Modal rebuilds the image only when the `Image` chain or `requirements-train.txt` changes (Modal hashes the image spec) — typical training runs reuse the cached image and start in <30 seconds.

### 7.3 Checkpoint metadata (sidecar JSON)

Each `.pt` file is accompanied by a `<name>.json` written by the training function:

```json
{
  "trained_on": "2026-05-20",
  "trained_with": "Modal T4 (image hash: sha256:abc123…)",
  "torch_version": "2.x.x",
  "modal_app": "deep-finance-train-momentum",
  "data_range": {"start": "2014-01-01", "end": "2020-12-31"},
  "split": {"train": 0.6, "val": 0.2, "test": 0.2},
  "hyperparameters": {"lr": 1e-4, "batch_size": 128, "epochs_trained": 87, "patience": 50},
  "final_metrics": {"val_loss": -1.42, "test_sharpe": 1.95, "test_mdd": 0.12},
  "git_commit": "abc1234"
}
```

The live app reads this and renders a "model card" badge on each DL page (training date, dataset range, achieved metrics). The `trained_with` field MUST name the Modal GPU type and image hash so future debugging can reproduce the exact environment (Constitution Principle III).

### 7.4 What does NOT run on Modal

These scripts run locally on the developer's machine, not on Modal:

- `scripts/fetch_data.py` — uses the developer's FMP key from `.env`; CPU-only; outputs to `${DEEP_FINANCE_DATA_DIR:-./data/}*.parquet`
- `scripts/run_backtests.py` — CPU-only; loads pretrained checkpoints and runs backtests; outputs to `data/backtests/`
- Anything called by the live Streamlit app at runtime
- `scripts/fetch_futures.py` and `scripts/fetch_lob_fi2010.py` — local I/O against the developer's data lake / Kaggle credentials

## 8. Backtest Precomputation

With the three-page structure, each page's **Tab 4 (Comparison)** is the consumer of precomputed results. Live recomputation of Tabs 2 and 3 is fast enough to do on demand (handfuls of vectorized pandas ops, sub-second model inference), but the master overlay in Tab 4 should be instant.

`scripts/run_backtests.py`:
1. Loads all data and pretrained models
2. Runs each strategy on the full common test period (e.g., 2020–today, with 2014–2019 used for training where applicable)
3. Computes daily P&L series + summary metrics
4. Writes one parquet per page:
   - `data/backtests/momentum_results.parquet`
   - `data/backtests/portfolio_results.parquet`
   - `data/backtests/order_book_results.parquet`

Each parquet has columns: `date`, `strategy`, `daily_return`, `cum_return`, plus a sidecar metrics file (`*_metrics.parquet`) with the summary table.

**Strategies covered per page:**

| Page | Strategies |
|---|---|
| Momentum | Long Only; Sgn(Returns); MACD ensemble; MLP-Sharpe; LSTM-Sharpe — **all computed both with and without σ_target = 15% vol scaling** (so the parquet has 10 strategy rows total) |
| Portfolio | Per universe (ETF / 20-stock / 100-stock), all in-scope benchmarks: Equal Weight, Min Variance, Max Diversification, DWP, Deep Portfolio. ETF basket additionally includes Allocations 1–4. Each strategy computed under **3 conditions** matching Table 1: (no vol scaling, C=0.01%) / (vol scaling, C=0.01%) / (vol scaling, C=0.10%). Output parquet has columns `universe`, `condition`, `strategy`, `date`, `daily_return`. |
| Order Book | All on FI-2010 benchmark: LDA, MLP, CNN baselines (re-implemented) + RR, SLFN, BoF, TABL, MCSDA (paper-cited) + DeepLOB. Computed per (setup ∈ {1, 2}, k ∈ {10, 20, 30, 50, 100}). Output parquet schema: `setup`, `k`, `method`, `accuracy`, `precision`, `recall`, `f1`, `reproduced`. |

The Momentum parquet additionally stores **per-asset daily returns** per strategy (long-format with `date`, `strategy`, `asset`, `daily_return` columns) so that Exhibit 5's per-asset box plots can be computed in the page without re-running the model 88 times.

## 9. Pedagogical / Showcase Framing

- **Tone:** Research showcase. Lead with the paper, lead with the contribution. Avoid textbook chapter numbers in titles — they're internal-only references.
- **Each page header format:** Method name (large), then `Based on [Authors] (Year), [Venue]. [arXiv link]` (smaller, muted color).
- **Math panels:** Use `st.latex` for equations. Each equation gets a one-sentence intuition below it.
- **Code panels:** Use `st.code` with Python syntax highlighting. Limit to the most important snippet per page (~30 lines max).
- **Model card badges:** Small box on each DL page showing "Trained on: …, Achieved Sharpe: …, Last updated: …".
- **Plotly theming:** Consistent dark or light theme across pages. Use a finance-appropriate palette (avoid garish colors).
- **Responsive layout:** Use `st.columns` so the app looks decent on tablets.

## 10. Streamlit-Specific Best Practices

- `@st.cache_data` for: parquet loaders, FMP fetch results, backtest results.
- `@st.cache_resource` for: PyTorch model loading.
- `st.session_state` for: user-toggled controls that need to persist across re-renders.
- Use `st.spinner` and `st.progress` for anything > 200ms.
- Set page config (title, icon, layout) at the top of every page.
- One `.streamlit/config.toml` defining the theme globally.

## 11. Build Phases & Acceptance Criteria

The three-page structure changes the natural phase boundaries. Instead of splitting work by "classical pages first, deep pages second," each phase now delivers **one complete page end-to-end** (Tab 1 + Tab 2 + Tab 3 + Tab 4), so the showcase always has at least one fully functional story even when partially built.

### Phase 0 — Skeleton & data
**Deliverables:**
- Repo skeleton matching §5
- User's existing utility files copied into `src/` as-is: `losses.py`, `early_stopper.py`, `torch_data.py`, `__init__.py` — **do not reimplement these** (see §13)
- User's CSVs copied into the repo: `data/aapl.csv` and `data/portfolio_data.csv` (live-app fallbacks); `notebooks/reference/data/limit_order_book_data.csv` (smoke-test sample for the reference notebook only, not a runtime input to the live app)
- `src/metrics.py` **extended** beyond the user's starter (which has only `annual_ret`/`annual_std`/`annual_sharpe`) to add: `sortino`, `calmar`, `max_drawdown`, `pct_positive_days`, `avg_win_loss_ratio`. Wrap all of these into a single `report_metrics()` that returns a dict. Keep the existing keys' names and conventions backward-compatible. (See §13.7 for formulas.)
- `src/data.py` — new file. Loads from parquet if available, falls back to bundled CSVs if not. FMP fetch logic lives here too.
- `src/universes.py` — new file. Universe definitions per §4.2.
- `scripts/fetch_data.py` — runs and produces all parquet files
- Landing page (`streamlit_app.py`) with the three paper cards and "Common Thread" section
- `requirements.txt`, `.streamlit/config.toml`, `.env.example`, `.gitignore` complete
- Empty placeholder `pages/1_*.py`, `pages/2_*.py`, `pages/3_*.py` (so multi-page nav works)

**Acceptance:**
- `streamlit run streamlit_app.py` works locally with no errors
- Landing page renders, all three nav links work (even if target pages are stubs)
- App works on a fresh clone **without** an FMP key — falls back to bundled CSVs and shows a banner noting the data source
- After running `scripts/fetch_data.py` with an FMP key, parquets exist in `data/` and the app silently switches to using them
- Unit tests for the extended `src/metrics.py` pass against a synthetic returns series (cross-check against the values reported in the source notebooks)

### Phase 1 — Page 1: Momentum (Exhibit replication)

This is the heaviest of the three page-build phases. Deliverables are expanded to cover the Exhibits 2/3/4/5 replication and the CME futures data pipeline.

**Prerequisites (developer must complete or grant before Claude Code can build):**
- Read access to `/data_lake/data_bento/cme` and `/QIS_Commodities`, OR a written-out copy of the relevant data-loading function and base trend-following strategy code committed to the repo
- Confirmation of the exact futures universe to ship in `data/cme_futures.parquet` (~88 contracts; subset by class is fine if the full set is too heavy)

**Deliverables:**
- `scripts/fetch_futures.py` — reads `/data_lake/data_bento/cme`, mirrors `/QIS_Commodities` conventions, produces `data/cme_futures.parquet`
- `src/strategies/tsmom.py` — Reference Model signals (Long Only, Sgn(Returns), MACD ensemble); conventions mirror `/QIS_Commodities`
- `src/strategies/vol_targeting.py` — volatility-scaling logic with configurable σ_target (default 15%, matching the paper)
- `src/models/deep_momentum.py` — both **MLP** and **LSTM** architectures, device-agnostic
- `src/losses.py` — use existing `Neg_Sharpe` / `SharpeLoss` as-is. **Do not add** Average Returns, MSE, or Binary loss classes (out of scope per Page 1 spec).
- `src/training/train_deep_momentum.py` — parameterized by `arch ∈ {MLP, LSTM}` with Sharpe loss fixed; declares a Modal `App` / `Image` / `Volume` / `@app.function(gpu="T4")` decorator at the top of the file (per §7.2 template); the device-agnostic `def train(...)` body runs locally on CPU for unit tests via `python -m src.training.train_deep_momentum`, and on Modal GPU for real training via `modal run src/training/train_deep_momentum.py --arch MLP` (and again with `--arch LSTM`). Intermediate checkpointing to the `deep-finance-data` Modal Volume; resume_from support
- **2 pretrained checkpoints** committed: `mlp_sharpe.pt` and `lstm_sharpe.pt`, each with JSON sidecar
- `scripts/run_backtests.py` extended to produce `data/backtests/momentum_results.parquet` containing daily returns for all 5 strategies (3 reference + MLP-Sharpe + LSTM-Sharpe) **both with and without** vol-scaling (10 rows total)
- All four tabs of the Momentum page functional, including:
  - Tab 1: universe summary chart and per-asset price drilldown
  - Tab 2: Reference Model overlay (Long Only / Sgn(Returns) / MACD)
  - Tab 3: architecture selector (MLP / LSTM / Both) with live equity curve
  - Tab 4: sub-tabs 4A (Exhibit 2 table, 5 rows), 4B (Exhibit 3 table, 5 rows), 4C (single-panel cumulative-return chart), 4D (Exhibit 5 three-box-plot panel)
- Math panel + code panel populated

**Acceptance:**
- Tab 4A and 4B reproduce the column structure of Exhibits 2 and 3 of the paper, with bolded best-per-column values, restricted to the 5 in-scope strategies
- Tab 4C shows a single cumulative-return chart on log scale with all 5 strategies overlaid (panel (a) of paper's Exhibit 4)
- Tab 4D shows three box plots aggregating per-asset metrics across the futures universe for the 5 strategies
- Sgn(Returns) and MACD reference signals reproduce the values in `/QIS_Commodities`'s existing backtests within numerical tolerance (data-pipeline parity check)
- Sharpe-loss MLP and LSTM achieve out-of-sample Sharpe ratios qualitatively consistent with the paper's Exhibits 2/3 (paper reports MLP-Sharpe ≈ 1.49 raw / 2.02 rescaled; LSTM-Sharpe ≈ 2.78 raw / 2.91 rescaled — our values will differ since we use CME not Pinnacle and a different test window, but the ordering should hold: LSTM > MLP > MACD > Sgn(Returns) > Long Only)
- Page loads in < 3 seconds on HF CPU Basic (excluding cold start). Tab switches < 500ms.
- "Train your own" button works on a tiny subsample (single futures contract, 1 year window) without OOM

### Phase 2 — Page 2: Portfolio (Table 1 + Figure 3 replication)

**Prerequisites:**
- FMP Premium key with access to: bulk historical EOD, historical S&P 500 constituents, and shares-outstanding (for DWP). Confirm before Phase 2 begins.
- Confirmed list of 100 sector-balanced S&P 500 tickers (developer to provide or accept Claude Code's proposal)

**Deliverables:**
- `scripts/fetch_data.py` produces three parquets: `etf_basket.parquet`, `sp500_20.parquet`, `sp500_100.parquet`
- `src/strategies/classical_portfolio.py` — Equal Weight, Min Variance, Max Diversification (ported), plus newly-added **Diversity-Weighted Portfolio (DWP)** and **Fixed Allocations 1–4** (ETF-only)
- `src/models/deep_portfolio.py` + `src/training/train_deep_portfolio.py` — device-agnostic; `train_deep_portfolio.py` declares the Modal `App` / `Image` / `Volume` / `@app.function(gpu="T4")` decorator at the top (per §7.2 template), parameterized by `--universe ∈ {etfs, 20stock, 100stock}` via `@app.local_entrypoint()`. Three sequential Modal runs (one per universe) produce all three checkpoints; intermediate checkpoints land on the `deep-finance-data` Modal Volume
- **3 pretrained checkpoints** committed: `deep_portfolio_etfs.pt`, `deep_portfolio_20stock.pt`, `deep_portfolio_100stock.pt` + JSON sidecars
- `scripts/run_backtests.py` extended to produce `data/backtests/portfolio_results.parquet` with rows for every (universe, condition, strategy) triple — see §8 for schema
- All four tabs of the Portfolio page functional:
  - Tab 1: rolling-correlation heatmap, summary stats, sector breakdown for stock universes
  - Tab 2: classical benchmarks with universe-aware method availability per §6 table
  - Tab 3: per-universe Deep Portfolio with model-card badge
  - Tab 4: sub-tab 4A (Table 1 replication), sub-tab 4B (Figure 3 three-panel chart)
- Math panel + code panel populated

**Acceptance:**
- Tab 4A reproduces the column structure of Table 1 of the paper for the ETF basket, with best-per-column values bolded, across all three conditions (no-vol-scale-C=0.01%, vol-scale-C=0.01%, vol-scale-C=0.10%)
- Tab 4B shows three log-scale cumulative-return panels matching Figure 3 of the paper, with all in-scope strategies overlaid
- Classical method results (Equal Weight, MV, MD) match `01_classical_portfolio_optimization.ipynb` outputs within numerical tolerance
- Deep Portfolio on the ETF basket reproduces the cumulative-return shape of Figure 3 (volatility-scaled, low-cost) from Zhang et al. 2020 — exact Sharpe values will differ since our test window extends to today
- Universe switcher (ETF / 20-stock / 100-stock) works without recomputation lag (uses precomputed parquet)
- For the ETF basket: DLS (Deep Portfolio) achieves higher Sharpe than all classical benchmarks under all three conditions

**Stretch deliverables (deferred to Phase 4):**
- Sub-tab 4C — Figure 6 sensitivity analysis heatmap via `torch.autograd.grad`
- Sub-tab 4D — COVID Stress Test (Figure 5)
- LSTM variant of Deep Portfolio (faithful to Zhang 2020; MLP is sufficient for Phase 2)

### Phase 3 — Page 3: Order Book (Tables I and II replication)

**Prerequisites:**
- Developer's Kaggle API token (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars) configured locally. FI-2010 is the only data source for this page; without it, Phase 3 cannot complete.

**Deliverables:**
- `scripts/fetch_lob_fi2010.py` — downloads FI-2010 from Kaggle, processes into `data/lob_fi2010.parquet`. Schema: `(day, timestamp, *40_features, label_k10, label_k20, label_k30, label_k50, label_k100)`. Includes attribution comment with Ntakaris et al. 2017 citation.
- `data/LOB_DATA_ATTRIBUTION.md` — full citation, license note, source links (Kaggle + Fairdata/Etsin)
- `src/strategies/lob_baselines.py` — implementations of representative baselines (LDA, plain MLP, plain CNN) that will be re-run on FI-2010 to produce fresh numbers; the remaining baselines (RR, SLFN, BoF, TABL, MCSDA) are cited from the paper with explicit badging in Tab 4 tables
- `src/models/deeplob.py` — DeepLOB architecture per Figure 3 of the paper: Conv1 (stride 1×2) → Conv2 (stride 1×2) → Conv3 → Inception → LSTM(64) → FC → **3-way softmax** output (single classification head, no regression head — the paper's task)
- `src/training/train_deeplob.py` — declares the Modal `App` / `Image` / `Volume` / `@app.function(gpu="A10G")` decorator at the top (DeepLOB is the largest model — ~140k params, A10G is the right tier per §7.1 budget); handles Setup 1 (anchored forward, 9 folds) and Setup 2 (7/3 split) via a `--setup` flag and all 5 prediction horizons via a `--k` flag, both routed through `@app.local_entrypoint()`. `modal run src/training/train_deeplob.py --setup 2 --k 10` reads FI-2010 from the `deep-finance-data` Modal Volume and produces `deeplob_fi2010.pt` as the headline checkpoint
- `data/pretrained/deeplob_fi2010.pt` + JSON sidecar committed
- `scripts/run_backtests.py` extended to produce `data/backtests/order_book_results.parquet` with rows for each (setup, k, method) triple — schema columns: `setup`, `k`, `method`, `accuracy`, `precision`, `recall`, `f1`, `reproduced` (bool — True for methods we reran, False for paper-cited values)
- All four tabs of the Order Book page functional:
  - Tab 1: LOB snapshot viz mirroring Figure 1, class-balance summary, Figure-2-style smoothed-label visualization
  - Tab 2: representative re-implemented baselines (LDA, MLP, CNN) with live demo
  - Tab 3: Architecture Explorer + Live Prediction Demo + Train Your Own
  - Tab 4: sub-tabs **4A Table I (Setup 1)**, **4B Table II (Setup 2)**, **4C Confusion Matrices**, **4D Per-Class Performance Curves**
- Math panel + code panel populated

**Acceptance:**
- DeepLOB achieves Setup-2 accuracy and F1 qualitatively consistent with Table II of the paper (paper reports F1 ≈ 83% for k=10 — our reproduced value should land within ±5pp)
- Tables 4A and 4B render with the paper's structure; cells from re-implemented baselines are badged distinctly from paper-cited cells
- LOB snapshot visualization renders for any selected timestamp; class-balance chart correctly reflects the FI-2010 label distribution
- Live Prediction Demo time slider works smoothly (< 200ms response per tick)
- Architecture Explorer shows correct block configs (kernel sizes 1×2 with stride 1×2 in first two conv blocks; Inception with 1×1, 3×1, 5×1, max-pool branches)
- Attribution file present and README links to it

**Stretch deliverables (deferred to Phase 4):**
- Sub-tab 4E — LIME sensitivity analysis reproducing Figure 9
- Setup-1 cross-validation training (9 folds × multiple horizons = expensive; ship Setup-2 results as primary, fill Setup-1 cells from paper-cited values initially)
- Train additional horizons (k=20, k=30, k=50, k=100) beyond the default k=10

### Phase 4 — Deployment & polish
**Deliverables:**
- `Dockerfile`, `.dockerignore`, `.streamlit/config.toml` complete per §15.1
- `README.md` with HF YAML front matter per §15.2, plus live-demo badge, quickstart, mermaid architecture diagram, citation block (BibTeX for the three papers + `CITATION.cff` for the app itself)
- `.github/workflows/sync-to-hf.yml` per §15.3, with `HF_TOKEN` secret + `HF_USERNAME` / `HF_SPACE_NAME` variables configured
- `FMP_API_KEY` configured as a Space secret
- LICENSE file (MIT)
- Optional CI: a second workflow that runs `ruff check` on PRs before allowing merge
- Hero screenshot + animated GIF of one of the pages in the README

**Acceptance:**
- A push to GitHub `main` triggers the sync workflow and the HF Space rebuilds successfully within 5 minutes
- HF Space cold-start time < 30 seconds on CPU Basic
- All three pages have working math + code panels
- README renders correctly on both GitHub (YAML visible as text block at top — acceptable) and HF Space (YAML parsed into the title card)
- The live HF Space URL is reachable, the app loads, all three pages render

## 12. Open Decisions to Confirm With User

Claude Code should pause and ask the user about these before locking them in:

1. **CME futures universe scope:** the paper uses 88 contracts. Confirm: (a) exact list of contract IDs to ship in `data/cme_futures.parquet`; (b) whether the Momentum page should expose all 88 in the asset-set picker or curate a subset for clarity.
2. **Access to `/data_lake/data_bento/cme` and `/QIS_Commodities`:** confirm Claude Code can read these paths, OR commit the relevant data-loading function and base trend-following strategy code into the repo as standalone files before Phase 1 begins.
3. **Portfolio 100-stock universe selection:** Claude Code should propose a sector-balanced list of 100 historical S&P 500 names; developer confirms. Watch for survivorship bias — use historical FMP constituents.
4. **20-stock ticker list:** confirmed from `portfolio_data.csv` header — AAPL, ABT, AEP, AXP, BAC, CI, GD, GE, HON, MMM, MO, MRK, NEM, NKE, NSC, PFE, PG, PTC, SNA, SO. FMP fetch extends history through today. No confirmation needed unless developer wants to swap tickers.
5. **VIX proxy:** brief specifies VIXY (ProShares VIX Short-Term Futures ETF) as the tradable proxy for the paper's "VIX". Alternative: VXX. Confirm preference.
6. **FMP shares-outstanding access:** required for the Diversity-Weighted Portfolio (DWP) benchmark. Confirm Premium tier supports the `enterprise-values` or `key-metrics` endpoint with historical sharesOutstanding. If not, DWP is skipped with a warning.
7. **COVID Stress Test (Figure 5):** previously in Phase 2 scope, now demoted to Phase 4 stretch per the latest Portfolio-page spec. Confirm whether to keep it in Phase 2 (it's a strong demo moment) or accept the demotion to keep Phase 2 acceptance tight.
8. **License:** MIT recommended unless user prefers Apache-2.0 or BSL.
9. **Color theme:** Light or dark default? Specific accent color?
10. **Order Book task framing:** ~~regression vs classification~~ **Resolved** — Page 3 uses 3-class classification on FI-2010 (matching the paper). The bundled `limit_order_book_data.csv` is no longer a runtime input; it lives in `notebooks/reference/data/` for anyone running the original GitHub notebook locally.
11. **Portfolio-page LSTM variant** (Zhang 2020 paper uses LSTM, not MLP): include in Phase 2 or defer to Phase 4 as a stretch? Recommended: defer — the MLP from the existing notebook is sufficient for Table 1 / Figure 3 replication. (Note: Momentum-page LSTM is non-optional per §6 Page 1.)

## 13. Reference Code

This section has two parts:

- **§13.0 — Pre-existing files (do not reimplement).** Code the user has already written and committed. Claude Code should copy these to `src/` and use them as-is, with extensions listed where applicable. Treat the user's existing implementation as canonical even where it differs from conventional defaults — e.g., `Neg_Sharpe` has no epsilon for numerical stability, and that's a deliberate match to the source notebooks.

- **§13.1 onward — Snippets to port from notebooks.** Reference implementations Claude Code must adapt from the user's source notebooks (`notebooks/reference/*.ipynb`). These are not yet in `src/`.

### 13.0 Pre-existing canonical files

These files were uploaded by the user. Copy verbatim into `src/`:

| File | Purpose | Extension needed? |
|---|---|---|
| `src/losses.py` | `Neg_Sharpe(portfolio)` function + `SharpeLoss(nn.Module)` class | No — use as-is |
| `src/early_stopper.py` | `EarlyStopping` class with patience, min_delta, model checkpoint saving | No — use as-is |
| `src/torch_data.py` | `MyDataset(data.Dataset)` — generic numpy → torch tensor wrapper | No — use as-is |
| `src/metrics.py` | `report_metrics(ret)` — currently returns annual return, std, Sharpe | **Yes — extend per §13.7** |
| `src/__init__.py` | Empty package marker | No — use as-is |

**Critical naming notes for Claude Code:**

- The Sharpe function is `Neg_Sharpe` (capital `N`, capital `S`). Do not rename to `neg_sharpe`.
- The dataset class is `MyDataset`. Generic but used throughout the notebooks — keep the name.
- The early-stopper takes `savepath` as its first positional argument and writes `model.state_dict()` to disk on improvement.

**On the `SharpeLoss` signature:** the user's implementation takes `(outputs_prev, future_rets)` and computes `outputs_prev * future_rets` element-wise. For the single-asset Momentum page this is exactly right. For the multi-asset Portfolio page where the model outputs a softmax weight vector per timestep, Claude Code should wrap or pre-aggregate the per-asset products into a portfolio return series **before** passing to `SharpeLoss`, rather than modifying the loss class. This preserves the loss as a one-line drop-in across both pages.

### 13.1 (was: Sharpe Loss — now replaced by §13.0)

The `SharpeLoss` and `Neg_Sharpe` are already in `src/losses.py` per §13.0. No code reproduction here.

### 13.2 Deep Portfolio MLP (Portfolio page)

```python
# src/models/deep_portfolio.py — from 02_deep_portfolio_optimization.ipynb
import torch
import torch.nn as nn

class DeepPortfolioMLP(nn.Module):
    def __init__(self, seq_length, n_features, y_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length * n_features, 4),
            nn.Tanh(),
            nn.Linear(4, y_dim),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # Softmax → long-only weights summing to 1
        return torch.softmax(x, dim=1)
```

### 13.3 Deep Momentum MLP (Momentum page)

```python
# src/models/deep_momentum.py — from 03_deep_momentum_strategy.ipynb
import torch.nn as nn

class DeepMomentumMLP(nn.Module):
    def __init__(self, seq_length, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length * n_features, 64),
            nn.Softsign(),
            nn.Linear(64, 32),
            nn.Softsign(),
            nn.Linear(32, 1),
            nn.Softsign(),  # output in (-1, 1) → continuous position
        )

    def forward(self, x):
        return self.net(x)
```

### 13.4 DeepLOB (Order Book page)

Copy `class deeplob` verbatim from `02_predictive_signal_lob.ipynb`. The architecture has three conv blocks (with gated/tanh activations between LeakyReLU blocks), an Inception module with three parallel branches, and a final LSTM + FC head. Output is a scalar (regression on scaled log-return).

### 13.5 TSMOM / SMA / MACD signals (Momentum page)

```python
# src/strategies/tsmom.py — from 01_time_series_momentum.ipynb
import numpy as np

def tsmom_signal(price, lookback=252):
    """Sign of past-year return."""
    return np.sign(price.pct_change(lookback))

def sma_signal(price, k1=10, k2=40):
    """Volatility-normalized SMA crossover."""
    diff = price.rolling(k1).mean() - price.rolling(k2).mean()
    return diff / diff.rolling(k2).std()

def macd_signal(price, k1=8, k2=24):
    """Volatility-normalized MACD."""
    diff = price.ewm(k1).mean() - price.ewm(k2).mean()
    return diff / diff.rolling(k2).std()

def macd_ensemble(price, shorts=(8, 16, 32), longs=(24, 48, 96)):
    """Baz et al. 2015 — average MACD across three timescales."""
    signals = [macd_signal(price, s, l) for s, l in zip(shorts, longs)]
    return np.mean(signals, axis=0)
```

### 13.6 Classical portfolio weights (Portfolio page)

```python
# src/strategies/classical_portfolio.py — from 01_classical_portfolio_optimization.ipynb
import numpy as np
from sklearn.covariance import ShrunkCovariance

def max_diversification_weights(returns_window):
    cov = ShrunkCovariance().fit(returns_window).covariance_
    inv_cov = np.linalg.inv(cov)
    vols = returns_window.std().values.reshape(-1, 1)
    num = inv_cov @ vols
    denom = (vols.T @ inv_cov @ vols).item()
    weights = (num / denom).flatten() * vols.flatten()
    return weights

def min_variance_weights(returns_window):
    cov = ShrunkCovariance().fit(returns_window).covariance_
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(returns_window.shape[1])
    num = inv_cov @ ones
    denom = (ones.T @ inv_cov @ ones).item()
    return num / denom
```

### 13.7 Metrics extension (`src/metrics.py`)

The user's starter `report_metrics(ret)` returns:

```python
res['annual_ret']    = np.mean(ret) * 252
res['annual_std']    = np.std(ret) * np.sqrt(252)
res['annual_sharpe'] = (np.mean(ret) / np.std(ret)) * np.sqrt(252)
```

**Extend `report_metrics(ret)` to also return:**

```python
# Downside deviation — std of negative returns only, annualized
res['downside_dev']  = np.std(ret[ret < 0]) * np.sqrt(252)

# Sortino — Sharpe-like ratio using downside deviation
res['sortino']       = (np.mean(ret) / np.std(ret[ret < 0])) * np.sqrt(252)

# Maximum drawdown — max peak-to-trough decline of the equity curve
equity               = (1 + ret).cumprod()
running_max          = np.maximum.accumulate(equity)
drawdown             = (equity - running_max) / running_max
res['max_drawdown']  = abs(drawdown.min())

# Calmar — annualized return / max drawdown
res['calmar']        = res['annual_ret'] / res['max_drawdown']

# Hit rate — fraction of days with positive returns
res['pct_positive']  = (ret > 0).mean()

# Average profit / average loss — magnitude ratio of wins vs losses
res['avg_p_over_l']  = ret[ret > 0].mean() / abs(ret[ret < 0].mean())
```

**Conventions:**
- All ratios annualized using `sqrt(252)` to match the user's existing convention
- All keys lowercase with underscores, matching the existing style
- Backward-compatible: existing keys (`annual_ret`, `annual_std`, `annual_sharpe`) remain unchanged
- The function continues to accept a 1-D numpy array of daily returns; no signature change

**Cross-checks for the unit test:** run on a synthetic returns series (e.g. `np.random.normal(0.0005, 0.01, 1000)`) and verify Sharpe ≈ 0.79, max_drawdown ∈ [0.05, 0.20], hit_rate ≈ 0.52. Match against values reported in the source notebooks (Exhibits 2 / Table 1 / etc.) where present.

## 14. README Outline

The final README should contain:

1. **Hero section** — title, one-line description, hero screenshot, live-demo button
2. **Live demo badge** — clickable badge linking to the HF Space (use `https://img.shields.io/badge/🤗_Space-Live_Demo-blue` style)
3. **Animated GIF** of the comparison dashboard scrolling through strategies
4. **Three paper cards** with citations + arXiv links
5. **Quickstart** — `git clone && pip install -r requirements.txt && streamlit run streamlit_app.py`
6. **Architecture diagram** — mermaid showing the data + deployment flow:
   - FMP → `scripts/fetch_data.py` (local) → parquet → repo
   - Modal Volume `deep-finance-data` → `modal run src/training/train_*.py` (GPU container) → `.pt` checkpoints on Volume → `modal volume get` → repo
   - GitHub `main` → GitHub Action → HF Space → live URL
7. **Repo map** — abbreviated tree
8. **How to refresh data** — for users with their own FMP keys
9. **How to retrain checkpoints** — pointer to `src/training/train_*.py` (each is a Modal app; run with `modal run src/training/train_*.py`), with a one-paragraph explainer that training runs on Modal serverless GPU containers, not locally, and that committed `.pt` files are produced via `modal volume get` after the training run lands them in the Modal Volume
10. **How to deploy** — pointer to §15 of this brief (or the equivalent DEPLOYMENT.md if extracted)
11. **Citation block** (BibTeX) for the three papers plus a `CITATION.cff` for the app itself
12. **Acknowledgments** — Zhang, Lim, Zohren, Roberts; Oxford-Man Institute
13. **License** — MIT
14. **Contributing** — short note

The README starts with a YAML front matter block that HF reads to render the Space card (title, emoji, colors, SDK metadata). GitHub renders this YAML as a fenced text block at the top of the page — slightly cosmetic but standard practice and most viewers ignore it. The body below the YAML is the GitHub-canonical README.

## 15. Deployment Workflow — GitHub → Hugging Face Space

This section assumes the HF Space already exists (you've completed `huggingface.co/new-space` with Docker SDK, CPU Basic hardware, MIT license). Goal: wire GitHub to be the canonical home and have every push to `main` auto-deploy to the Space.

### 15.1 Dockerfile (repo root)

HF reads this on every push and rebuilds the runtime image. Keep it minimal:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Optional: apt deps (most likely empty for this project)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         <pkg1> <pkg2> \
#     && rm -rf /var/lib/apt/lists/*

# Install Python deps first so Docker layer-caches them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# HF Spaces route traffic to port 7860 by default
EXPOSE 7860

# Belt-and-suspenders: also set these in .streamlit/config.toml
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "streamlit_app.py"]
```

A matching `.streamlit/config.toml`:

```toml
[server]
headless = true
port = 7860
address = "0.0.0.0"

[browser]
gatherUsageStats = false
```

A matching `.dockerignore` to keep build context small (papers, notebooks, and `.git/` don't need to land inside the container):

```
.git/
.github/
papers/
notebooks/
.streamlit/secrets.toml
.env
__pycache__/
*.pyc
```

### 15.2 README front matter (repo root `README.md`)

The very top of `README.md` is a YAML block HF parses to render the Space's title card:

```markdown
---
title: Deep Finance Showcase
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Interactive demos of three Oxford-Man Institute papers applying deep learning to portfolio optimization, momentum, and limit order books
---

# Deep Finance Showcase

[Live Demo badge here] [GitHub badge here] [arXiv links here]

[Rest of the GitHub-canonical README starts here]
```

### 15.3 GitHub Action — auto-deploy on every push

File: `.github/workflows/sync-to-hf.yml`

```yaml
name: Sync to Hugging Face Space

on:
  push:
    branches: [main]
  workflow_dispatch:  # also allow manual triggers from the GitHub UI

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout (full history)
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # HF rejects shallow pushes
          lfs: true       # in case any files are LFS-tracked

      - name: Push to Hugging Face Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USERNAME: ${{ vars.HF_USERNAME }}
          HF_SPACE_NAME: ${{ vars.HF_SPACE_NAME }}
        run: |
          git push --force \
            https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE_NAME} \
            main
```

**Why `--force`:** the HF Space was created with its own initial commit (the auto-generated Docker template). Once GitHub becomes canonical, that history is overwritten on every push. Trade-off: you lose the HF Space's independent commit history, but nobody was looking at it anyway.

### 15.4 Secrets and variables — four stores, four purposes

**GitHub repo Secrets** (`Settings → Secrets and variables → Actions → Secrets` tab):

| Name | Value | Used by |
|---|---|---|
| `HF_TOKEN` | HF user access token with **write** scope. Create at `huggingface.co/settings/tokens` | the sync workflow |

**GitHub repo Variables** (same page, `Variables` tab — non-secret, just config):

| Name | Example | Used by |
|---|---|---|
| `HF_USERNAME` | `JJ-JIN12345` | the sync workflow |
| `HF_SPACE_NAME` | `deep-finance-showcase` | the sync workflow |

**HF Space secrets** (`Space → Settings → Variables and secrets`):

| Name | Value | Used by |
|---|---|---|
| `FMP_API_KEY` | Your FMP Premium key | the live Streamlit app, only when the user clicks "refresh data" |

**Modal account credentials & Secrets** (developer's machine + `modal.com` dashboard):

| Name | Where it lives | Used by |
|---|---|---|
| Modal token | `~/.modal.toml`, written by `modal token new` | every `modal run` and `modal volume` invocation from your terminal |
| `deep-finance-train-env` (Modal Secret) | Created with `modal secret create deep-finance-train-env FMP_API_KEY=...` | training functions that need FMP at runtime (rare — most training reads parquet from the Volume) |

Modal credentials are **never committed to the repo** and **never set as GitHub or HF secrets** — Modal is a developer-side tool only; HF Space does not call Modal at runtime. Adding `~/.modal.toml` to your dev machine's gitignore is good belt-and-braces (it lives under `$HOME`, outside the repo, but worth noting).

**Local `.env`** (gitignored; `.env.example` committed):

```
FMP_API_KEY=your_fmp_premium_key_here
DEEP_FINANCE_DATA_DIR=~/data_lake/deep-finance/   # optional; default ./data/
```

The app reads `FMP_API_KEY` via `os.environ.get("FMP_API_KEY")` in both local and HF environments. On HF, Streamlit also exposes Space secrets via `st.secrets["FMP_API_KEY"]` — either pattern works, pick one and stick to it (recommend `os.environ` for local/HF symmetry).

### 15.5 First-time setup checklist

The Space exists, the GitHub repo does not yet (or exists empty). Walk through this once:

1. **Initialize the GitHub repo locally**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: project skeleton"
   git branch -M main
   git remote add origin git@github.com:<user>/<repo>.git
   git push -u origin main
   ```

2. **In GitHub repo `Settings → Secrets and variables → Actions`**:
   - Add Secret: `HF_TOKEN` (paste the token from `huggingface.co/settings/tokens`, create with **write** scope)
   - Add Variable: `HF_USERNAME` = your HF username
   - Add Variable: `HF_SPACE_NAME` = your Space name

3. **In HF Space `Settings → Variables and secrets`**:
   - Add Secret: `FMP_API_KEY` = your FMP Premium key

4. **Push any commit to GitHub `main`** (or trigger the workflow manually from the Actions tab). The Action runs, force-pushes to HF, HF rebuilds the Docker image (~2–5 minutes), Space goes live.

5. **Verify**: open the HF Space URL, confirm the app loads. Check `Settings → Logs` if it doesn't.

### 15.6 Routine deployment loop

After initial setup, deployment is invisible:

```bash
# Develop locally
streamlit run streamlit_app.py

# Ship it
git add .
git commit -m "feat: add Order Book page DeepLOB demo"
git push origin main

# GitHub Action triggers, HF Space rebuilds, live in ~2 minutes
```

To redeploy without a code change (e.g. after rotating `FMP_API_KEY` on the HF side), hit "Run workflow" on the GitHub Actions tab — that's what `workflow_dispatch` enables.

### 15.7 Common failure modes

- **First push rejected** because HF Space has a different initial commit. Force-push (`--force` in the workflow) handles this. If you ran the workflow before adding the force flag, manually run `git push --force <hf-url> main` once.
- **File too large for HF push** (>10 MB without LFS). The parquet files in this project should stay under 50 MB total; commit them directly. If you ever cross 50 MB, switch to Git LFS — HF supports it natively, GitHub does too, just `git lfs track "*.parquet"` and recommit.
- **Auth error from sync workflow** — the HF token expired or was revoked. Regenerate at `huggingface.co/settings/tokens` with write scope, update the GitHub secret.
- **App loads but FMP refresh fails silently** — `FMP_API_KEY` not set in HF Space secrets (the app reads from env, so on HF it must be a Space secret, not a GitHub secret).
- **Streamlit shows a blank page** — port mismatch. Verify the Dockerfile `EXPOSE 7860` and the README front-matter `app_port: 7860` agree, and that `.streamlit/config.toml` sets `port = 7860`.
- **`pip install` OOMs during Docker build** — unlikely with this project's footprint, but if it happens, the build env on HF has 16 GB RAM; switching to multi-stage build or `--no-cache-dir` (already in the Dockerfile) usually fixes it.

### 15.8 Optional refinements

- **Secondary deploy to Streamlit Community Cloud** — point a separate Streamlit Cloud app at the same GitHub repo. It will auto-detect `streamlit_app.py` and run it (ignoring the Dockerfile). Costs nothing extra, gives you a backup URL.
- **Custom domain on HF** — paid feature, not necessary for a research showcase. The default `https://huggingface.co/spaces/<user>/<space>` URL is fine and SEO-friendly.
- **CI checks** — add a second GitHub Action that runs `ruff check` and basic import tests before the sync job, so broken builds never reach HF.
- **Preview deploys for PRs** — would need a separate HF Space per branch. Overkill for this project, but the pattern is: clone the Space, give it a different name, point a branch-specific workflow at it.

## 16. Non-Goals

To keep scope tight, the agent should NOT:
- Build any kind of user authentication
- Build a database — everything is parquet files
- Implement live trading or order execution
- Hyperparameter-tune the pretrained models beyond what's in the notebooks
- Add transaction-cost modeling beyond a flat-bps slider (Phase 3 stretch)
- Replicate every figure from every paper — just the headline ones called out per-page
- Build a backend API — Streamlit is the whole app