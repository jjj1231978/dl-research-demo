# Refinement brief: Deep Finance Showcase

**Repo:** `jjj1231978/dl-research-demo`
**Live:** https://jj-jin12345-dl-research-demo.hf.space
**Reviewed:** 2026-09-20, against the deployed Space
**Goal:** the site should read as finished research work, not as a project in progress.

This document is the *why*. `TASKS.md` is the *what*. Read this once before starting
Tier 0, then work from `TASKS.md`.

---

## 1. What the site gets right

Worth stating, because none of it should be lost in the refinement.

- The **Common Thread** section on the landing page is the strongest writing on the
  site. It makes a real intellectual claim (bypass the forecast step, optimise the
  financial objective directly) and supports it across all three papers. Most
  paper-replication demos never attempt this.
- Every page cites its paper with an arXiv link. Substrate disclosure is honest about
  where the data substitutes for the paper's universe.
- The math and code expanders are the right pattern: depth available, not forced.
- The fallback to bundled CSVs when no FMP key is present means a visitor never hits
  a broken page.

The content is good. The problems below are presentation, structure, and packaging.

---

## 2. Findings

### 2.1 The Space card advertises that the site is unfinished

`README.md` renders directly beneath the app on the Hugging Face Space page. It
currently contains:

> **Phase 0 minimal README** — full README with hero screenshot, live-demo badge,
> citation block, and deployment instructions lands in Phase 4.

Anyone you send the link to reads that before they read the app. It is the single
highest-leverage fix in this package.

### 2.2 The live-demo badge is broken

```
[![Live demo](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces/jjj1231978/dl-research-demo)
```

`jjj1231978` is the GitHub username. The Space is `JJ-JIN12345/dl-research-demo`.
The badge URL returns 401 (verified). The GitHub badge on the next line is correct,
which is how the error crept in.

### 2.3 No theme is configured

`.streamlit/config.toml` has `[server]` and `[browser]` only. There is no `[theme]`
block, so every widget renders in Streamlit's default `#FF4B4B`: slider tracks and
handles, the active tab underline, checkboxes, the running spinner, and link colour.

That red is the most recognisable "unstyled Streamlit app" signal there is. For an
institutional quant audience it is also simply the wrong register.

### 2.4 Line length is unbounded

Measured on the deployed site at a 1440px viewport:

| Element | Computed |
|---|---|
| `[data-testid="stSidebar"]` | 300px |
| `[data-testid="stMainBlockContainer"]` | 970px, `max-width: none`, 80px horizontal padding |
| Body font size | 16px |

970px at 16px is roughly 120 characters per line. On a 27-inch display the column
grows without limit. The research consensus for sustained reading is 50 to 75
characters. This is the main reason the prose is tiring, and it is invisible on a
laptop screenshot because the effect scales with the monitor.

The fix is to cap the **measure** (the text), not the container, so charts and tables
keep the full width they need.

> Note: an earlier read of this site reported the content column as half-width with
> dead space on the right. That was a misread of a downscaled screenshot. The DOM
> measurement above is correct and supersedes it.

### 2.5 Metric values are truncating in front of the reader

Verified on two pages:

- Portfolio Optimization, "Date range": renders as `2011-01-03 → 2…`
- Momentum, "Date range": renders as `2010-06-06 → 2026…`

`st.metric` clips its value to the column width with an ellipsis. A date range is not
a metric, and it should never have been in one. The package ships a `stat_row()`
helper that lays out label/value pairs without clipping.

### 2.6 The headline result is three levels deep

Current navigation to the paper's Table 1:

```
sidebar page nav  ->  "4. Key Results"  ->  sub-tab "Performance Across Methods"
```

Confirmed in the DOM: six `role="tab"` elements on the Portfolio page, four top-level
plus two nested inside the fourth panel. Tabs inside tabs are hard to discover and
give the reader no sense of where they are.

The walkthrough order (problem, benchmarks, method, results) is the order a *paper*
is written in. A showcase is read in the opposite order: the reader wants to know
whether the result is interesting before they invest attention in the method.

### 2.7 Everything renders eagerly

All seven Plotly figures on the Portfolio page exist in the DOM on first paint, even
though five of them sit in hidden tab panels. Streamlit renders every tab body
regardless of which tab is active. That is why first load sits on skeleton
placeholders for roughly five seconds, and why every widget change recomputes work
the reader cannot see.

### 2.8 Chart problems

- **Correlation heatmap.** The unit diagonal sets the top of the colour scale, so the
  informative off-diagonal correlations collapse to near-white. Both triangles of a
  symmetric matrix are drawn, saying everything twice. Both axis titles read `symbol`,
  a dataframe column name leaking into the exhibit.
- **Modebar.** The full Plotly toolbar (zoom, pan, box select, lasso, autoscale, reset
  axes, spike lines) sits above static exhibits. It reads as leftover tooling.
- **LOB snapshot.** Saturated red/green bars. Poor for the roughly 8% of male readers
  with a red-green deficiency, and louder than the data warrants.
- **Coverage timeline.** Tick labels are too small to read at normal viewing distance.
- **No shared palette.** Each chart picks its own colours, so "the deep model" is not
  the same colour twice.

### 2.9 Smaller items, all reader-visible

| Item | Where | Problem |
|---|---|---|
| `Project_brief.md` link | `src/data/__init__.py` sidebar | Relative href. Streamlit will not serve it. Dead link. |
| `Constitution v1.1.0` | footer of every page | Internal process artifact in visitor-facing chrome. |
| `Backtest range (Tab 2 / Tab 4)` | sidebar control | Makes the reader map controls to tab indices. |
| `initial_sidebar_state="expanded"` | all four entry points | On mobile the sidebar covers the whole page on arrival. |
| H1 carries full citation | page headers | `Portfolio Optimization — Zhang, Zohren, Roberts (2020)` wraps to two lines and buries the subject. |
| Emoji density | title, nav, tabs, cards, expanders | Carrying the visual identity rather than decorating it. |
| Unpinned dependencies | `requirements.txt` | Nothing is pinned, not even `streamlit` or `torch`. A rebuild in six months pulls whatever is current. |

---

## 3. The visual system

One system, applied everywhere. The two shipped assets implement it; this section
records the reasoning so later changes stay consistent.

### 3.1 Palette

| Token | Hex | Use |
|---|---|---|
| `INK` | `#1A1D24` | body text |
| `MUTED` | `#5B6472` | captions, axis labels, secondary values |
| `RULE` | `#E2E6EB` | gridlines, borders, dividers |
| `ACCENT` | `#1F4E79` | primary, and "the deep model" in every chart |

Categorical series, in order: `#1F4E79` deep blue, `#C77B3C` amber, `#4C8C7A` teal,
`#8A6FA0` violet, `#9C4F4F` brick, `#6E7B8B` slate. Lightness varies as well as hue,
so lines stay separable in greyscale and under red-green deficiency.

Colours are also assigned **semantically** (`deep`, `benchmark`, `classical`,
`baseline`). Use the semantic mapping whenever a chart compares the paper's model
against classical methods. Readers learn one colour for one idea and carry it across
pages, which is most of what makes a multi-page site read as a single artifact.

Diverging scale for correlations and signed weights is symmetric in lightness around
a near-white midpoint, so zero reads as "nothing here" rather than as a value.

### 3.2 Typography

- Cap prose at `74ch`. Leave charts, tables and dataframes at full container width.
- Cap the container at `1180px` so the page has a shape on a wide monitor.
- Headings get tighter tracking than the Streamlit default, which is tuned for
  dashboards rather than for reading.
- `h2` carries a hairline rule above it. On a long scrolling page that does more work
  than size alone.
- Numbers use `font-variant-numeric: tabular-nums` so columns of figures align.

### 3.3 Charts

- One registered Plotly template (`dfs`), set as the default, so nothing has to be
  styled per chart.
- Horizontal gridlines only, dotted, in `RULE`. The reader compares values, not dates.
- No chart title inside the figure. The heading above it is the title. Two titles is
  the most common Streamlit chart smell.
- Legend horizontal, above the plot, left-aligned to the axis, no legend title.
- Modebar off by default. On where zoom genuinely helps (the order-book tick slice),
  trimmed to the buttons that do something.
- `theme=None` on `st.plotly_chart`, otherwise Streamlit's own theme overrides the
  template's colourway.

### 3.4 Register

The target is an institutional research note: quiet, dense, confident. Emoji are the
main thing working against that register right now. Keep at most one, as a favicon or
a single wordmark. Let the typography and the palette carry the identity.

---

## 4. Constraints for whoever implements this

1. **Do not touch model, metric, backtest or data-loading code.** Everything in this
   package is presentation. If a task appears to require a change under
   `src/models/`, `src/strategies/`, `src/training/` or `src/metrics.py`, stop and
   flag it instead.
2. **`tests/integration/test_*_page.py` will break.** Those tests assert on UI
   strings and structure. Renaming a tab or a control label breaks them by design.
   Update the test to the new string deliberately, in the same commit. Never loosen
   an assertion to make it pass.
3. **The Space builds from the Dockerfile** (`sdk: docker` in the README front
   matter, `app_port: 7860`). Changing `.streamlit/config.toml` does not require a
   Dockerfile change, but do confirm the file is copied into the image.
4. **Keep the spec-kit scaffolding in the repo.** `.specify/`, `specs/` and
   `CLAUDE.md` stay on GitHub where they document the process. They just stop being
   referenced from visitor-facing chrome.
5. **One task at a time.** Each task in `TASKS.md` is independently committable and
   has its own acceptance criteria. Finish and verify one before starting the next.
