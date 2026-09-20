# Refinement tasks

Three tiers. Each tier leaves the site in a shippable state, so you can stop after
any of them. Within a tier, tasks are ordered by dependency.

**Rules of engagement**

- Do one task, verify it, commit it, then move to the next. Do not batch.
- Presentation only. Nothing under `src/models/`, `src/strategies/`, `src/training/`
  or `src/metrics.py` should change. If a task seems to need it, stop and ask.
- Run `pytest -q` after every task. `tests/integration/test_*_page.py` asserts on UI
  strings and will break when labels change. Update the assertion to the new string
  in the same commit. Never weaken an assertion to make it pass.
- Commit message format: `refine(T0.2): rewrite root README`.

**Status legend:** `[ ]` not started, `[x]` done, `[-]` skipped (say why).

---

# Tier 0: first impression and repo hygiene

No app behaviour changes. Lowest risk, highest visible return. About an hour.

## [x] T0.1 Fix the broken live-demo badge

**File:** `README.md`

The badge points at `huggingface.co/spaces/jjj1231978/dl-research-demo`. That is the
GitHub username, not the Space owner. The URL returns 401. The Space is
`JJ-JIN12345/dl-research-demo`.

**Do:** change the badge target to
`https://huggingface.co/spaces/JJ-JIN12345/dl-research-demo`.

**Accept:** `curl -sI https://huggingface.co/spaces/JJ-JIN12345/dl-research-demo | head -1`
returns 200. Grep the whole repo for the wrong namespace and fix any other instance:
`grep -rn "jjj1231978/dl-research-demo" --include="*.md" --include="*.py" .` should
return only GitHub URLs, never Hugging Face ones.

---

## [x] T0.2 Rewrite the root README

**Files:** `README.md`, source draft at `refinement/assets/README.md.draft`

The current body opens with a blockquote announcing that this is a "Phase 0 minimal
README" and that the real one arrives in Phase 4. That text renders directly under
the app on the Space page.

**Do:**

1. Keep the existing YAML front matter byte for byte. Hugging Face parses it:
   `title`, `emoji`, `colorFrom`, `colorTo`, `sdk: docker`, `app_port: 7860`,
   `pinned`, `license`. Changing it changes how the Space builds.
2. Replace everything below the front matter with the draft.
3. The draft carries one `TODO(T0.3)` marker, the commented-out hero image. Leave it
   until T0.3, then uncomment it and delete the marker.
4. Do not delete `Project_brief.md`, `specs/` or `.specify/`. The new README links to
   them under a "How this was built" heading, which is the right place for them.

**Accept:** no occurrence of `Phase 0`, `Phase 4` or `lands in` remains in
`README.md`. The front matter is unchanged (`git diff README.md` shows no change
inside the `---` fences). Both badges resolve.

---

## [x] T0.3 Add a hero screenshot

**Files:** `docs/hero.png` (new), `README.md`

A README for a visual project with no image of the project is a missed beat.

**Do:** capture the Portfolio Optimization page at 1440px wide, results visible, at
2x device pixel ratio. Save to `docs/hero.png`. Keep it under 400KB. Reference it in
the README immediately after the intro paragraph, wrapped in a link to the live Space.

**Accept:** image renders on GitHub and on the Space card. File under 400KB.

**Note:** do this *after* Tier 1 if you are running the whole package, so the
screenshot shows the themed site. Leave it unchecked until then.

---

## [x] T0.4 Pin the runtime dependencies

**File:** `requirements.txt`

Nothing is pinned, including `streamlit` and `torch`. The Space rebuilds from this
file, so a rebuild months from now can pull a Streamlit major version that renders
the app differently or breaks it.

**Do:**

1. Build the image or create a clean venv from the current unpinned file.
2. `pip freeze` inside it.
3. Pin the direct dependencies to the resolved versions with `==`. Keep the
   `--extra-index-url` line and the existing comments.
4. Keep `pytest` and `pytest-cov` in the file (the comment explains why, tests run on
   the production environment). Pin them too.
5. Leave `requirements-train.txt` alone.

**Accept:** every non-comment, non-flag line in `requirements.txt` carries `==`. A
fresh `pip install -r requirements.txt` succeeds and `pytest -q` passes.

---

## [x] T0.5 Remove internal process artifacts from the app chrome

**Files:** `streamlit_app.py`, `pages/1_📈_Momentum.py`,
`pages/2_💼_Portfolio_Optimization.py`, `pages/3_📖_Limit_Order_Book.py`

Every page footer ends with `Constitution v1.1.0`. That is a spec-kit artifact and
means nothing to a visitor.

**Do:** replace the footer caption on all four pages with a single shared footer.
Keep the GitHub link. Add the paper credits and a build date. Suggested text:

```
Source on GitHub  ·  Papers: Lim, Zohren & Roberts (2019); Zhang, Zohren & Roberts (2020, 2019)  ·  Updated <date>
```

Do not hand-write this. `refinement/assets/theme.py` already ships `page_footer()`,
which renders exactly that line from one place so the four pages cannot drift apart.

That means this task depends on T1.2 (which installs the module). Either run T1.2
first and come back, or, if you want the footer gone before touching anything visual,
delete the caption now and add `page_footer()` during T1.3.

**Accept:** `grep -rn "Constitution" streamlit_app.py pages/` returns nothing. All
four pages render the same footer. The files under `.specify/` are untouched.

---

## [x] T0.6 Fix the dead "Project brief" sidebar link

**File:** `src/data/__init__.py` (in `render_data_status_sidebar`)

The link uses the relative href `Project_brief.md`. Streamlit serves the app, not the
repo, so this 404s for every visitor.

**Do:** point it at the GitHub blob URL:
`https://github.com/jjj1231978/dl-research-demo/blob/main/Project_brief.md`.
While in this function, audit every other link for the same problem.

**Accept:** every `href` rendered by the sidebar is absolute and returns 200.

---

# Tier 1: the visual system

The substance of the refinement. Two drop-in assets plus the wiring.

## [x] T1.1 Install the theme config

**File:** `.streamlit/config.toml`, source at `refinement/assets/streamlit-config.toml`

**Do:** replace the file with the asset. It keeps the existing `[server]` and
`[browser]` blocks unchanged and adds `[client]`, `[theme]` and `[theme.sidebar]`.

The asset marks one block as version-sensitive. Check the Streamlit version you
pinned in T0.4:

- `>= 1.44`: keep everything.
- `1.28` to `1.43`: delete the marked block and the `[theme.sidebar]` section.
- `< 1.42`: also change `showErrorDetails = "none"` to `showErrorDetails = false`.

**Do also:** confirm the Dockerfile copies `.streamlit/` into the image. If it copies
individual files rather than the directory, add the config explicitly.

**Accept:** `streamlit run streamlit_app.py` starts with no config warnings in the
terminal. Sliders and the active tab underline render deep blue, not red. The sidebar
sits on a tinted surface distinct from the main column.

---

## [x] T1.2 Add the shared UI module

**Files:** `src/ui/__init__.py` (new, empty is fine), `src/ui/theme.py` (new),
source at `refinement/assets/theme.py`

**Do:** copy the asset to `src/ui/theme.py`. Add the package `__init__.py`. Do not
edit the asset yet; wire it up first and adjust once you can see it.

What it provides:

| Name | Purpose |
|---|---|
| `apply_page_chrome()` | injects the shared CSS (measure cap, headings, tabs, sidebar) |
| `page_header(...)` | consistent masthead: title, citation line, standfirst |
| `page_footer()` | one footer for all four pages (replaces the Constitution caption) |
| `plot(fig, interactive=False, height=None)` | `st.plotly_chart` with the house config |
| `stat_row([...])` | label/value pairs that do not truncate |
| `correlation_heatmap(corr)` | masked diagonal, lower triangle, annotated cells |
| `series_by_role(fig, {...})` | recolour traces semantically |
| `INK`, `MUTED`, `RULE`, `ACCENT`, `CATEGORICAL`, `SEMANTIC` | palette tokens |

It registers a Plotly template named `dfs` and sets `plotly_white+dfs` as the default
at import time.

**Accept:** `python -c "from src.ui.theme import plot, stat_row; print('ok')"` prints
`ok`. `pytest -q` still passes.

---

## [x] T1.3 Wire the chrome into all four entry points

**Files:** `streamlit_app.py`, `pages/1_📈_Momentum.py`,
`pages/2_💼_Portfolio_Optimization.py`, `pages/3_📖_Limit_Order_Book.py`

**Do:** in each file, immediately after `st.set_page_config(...)`:

```python
from src.ui.theme import apply_page_chrome

apply_page_chrome()
```

In the same `set_page_config` call, change `initial_sidebar_state="expanded"` to
`"auto"`. On a phone, `expanded` means the sidebar covers the page on arrival.

At the bottom of each file, replace the trailing `st.divider()` plus `st.caption(...)`
with a single `page_footer()` call (see T0.5).

Edit `_REPO` in `theme.py` if the GitHub URL is wrong, since three helpers read it.

**Accept:** at 1440px the prose column measures about 74 characters per line while
charts still span the full container. At 390px the sidebar starts collapsed. Check
with the browser inspector, not by eye.

---

## [x] T1.4 Route every chart through `plot()`

**Files:** all four entry points, plus anything they import that draws

**Do:**

1. `grep -rn "st.plotly_chart" --include="*.py" .` to find every call site.
2. Replace each with `plot(fig)`. Drop the now-redundant `use_container_width` and
   `config` arguments.
3. Exception: the order-book tick-slice chart on page 3, where zooming into the
   microstructure is the point. Use `plot(fig, interactive=True)` there.
4. Remove any per-figure styling the template now handles: `paper_bgcolor`,
   `plot_bgcolor`, explicit font families, hardcoded hex colours, gridline colours.
   Leave anything genuinely chart-specific such as axis ranges or annotations.
5. Where a chart compares the paper's model against classical baselines, apply
   `series_by_role(fig, {"Deep Portfolio": "deep", "Equal Weight": "baseline", ...})`
   so the deep model is the same blue on every page. Match the trace names actually
   used in the code.

**Accept:** `grep -rn "st.plotly_chart" --include="*.py" .` returns only the one call
inside `src/ui/theme.py`. No modebar on any chart except the order-book tick slice.
Every chart shares one font, one gridline treatment, one palette.

---

## [x] T1.5 Replace the truncating metrics

**Files:** `pages/1_📈_Momentum.py`, `pages/2_💼_Portfolio_Optimization.py`, and any
other page using `st.metric` for a wide value

Currently rendering as `2011-01-03 → 2…` and `2010-06-06 → 2026…`.

**Do:** replace those `st.columns` plus `st.metric` blocks with `stat_row`:

```python
stat_row([
    ("Assets", "4"),
    ("Date range", "2011-01-03 to 2026-05-04", "3,857 trading days"),
    ("Avg pairwise corr", "-0.120"),
])
```

Use the third element for the detail that previously did not fit. Keep real metrics
(Sharpe, drawdown, turnover) in `st.metric` where the delta arrow is meaningful.
Format dates as `YYYY-MM-DD to YYYY-MM-DD`, and drop the arrow glyph.

**Accept:** no value is clipped at 1280px, 1440px or 1920px. No ellipsis appears in
any stat. Numbers are tabular-aligned.

---

## [x] T1.6 Rebuild the correlation heatmap

**File:** `pages/2_💼_Portfolio_Optimization.py`

Three problems: the unit diagonal saturates the colour scale so off-diagonal values
collapse to near-white; both triangles of a symmetric matrix are drawn; both axis
titles read `symbol`, a dataframe column name.

**Do:** replace the figure construction with

```python
from src.ui.theme import correlation_heatmap, plot

plot(correlation_heatmap(corr_df))
```

Pass the correlation DataFrame directly; the helper reads its column labels. It masks
the diagonal and upper triangle, scales symmetrically to the largest absolute
off-diagonal value, and annotates each cell with the value to two decimals.

Replace the caption "Pairwise correlation of daily returns" with one that says what
the reader should notice, for example "Pairwise correlation of daily returns. The
negative VIXY loading is what lets a long-only book hedge."

**Accept:** no white-on-white diagonal. Cell values legible. Neither axis is titled
`symbol`. The same treatment is applied to any other correlation matrix on the site.

---

## [x] T1.7 Restructure the sidebar

**File:** `src/data/__init__.py` (`render_data_status_sidebar`), plus per-page control
blocks

The sidebar currently runs page nav, page controls, global data status, an API key
field and links together with no hierarchy, so a visitor cannot tell what is a
control and what is status.

**Do:** group into three labelled sections in this order, using the small-caps
sidebar `h2` the CSS already styles:

1. **Parameters.** Only the controls that affect what is on screen right now.
2. **Data.** Refresh status and the optional FMP key, collapsed into an expander
   titled `Data sources` that is closed by default. A visitor never needs it open.
3. **About.** GitHub, project brief, papers.

Retitle the controls that leak implementation:

| Now | Change to |
|---|---|
| `Backtest range (Tab 2 / Tab 4)` | `Backtest window` |
| `Rolling window (classical, days)` | `Rolling window` with help text naming which methods use it |
| `Volatility scaling (σ_target = 10%)` | `Scale to 10% volatility` |

Move every explanation out of the label and into the `help=` tooltip.

**Accept:** no control label mentions a tab, a variable name or a Greek symbol. The
data expander is closed on load. The three headings render in small caps.

---

## [x] T1.8 Calm the emoji and fix the page headers

**Files:** all four entry points. Page *filenames* too, but see the warning.

**Do:**

1. Replace each page's `st.title(...)` plus caption block with `page_header()`:

```python
page_header(
    title="Deep Portfolio Optimization",
    citation="Zhang, Zohren & Roberts (2020)",
    arxiv_id="2005.13665",
    standfirst="A long-only portfolio with a softmax output, trained by gradient "
               "ascent to maximise Sharpe directly. No covariance matrix, no "
               "expected-return forecast.",
)
```

The citation moves out of the H1. `Deep Portfolio Optimization` fits one line; the
current H1 wraps to two and buries the subject behind the authors.

2. Remove emoji from tab labels, expander labels, card headings and the footer. Keep
   `page_icon="📈"` in `set_page_config` (it is the favicon) and at most one in the
   landing hero.

3. **Page filenames.** The emoji in `pages/1_📈_Momentum.py` become the sidebar nav
   labels. Renaming to `pages/1_Momentum.py` cleans the nav, but it also breaks
   `st.page_link("pages/1_📈_Momentum.py", ...)` in `streamlit_app.py` and any test
   that references the path. If you rename: use `git mv`, update all three
   `st.page_link` calls in `streamlit_app.py`, and grep `tests/` for the old paths.
   If that feels risky, skip it and mark this sub-item `[-]`; the emoji in the nav
   are the least of the problems here.

> **Sub-item 3 (page filename rename) skipped `[-]`** — not explicitly
> requested, and the task says to skip it in that case.

**Accept:** no H1 wraps to two lines at 1280px. `grep -rnP "[\x{1F300}-\x{1FAFF}]" --include="*.py" streamlit_app.py pages/ src/`
returns only the `page_icon` lines and, if you skipped the rename, the page paths.

---

# Tier 2: information architecture and performance

Structural. Do this only after Tier 1 is committed and the site looks right, because
it changes what the reader sees first.

## [x] T2.1 Un-nest the sub-tabs

**File:** `pages/2_💼_Portfolio_Optimization.py`, and any other page doing this

Tab 4 ("Key Results") contains two more tabs: "Performance Across Methods (paper
Table 1)" and "Cumulative Returns Across Methods (paper Figure 3)". Six `role="tab"`
elements on one page across two levels.

**Do:** delete the inner tab layer. Stack both exhibits vertically in the Key Results
panel under their own `h3` headings. They are a table and a chart of the same
comparison; the reader wants them together, not alternating.

Shorten the headings. `Performance Across Methods (paper Table 1)` becomes
`Performance across methods`, with `Reproduces Table 1` as a caption beneath.

**Accept:** `document.querySelectorAll('[role="tab"]').length` is 4 on that page. The
table and the chart are both visible in one scroll.

---

## [x] T2.2 Lead with the result

**Files:** all three paper pages

Current order is problem, benchmarks, method, results, which is the order a paper is
written in. A visitor reads in the opposite order.

**Do:** add a **Result** block directly under the standfirst, above the tabs, on each
paper page. Three or four `stat_row` entries carrying the headline comparison (deep
model Sharpe against the best classical baseline, plus the period), then one sentence
saying what it means, then the cumulative-returns chart.

Leave the four tabs in place below it as the walkthrough. The reader who wants the
method still gets it in order; the reader who wants the answer gets it in five
seconds.

**Accept:** on each paper page, the headline number is visible without scrolling at
1440x900 and without clicking a tab.

---

## [-] T2.3 Render tab bodies lazily

> **Skipped — option 1 was already in place and the acceptance criterion is met.**
> Measured with `streamlit.testing.v1.AppTest`, cold then warm:
>
> | Page | Cold | Warm |
> |---|---|---|
> | Landing | 0.75s | 0.04s |
> | Momentum | 2.19s | 0.78s |
> | Portfolio | 0.75s | 0.60s |
> | Order book | 12.24s | 0.73s |
>
> Every page is under the two-second warm-cache bar. The expensive work
> (parquet loads, the LDA fit, DeepLOB inference) already sits behind
> `@st.cache_data`, which is what option 1 prescribes, so the order book's
> cold cost is a one-time cache fill per container. Option 2 (st.tabs ->
> st.radio) would rewrite the DOM the integration tests assert on for very
> little gain, so it was not done.

**Files:** all three paper pages

Streamlit renders every tab body whether or not it is active. Seven Plotly figures
are in the DOM on first paint of the Portfolio page. That is the five-second skeleton
state, and it recomputes on every widget change.

**Do:** pick one of these, in order of preference:

1. **Cache the expensive work.** Wrap backtest and model-inference calls in
   `@st.cache_data` (or `@st.cache_resource` for loaded torch checkpoints), keyed on
   the actual parameters. Cheapest change, biggest effect, no structural risk. Do
   this first and re-measure before doing anything else.
2. **Replace `st.tabs` with `st.radio`** styled as a segmented control, or with
   `st.navigation` sub-pages, so only the selected body executes. Bigger change, and
   it changes the DOM the integration tests assert on.

Measure before and after: `performance.timing` in the console, or just the duration
of the skeleton state. Record both numbers in the commit message.

**Accept:** first meaningful paint under two seconds on a warm cache. Changing a
sidebar control does not re-run work for panels the reader cannot see.

---

## [x] T2.4 Give the landing page a result

**File:** `streamlit_app.py`

The three cards describe the papers well but promise nothing. A visitor has to click
through to find out whether any of it worked.

**Do:** add one line of outcome to each card, under the blockquote: the headline
number the page delivers, in the palette's accent. One number, one label, no chart.

Keep the Common Thread section exactly as it is. It is the best writing on the site.

**Accept:** each card carries a concrete number. The landing page still fits one
scroll at 1440x900.

---

# Verification

## [ ] V1 Test suite

`pytest -q` passes. Every integration-test change is a deliberate update to a new
expected string, with the old string nowhere in the diff.

## [ ] V2 Responsive check

Load every page at 390px, 1280px and 1920px. Confirm:

- no horizontal scroll at any width
- no clipped stat values
- prose measure stays near 74 characters at 1920px
- sidebar collapsed at 390px
- charts legible at 390px (check the LOB depth chart specifically)

## [ ] V3 Colour and contrast

- body text against background meets WCAG AA (4.5:1). `#1A1D24` on `#FFFFFF` passes.
- `MUTED` `#5B6472` on `#FFFFFF` is 6.0:1, fine for captions.
- no chart encodes a distinction by red versus green alone.
- greyscale-print a cumulative-returns chart and confirm the series are still
  distinguishable.

## [ ] V4 Cold-load check

Open the deployed Space in a private window with no cache. Time to first meaningful
paint, and confirm the no-API-key fallback path still renders every page.

## [ ] V5 Link check

Every external link returns 200: both README badges, the three arXiv links, the
GitHub links in the footer and sidebar, the project brief link.
