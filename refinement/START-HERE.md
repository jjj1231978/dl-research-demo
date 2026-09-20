# Start here

A refinement package for `dl-research-demo`, built to be handed to Claude CLI.

## What is in it

```
refinement/
  START-HERE.md              this file
  BRIEF.md                   findings, evidence, and the visual system spec
  TASKS.md                   18 tasks in 3 tiers, each independently committable
  assets/
    streamlit-config.toml    drop-in replacement for .streamlit/config.toml
    theme.py                 new file: src/ui/theme.py
    README.md.draft          replacement body for the root README
```

## Install

From the repo root:

```bash
git checkout -b refine/visual-system
unzip ~/Downloads/dl-research-demo-refinement.zip -d .
git add refinement && git commit -m "docs: add refinement package"
```

The package lives in the repo so the CLI can read it and tick off `TASKS.md` as it
goes. Delete the folder, or add it to `.gitignore`, once the work is merged.

## Kick off the CLI

```bash
cd /path/to/dl-research-demo
claude
```

Then paste this:

> Read `refinement/BRIEF.md` and `refinement/TASKS.md`. You are implementing this
> package.
>
> Work one task at a time, in order, starting with T0.1. For each task: make the
> change, run `pytest -q`, show me the diff, and wait for me to approve before you
> commit. Do not start the next task until I say go.
>
> Hard constraints:
> - Presentation only. Nothing under `src/models/`, `src/strategies/`,
>   `src/training/` or `src/metrics.py` changes. If a task appears to need it, stop
>   and tell me.
> - `tests/integration/test_*_page.py` asserts on UI strings and will break when
>   labels change. Update the assertion to the new expected string in the same
>   commit. Never weaken an assertion to make a test pass.
> - Tick the checkbox in `TASKS.md` as each task lands.
>
> Start with T0.1 and stop when it is ready for review.

## If you would rather move faster

Tier 0 is six small, independent tasks with no app risk. You can let the CLI run all
of Tier 0 unattended and review the single diff at the end:

> Read `refinement/BRIEF.md` and `refinement/TASKS.md`. Implement all of Tier 0
> (T0.1 through T0.6), skipping T0.3 for now. Run `pytest -q` after each task. Commit
> each one separately using the message format in TASKS.md. Then show me the full
> diff and stop.

Tier 1 and Tier 2 are worth reviewing task by task, because they are judgement calls
about how the site reads and you will want to see each one on screen.

## Suggested order across sittings

| Sitting | Tasks | Roughly |
|---|---|---|
| 1 | T0.1, T0.2, T0.4, T0.5, T0.6 | 1 hour, and the Space card stops saying "Phase 0" |
| 2 | T1.1, T1.2, T1.3 | 1 hour, and the site changes character |
| 3 | T1.4, T1.5, T1.6 | 2 hours, charts and stats |
| 4 | T1.7, T1.8 | 1 hour, sidebar and headers |
| 5 | T0.3, V1 to V5 | 1 hour, screenshot and verification |
| 6 | Tier 2 | half a day, optional |

Stopping after sitting 4 leaves you with a site that looks finished. Tier 2 is the
difference between looking finished and being genuinely good to read.

## Two things worth deciding yourself

**The results-first restructure (T2.2).** It puts the headline number above the
walkthrough on every paper page. If you built this as a teaching artifact, where the
walk from problem to method to result is the point, then T2.2 works against you and
you should skip it. If you built it as evidence of what you can do, T2.2 is the most
valuable task in the package.

**The page filename rename (T1.8, step 3).** Dropping the emoji from
`pages/1_📈_Momentum.py` cleans the sidebar nav, but it touches three `st.page_link`
calls and any test that references the path. Small reward, real breakage risk. The
task tells the CLI to skip it if you do not explicitly ask for it.

## Where the findings came from

Everything in `BRIEF.md` was measured against the deployed Space on 2026-09-20, not
inferred from the source: computed CSS values from the DOM, the rendered text of the
truncated metrics, the count of `role="tab"` elements, the 401 on the badge URL, and
the Plotly figures present in the DOM on first paint. Section 2.4 notes one earlier
claim that turned out to be a misread and has been corrected.
