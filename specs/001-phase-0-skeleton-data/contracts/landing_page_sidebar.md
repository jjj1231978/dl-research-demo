# Contract: Landing-page sidebar UI

**Implemented in**: `streamlit_app.py` (via the helper
`src.data.render_data_status_sidebar`).
**Consumed by**: `tests/integration/test_landing_page.py` to assert presence
of named widgets.

## Visual order (top to bottom)

1. **Heading**: `## Data status` (markdown level-2 inside the sidebar).
2. **Per-universe rows** (one row per universe in `src.universes.UNIVERSES`,
   rendered in the order ETF → 20-stock → 100-stock → CME). The `aapl_toy`
   universe is intentionally skipped: it is consumed only by the Momentum
   page's single-asset toy fallback (which reads `data/aapl.csv` directly
   via `BUNDLED_CSV_DIR`) and listing it here added noise to every page's
   sidebar.
   - Icon column: `✓` (parquet), `⚠` (CSV fallback), `⊝` (missing).
   - Label column: `Universe.label`.
   - Caption column: refresh date or fallback notice text.
3. **Divider** (`st.divider()`).
4. **Heading**: `## Refresh data`.
5. **FMP key input**: `st.text_input("FMP API key (optional)",
   type="password", key="fmp_api_key_input")`.
6. **Helper caption**: explains that the in-app refresh feature lands in a
   future phase; for now the key is captured but unused (Phase 0 keeps the
   fetch workflow as a developer-local script).
7. **Divider**.
8. **Heading**: `## Links`.
9. **Markdown bullet list**:
   - `🔗 [GitHub repository](https://github.com/<owner>/<repo>)` — URL drawn
     from `src.__init__.GITHUB_REPO_URL`.
   - `📄 [Project brief](Project_brief.md)` (link to the in-repo brief, opens
     in a new tab if the deployment exposes it).

## Required widget keys (used by integration tests)

- `fmp_api_key_input` — the FMP key text input. Tests assert it exists in
  the rendered AppTest state.

## Copy-text invariants (verified by tests/integration/test_landing_page.py)

The CSV-fallback notice text MUST literally contain the substring
`"Bundled CSV fallback"` and the substring `"scripts/fetch_data.py"`. This
satisfies FR-008 (operator guidance) and SC-007 (visible notice naming the
upgrade action). The tests grep the rendered sidebar markdown for both
substrings.

The "100-stock universe pending confirmation" notice (when `sp500_100` is in
`missing` state because OD-2 confirmation has not yet landed) MUST literally
contain `"pending confirmation"`.

## What the sidebar does NOT do

- Does NOT trigger a fetch when an FMP key is entered. Phase 0 deliberately
  defers the in-app refresh button (the fetch script stays
  developer-local). Tests assert the input exists; they do not assert any
  side-effect of typing into it.
- Does NOT display per-page status (e.g., "Phase 1 page complete"). That is
  hard-coded into the page placeholders themselves and is out of the
  sidebar's scope.
- Does NOT show model-checkpoint status. No `.pt` files exist in Phase 0;
  Phase 1+ adds a separate "Model status" sidebar component.

## Render assertions (integration test)

`tests/integration/test_landing_page.py` runs four parameterised cases:

| Case | parquet | FMP key | Sidebar expectations |
|---|---|---|---|
| A | absent | unset | All universes show `⚠` (CSV fallback) or `⊝` (missing); FMP key input is empty; fallback notice substring present for at least one universe. |
| B | absent | set | Same as A but `st.session_state.fmp_api_key_input` carries the set value when the FMP input is interacted with via `at.text_input("fmp_api_key_input").set_value("...")`. |
| C | present | unset | At least one universe shows `✓` with a refresh date; fallback notice absent for that universe; FMP key input is empty. |
| D | present | set | As C, plus key input interaction works. |

All four cases pass when `AppTest.run()` raises no exception and the
assertions above succeed.
