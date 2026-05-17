# Quickstart: Phase 0 — Skeleton & Data Foundation

**Audience**: developers running the Phase 0 deliverable for the first time.
**Prerequisites**: Python 3.11, `git`, optionally an FMP Premium API key.

This walkthrough exercises all three Phase 0 user stories end-to-end so a
reviewer can validate the feature without reading the test suite.

---

## US1 — Fresh-clone landing page (P1)

**Goal**: prove `streamlit run streamlit_app.py` works on a clean checkout
with no API key and no parquet files.

```bash
git clone https://github.com/<owner>/deep-finance-showcase
cd deep-finance-showcase
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Expected**:

- Browser opens at `http://localhost:7860`.
- Landing page renders within 30 seconds of the first request (SC-001).
- Three large paper cards visible (DeepLOB, Deep Momentum Networks, Deep
  Portfolio Optimization) (SC-002).
- "Common Thread" section present.
- Sidebar shows one row per universe with `⚠ Bundled CSV fallback — run
  scripts/fetch_data.py for live data` for the ETF and 20-stock universes
  (those are the ones with CSV fallbacks); the 100-stock universe shows
  `⊝ Not yet available` until OD-2 confirmation lands.
- Clicking any paper card or page in the multi-page nav loads a placeholder
  page with `Coming in Phase N` text and the corresponding arXiv link.

---

## US2 — FMP refresh (P2)

**Goal**: prove the fetch script produces three parquets and the landing
page reflects the new data without operator intervention.

### One-time FMP key setup

```bash
cp .env.example .env
# Edit .env, set FMP_API_KEY=<your_premium_key>
```

### One-time data-lake setup (developer-local, optional)

If you want fetched parquets to land in your existing data lake
(co-locating with `/data_lake/data_bento/cme/`) instead of the repo's
`./data/`, set the env var in your shell or `.env`:

```bash
# Pick one; both work
export DEEP_FINANCE_DATA_DIR=~/data_lake/deep-finance/
# OR add DEEP_FINANCE_DATA_DIR=~/data_lake/deep-finance/ to .env
mkdir -p "$DEEP_FINANCE_DATA_DIR"
```

When the variable is unset (the default), parquets land in `./data/`
directly — which is what HF Space sees. The bundled CSV fallbacks
(`data/aapl.csv`, `data/portfolio_data.csv`) are not affected by this
variable; they always resolve relative to the repo root.

### Confirm the 100-stock universe (OD-2)

```bash
# First pass — script writes the proposed list and exits with code 5
python scripts/fetch_data.py --universe sp500_100
cat data/sp500_100.proposed.json   # review the 100 sector-balanced tickers
# Accept as-is:
mv data/sp500_100.proposed.json data/sp500_100.confirmed.json
# Or edit inline before renaming.
```

### Fetch all three universes

```bash
python scripts/fetch_data.py
```

**Expected stdout**:

```
[fetch_data] etf_basket: 7842 rows, 4 symbols, 2011-01-04 → 2026-05-15 (shares_outstanding: yes)
[fetch_data] sp500_20:   158430 rows, 20 symbols, 1992-11-25 → 2026-05-15 (shares_outstanding: yes)
[fetch_data] sp500_100:  757000 rows, 100 symbols, 1995-01-03 → 2026-05-15 (shares_outstanding: yes)
```

(Row counts and date ranges are illustrative.) If your FMP tier does NOT
include shares-outstanding (per OD-3), the closing parenthetical reads
`(shares_outstanding: no)` and a one-line WARNING precedes the universe's
summary — exit code is still 0.

### Verify the app picks up the parquets

```bash
# If you set DEEP_FINANCE_DATA_DIR earlier, the live app must see the same value:
export DEEP_FINANCE_DATA_DIR=~/data_lake/deep-finance/
streamlit run streamlit_app.py
```

**Expected**:

- Sidebar now shows `✓ {universe.label} — Refreshed YYYY-MM-DD` for ETF,
  20-stock, and 100-stock (SC-003).
- Fallback notice substring `Bundled CSV fallback` is absent for those
  three rows.

### Publish to HF Space (when ready to deploy)

The fetched parquets live in your data lake; HF Space sees only what is
committed to the repo at `./data/`. Before pushing, copy them in:

```bash
cp -v "$DEEP_FINANCE_DATA_DIR"/*.parquet ./data/
cp -v "$DEEP_FINANCE_DATA_DIR"/*.parquet.json ./data/   # sidecar metadata
cp -v "$DEEP_FINANCE_DATA_DIR"/sp500_100.confirmed.json ./data/  # if present
git add data/*.parquet data/*.json
git commit -m "data: refresh parquets $(date +%Y-%m-%d)"
git push  # triggers the HF sync workflow (lands in Phase 4)
```

HF Space's runtime has `DEEP_FINANCE_DATA_DIR` unset, so it falls back to
`./data/` (now containing the publish-step parquets) and serves real data.

### Error-path check

```bash
unset FMP_API_KEY   # or invalidate the key in .env
python scripts/fetch_data.py
echo "exit code: $?"  # expect 1
ls data/*.parquet  2>/dev/null  # expect existing parquets untouched; no partial writes
```

---

## US3 — Extended metrics with test coverage (P3)

**Goal**: prove the extended `report_metrics` returns all nine keys, the
three original keys are unchanged against a fixed-seed series, and the
landing page integration test passes in all four data states.

```bash
# Unit tests
pytest tests/unit/test_metrics.py -v
```

**Expected**: all assertions pass. Key tests:

- `test_original_keys_regression` — `annual_ret`, `annual_std`,
  `annual_sharpe` return values byte-identical to the pre-extension version
  for `rng.normal(0.0005, 0.01, 1000)` with seed 0 (the "original-keys
  regression invariant" from `contracts/metrics_report_schema.md`).
- `test_extended_keys_present` — all six new keys are in the returned dict.
- `test_extended_keys_finite` — every value is a finite float.
- `test_extended_keys_in_tolerance_bands` — values land in the tolerance
  bands documented in §13.7 of the brief (SC-004).

```bash
# Integration test
pytest tests/integration/test_landing_page.py -v
```

**Expected**: four parameterized cases pass:

| Case | parquet | FMP key | Result |
|---|---|---|---|
| A | absent  | unset | ✓ |
| B | absent  | set   | ✓ |
| C | present | unset | ✓ |
| D | present | set   | ✓ |

(SC-005.)

### Optional: full sweep

```bash
pytest -v
```

Should exit 0 with all unit + integration tests passing.

---

## Acceptance check (run before declaring Phase 0 done)

```bash
# Layout conformance (SC-006)
ls streamlit_app.py pages/ src/ data/ scripts/ notebooks/reference/data/limit_order_book_data.csv \
   tests/unit/test_metrics.py tests/integration/test_landing_page.py \
   .streamlit/config.toml requirements.txt requirements-train.txt \
   .env.example .gitignore Dockerfile LICENSE README.md packages.txt

# All eight Phase 0 acceptance criteria (from spec.md)
# A1: fresh-clone app runs — manual verification per US1 above
# A2: landing page + 3 nav links + Common Thread — manual verification per US1
# A3: parquets after fetch + sidebar reflects them — manual verification per US2
# A4: pytest tests/unit/test_metrics.py passes
# A5: pytest tests/integration/test_landing_page.py passes
# A6: fallback notice visible — manual verification per US1
# A7: requirements.txt resolves on 3.11 CPU — verified by the `pip install` above
# A8: §5 layout — verified by the `ls` line above
```

When all of those are green, Phase 0 is done and Phase 1 can begin.
