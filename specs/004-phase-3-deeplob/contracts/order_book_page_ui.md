# Contract: `pages/3_📖_Order_Book.py`

**Type**: Streamlit page; replaces Phase 0 placeholder.

## Page header

- Title: `📖 Order Book — Zhang, Zohren, Roberts (2019)`
- Subtitle: *"DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"*, IEEE TSP.
- arXiv: 1808.03668
- Elevator pitch: "A CNN + Inception + LSTM that learns universal LOB
  microstructure features for 3-class mid-price-movement prediction."

## Sidebar

| Widget | `key` | Default |
|---|---|---|
| `st.selectbox` "Prediction horizon (k)" | `lob_horizon_k` | `10` (only k=10 reproduced; others are paper-reported) |
| `st.selectbox` "Classical baseline" | `lob_classical` | `"LDA"` |
| `st.slider` "Tick index in demo slice" | `lob_tick` | middle of demo slice |
| `st.divider` |  |  |
| `render_data_status_sidebar` | (re-used) |  |

## Tabs

`st.tabs(["1. Problem & Data", "2. Classical Baselines",
"3. DeepLOB", "4. Table II Replication"])`

### Tab 1 — Problem & Data

- One paragraph: LOB definition, mid-price prediction, FI-2010 as benchmark
- LOB snapshot at selected tick: 10 levels each side, Plotly stacked bars
- Class-balance summary across (stock, k) — Plotly stacked bar
- Smoothed-label visualisation: mid-price line with up/down shading per Eq. 4

### Tab 2 — Classical Baselines

- Selected baseline (sidebar) — show predictions on demo slice
- Per-baseline metrics card (accuracy / F1)

### Tab 3 — DeepLOB

Three sub-tabs:

- **3A — Architecture**: ASCII art / markdown diagram of CNN-Inception-LSTM
- **3B — Live Prediction Demo**: scrub through demo slice, see
  predicted class probabilities + realized label + running accuracy
- **3C — Train your own**: surface CLI command (Constitution III)

### Tab 4 — Table II Replication

`st.tabs(["4A. Table II", "4B. Confusion Matrices", "4C. Per-class F1"])`

- **4A**: 6-row × 4-column dataframe (5 reproduced + 1 reference); columns
  Accuracy / Precision / Recall / F1 (×100). Best F1 bolded.
  "(reproduced here)" / "(paper-reported)" badge per row.
- **4B**: 3×3 confusion matrices side-by-side for top 3 methods
- **4C**: Per-class F1 grouped bar chart

## Math panel

- Mid-price formula (paper Eq. 1)
- Smoothed-label formula (paper Eq. 4)
- Gated activation pattern in conv blocks
- Inception parallel-filter idea

## Code panel

- `inspect.getsource(DeepLOB)`
- `inspect.getsource(LOBSimpleMLP)`

## Render assertions (4 parametrised cases per FR-022)

| Case | demo_slice | checkpoints | Assertion |
|---|---|---|---|
| A | absent | absent | Setup-instruction banner; no exception |
| B | absent | present | Setup-instruction banner; no exception |
| C | present | absent | Tab 2/3 show demo; Tab 3 model card "missing checkpoint" warning |
| D | present | present | All four tabs functional; Tab 4A 6-row table |

## Copy-text invariants

- Setup-instruction banner: contains `"lob_fi2010.parquet"` AND
  `"scripts/fetch_lob_fi2010.py"`
- Substrate disclosure (Tab 1): contains `"FI-2010"` AND `"Ntakaris et al."`
