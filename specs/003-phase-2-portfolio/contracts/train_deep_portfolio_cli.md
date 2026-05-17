# Contract: `src/training/train_deep_portfolio.py`

**Type**: Modal app + plain Python module, importable for unit smoke tests.
**Runs on**: Modal T4 (real) OR local CPU (smoke / unit tests). Constitution
Principle II requires both modes to work without code changes.

Mirrors Phase 1's `train_deep_momentum.py` structure exactly — same Modal
scaffolding at top, same device-agnostic `train()` body at bottom — only
the universe-dispatch and the data-shape differ.

## File structure

```python
# Top: Modal scaffolding
import modal
app = modal.App("deep-finance-train-portfolio")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-train.txt")
    .add_local_python_source("src")
)
volume = modal.Volume.from_name("dl-research-data", create_if_missing=True)

@app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=3600)
def train_remote(universe: str = "etfs", max_epochs: int = 200, resume_from=None):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return train(data_dir=Path("/data"), universe=universe, device=device,
                 checkpoint_dir=Path("/data/pretrained"),
                 max_epochs=max_epochs, resume_from=resume_from)

@app.local_entrypoint()
def main(universe: str = "etfs", max_epochs: int = 200, resume_from=None):
    metrics = train_remote.remote(universe=universe, max_epochs=max_epochs,
                                  resume_from=resume_from)
    print(f"[train_deep_portfolio] universe={universe} metrics={metrics}")
    ...

# Bottom: device-agnostic training body
def train(data_dir, universe, device, checkpoint_dir,
          max_epochs=200, resume_from=None, modal_image_id=None) -> dict:
    ...
```

## CLI surface (two equivalent forms)

| Form | Use case | Where it runs |
|---|---|---|
| `modal run src/training/train_deep_portfolio.py --universe etfs` | Real training | Modal T4 |
| `modal run src/training/train_deep_portfolio.py --universe 20stock` | Real training | Modal T4 |
| `python -m src.training.train_deep_portfolio --universe etfs --max-epochs 1` | CPU smoke | Local |

Flags:

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--universe` | `etfs \| 20stock` | `etfs` | Universe selector. |
| `--max-epochs` | int | `200` | Hard cap; early-stopping triggers earlier. |
| `--resume-from` | path | `None` | Resume from a `.pt` under `/data/pretrained/`. |

## `train()` body contract

Pure Python, device-agnostic. Signature:

```python
def train(
    data_dir: Path,                # /data on Modal; data_root() locally
    universe: Literal["etfs", "20stock"],
    device: torch.device,
    checkpoint_dir: Path,
    max_epochs: int = 200,
    resume_from: str | None = None,
    modal_image_id: str | None = None,
) -> dict[str, float]:
```

**Required behaviour**:

1. Load `data_dir / f"{universe_to_parquet_name(universe)}.parquet"`.
2. Apply per-universe chronological split (TEST_START = 2020-01-01 for
   both; train end = 2019-12-31).
3. Build per-day feature tensors via `_build_per_day_features_portfolio()`
   (shape: (n_days, lookback=50, 2*n_assets)). Features per (day, asset):
   close, daily_return. Stacked across assets.
4. Per-day target: next-day returns vector (n_days, n_assets).
5. DataLoader with `batch_size_days=32`, shuffle=True for train.
6. Instantiate `DeepPortfolioMLP(n_assets, lookback=50, hidden_size=64)`.
   Move to device.
7. Optimizer: `Adam(lr=1e-3)`. Loss = canonical `Neg_Sharpe`.
8. Per-batch step: forward all (B*1, lookback, 2N) → weights (B, N) →
   portfolio_returns_per_day = `(weights * yb).sum(dim=1)` → `Neg_Sharpe`.
   No mask — Portfolio always uses all assets.
9. `EarlyStopping(savepath=str(final_path), patience=25, min_delta=1e-4)`.
10. Save final `.pt` to
    `checkpoint_dir / f"deep_portfolio_{universe}.pt"`; sidecar JSON per
    data-model §E3.
11. Return final test metrics dict.

**Forbidden** (Constitution II):
- `.cuda()` calls.
- Hardcoded `device='cuda'`.

## Universe → parquet name mapping

```python
_PARQUET_MAP = {"etfs": "etf_basket", "20stock": "sp500_20"}
```

Why this mapping: matches `src/universes.py` keys; matches CLI tradition
(`--universe etfs` is shorter than `--universe etf_basket`).

## Reproducibility

`torch.manual_seed(42)` + `numpy.random.seed(42)` at function entry. Same
Phase 1 precedent (research.md R6).

## Sidecar JSON

Written after final save. Schema per `data-model.md` §E3. `trained_with`
populated from `_modal_image_id_or_local()` helper imported from
`src.training.train_deep_momentum` (DRY — single helper).
