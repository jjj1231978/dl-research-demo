# Contract: `src/training/train_deep_momentum.py`

**Type**: Modal app + plain Python module, importable for unit smoke tests.
**Runs on**: Modal T4 container (real training) OR local CPU (smoke
runs / unit tests). Constitution Principle II requires both modes to
work without code changes.

## File structure (per constitution v1.1.0 §"Training workflow (Modal)")

```python
# src/training/train_deep_momentum.py

# ── Top: Modal scaffolding ──
import modal
app = modal.App("deep-finance-train-momentum")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-train.txt")
    .add_local_python_source("src")
)
volume = modal.Volume.from_name("dl-research-data", create_if_missing=True)

@app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=3600)
def train_remote(arch: str = "MLP", max_epochs: int = 200,
                 resume_from: str | None = None) -> dict:
    import torch
    from src.training.train_deep_momentum import train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return train(data_dir=Path("/data"), arch=arch, device=device,
                 checkpoint_dir=Path("/data/pretrained"),
                 max_epochs=max_epochs, resume_from=resume_from)

@app.local_entrypoint()
def main(arch: str = "MLP", max_epochs: int = 200,
         resume_from: str | None = None):
    metrics = train_remote.remote(arch=arch, max_epochs=max_epochs,
                                  resume_from=resume_from)
    print(f"[train_deep_momentum] arch={arch} metrics={metrics}")
    print(f"To pull the checkpoint locally:")
    fname = f"{arch.lower()}_sharpe.pt"
    print(f"  modal volume get dl-research-data /pretrained/{fname} ./data/pretrained/{fname}")

# ── Bottom: device-agnostic training body ──
def train(data_dir, arch, device, checkpoint_dir, max_epochs=200,
          resume_from=None):
    """Plain Python training loop. Importable + callable on CPU for unit tests."""
    # ... see body contract below ...
```

## CLI surface (two equivalent forms)

| Form | Use case | Where it runs |
|---|---|---|
| `modal run src/training/train_deep_momentum.py --arch MLP` | Real training | Modal T4 container |
| `modal run src/training/train_deep_momentum.py --arch LSTM` | Real training | Modal T4 container |
| `python -m src.training.train_deep_momentum --arch MLP --max-epochs 1` | CPU smoke run | Developer's machine |

Common flags:

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--arch` | `MLP \| LSTM` | `MLP` | Architecture selector. |
| `--max-epochs` | int | `200` | Hard cap on training epochs (early stopping triggers earlier). |
| `--resume-from` | path | `None` | Path under `/data/pretrained/` to resume from a previous container preemption. |

## `train()` body contract

Located at the bottom of the file. Pure Python, device-agnostic. Signature:

```python
def train(
    data_dir: Path,                # /data on Modal; whatever locally
    arch: Literal["MLP", "LSTM"],
    device: torch.device,          # "cuda" on Modal, "cpu" locally
    checkpoint_dir: Path,
    max_epochs: int = 200,
    resume_from: str | None = None,
) -> dict[str, float]:             # returns final test metrics
```

**Required behaviour**:

1. Load `cme_futures.parquet` from `data_dir / "cme_futures.parquet"`.
2. Apply the 60/40 chronological split from
   `src/data/futures/__init__.py:TRAIN_END = 2020-01-01`.
3. Build feature tensors per FR-008: 5 normalized return horizons (1d,
   1m, 3m, 6m, 1y) + 3 MACD timescales (8/24, 16/48, 32/96). Wrap in
   the canonical `src/torch_data.py:MyDataset`.
4. Instantiate `DeepMomentumMLP` or `DeepMomentumLSTM` per `arch`. Move
   to `device`.
5. Train with `Adam(lr=1e-3, batch_size=128)`, loss =
   `src/losses.py:SharpeLoss` used as-is (Constitution Principle V).
   For each training step: forward → multi-asset portfolio aggregation
   (per `src/losses.py` comment at line 12–17 for multi-asset usage)
   → loss → backward.
6. Use the canonical `src/early_stopper.py:EarlyStopping(savepath=...,
   patience=25, min_delta=1e-4)` for checkpoint management.
7. On Modal, the savepath MUST be inside `checkpoint_dir` (which is the
   Volume mount); call `volume.commit()` after each checkpoint write so
   a preemption preserves progress.
8. After training, write the final checkpoint to
   `checkpoint_dir / f"{arch.lower()}_sharpe.pt"` and the sidecar JSON
   to `checkpoint_dir / f"{arch.lower()}_sharpe.json"` per the
   `DeepMomentumCheckpoint` schema in `data-model.md` §E3.
9. Return the final test metrics dict (same keys as the sidecar's
   `final_metrics` block).

**Forbidden** (Constitution Principle II):
- `.cuda()` calls.
- Hardcoded `device='cuda'`.
- `torch.cuda.synchronize` or other GPU-only ops.

## Sidecar JSON

Written by `train()` after final checkpoint save. Schema per
`data-model.md` §E3. The Modal image hash (`trained_with` field) is
populated by inspecting `modal.functions.current_function_call` at
runtime; on local CPU smoke runs the field gets the literal value
`"local CPU smoke run"` to make the provenance unambiguous.

## Exit codes (when invoked via `modal run`)

Modal exits 0 if `train_remote.remote()` returns; non-zero otherwise
(container error). The script does not define custom exit codes — Modal's
function-result protocol owns this surface.

## Logging

`logging.getLogger(__name__)` at WARNING by default; `-v` lifts to INFO.
Per-epoch loss and per-N-epoch checkpoint-saved messages at INFO. No
DEBUG.

## Idempotency

Re-running `modal run … --arch MLP` on the same data, same image, same
hyperparameters produces a *similar but not identical* `.pt` — random
weight initialization differs unless `torch.manual_seed(...)` is called.
The training body MUST set `torch.manual_seed(42); numpy.random.seed(42)`
at function entry to make runs reproducible to the extent CUDA non-
determinism allows.
