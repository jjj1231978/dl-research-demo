# Contract: `src/training/train_deeplob.py`

**Type**: Modal app + plain Python module.
**Runs on**: Modal T4 (real) OR local CPU (smoke).

Mirrors Phase 1/2 trainers — Modal scaffolding at top, device-agnostic
`train()` body at bottom.

## CLI

| Form | Use case |
|---|---|
| `modal run src/training/train_deeplob.py --arch DeepLOB` | T4 training, headline model |
| `modal run src/training/train_deeplob.py --arch MLP` | T4 training, baseline |
| `modal run src/training/train_deeplob.py --arch CNN1` | (same) |
| `modal run src/training/train_deeplob.py --arch CNN2` | (same) |
| `modal run src/training/train_deeplob.py --arch LSTM` | (same) |
| `python -m src.training.train_deeplob --arch DeepLOB --max-epochs 1` | CPU smoke |

Flags:

| Flag | Type | Default |
|---|---|---|
| `--arch` | `DeepLOB \| MLP \| CNN1 \| CNN2 \| LSTM` | `DeepLOB` |
| `--max-epochs` | int | `100` |
| `--resume-from` | path | `None` |
| `--data-dir` | path | `data_root()` |
| `--checkpoint-dir` | path | `data/pretrained/` |
| `--demo-data` | flag | off | Use `lob_fi2010_demo.parquet` instead of full benchmark (for CPU smoke) |
| `-v` | flag | off |

## `train()` body contract

```python
def train(
    data_dir: Path,
    arch: Literal["DeepLOB", "MLP", "CNN1", "CNN2", "LSTM"],
    device: torch.device,
    checkpoint_dir: Path,
    max_epochs: int = 100,
    resume_from: str | None = None,
    modal_image_id: str | None = None,
    demo_data: bool = False,
) -> dict[str, float]:
```

**Required behaviour**:

1. Load `data_dir / ("lob_fi2010_demo.parquet" if demo_data else "lob_fi2010.parquet")`.
2. Split into train (days 1-7) + test (days 8-10) per FI-2010 Setup 2.
3. Build sliding-window tensors:
   - For each starting tick `t`, window = features at `[t, t+lookback)`
   - Target = `label_k10` at the END of the window (the label aligned with
     the prediction horizon)
   - Drop windows where any feature is NaN
4. Per arch, instantiate model + shape input appropriately:
   - DeepLOB / CNN-I / CNN-II: input `(B, 1, T, 40)`
   - LSTM: input `(B, T, 40)`
   - MLP: input `(B, T*40)`
5. Train with `Adam(lr=1e-3)`, `nn.CrossEntropyLoss()`, batch_size=64.
6. EarlyStopping (canonical `src.early_stopper.EarlyStopping`) on
   validation loss. Save best `.pt` to
   `checkpoint_dir / f"{arch_lower}_fi2010_k10.pt"`.
7. After training, compute test metrics: Accuracy + Precision (macro) +
   Recall (macro) + F1 (macro) + 3×3 confusion matrix.
8. Write sidecar JSON per data-model E3.
9. Return final test metrics dict.

## Checkpoint filename convention

```
{arch.lower()}_fi2010_k10.pt
```

So: `deeplob_fi2010_k10.pt`, `mlp_fi2010_k10.pt`, etc.
