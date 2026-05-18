# Contract: `src/models/deeplob.py`

**Importers**: `src/training/train_deeplob.py`, `pages/3_📖_Order_Book.py`,
unit tests.

## Public surface

```python
from src.models.deeplob import (
    DeepLOB,         # CNN + Inception + LSTM, paper §III
    LOBSimpleMLP,    # baseline flat MLP
    LOBCNN_I,        # Tsantekidis 2017 CNN-I
    LOBCNN_II,       # Tsantekidis 2017 CNN-II
    LOBLSTM,         # baseline LSTM
)
```

## Common contract — all 5 models

- `__init__(n_classes=3, lookback=100, ...)` — n_classes fixed at 3
  (down / stationary / up).
- `forward(x: torch.Tensor) -> torch.Tensor`:
  - Input `x` shape: `(B, 1, T=lookback, 40)` for DeepLOB / CNN-I / CNN-II
    (channels-first conv layout), or `(B, T, 40)` for LSTM, or
    `(B, T*40=4000)` for SimpleMLP.
  - Output shape: `(B, 3)` — Softmax probabilities, sums to 1 ± 1e-6.
- Device-agnostic (Principle II) — no `.cuda()`, no hardcoded device.
- Exposes `.arch_name` attribute matching the sidecar JSON's `arch` field.

## `DeepLOB`

Architecture per research.md §R1. ~140k params.

```python
class DeepLOB(nn.Module):
    arch_name = "DeepLOB"
    def __init__(self, n_classes=3, lookback=100):
        ...
    def forward(self, x):  # x: (B, 1, T, 40) → (B, 3)
        ...
```

## `LOBSimpleMLP`

```python
class LOBSimpleMLP(nn.Module):
    arch_name = "MLP"
    def __init__(self, n_classes=3, lookback=100, hidden=64):
        # input dim = lookback * 40
        ...
    def forward(self, x):  # x: (B, T*40) → (B, 3)
        ...
```

## `LOBCNN_I`, `LOBCNN_II`, `LOBLSTM`

Same `arch_name` convention. Forward signatures per common contract.

## Forward-pass smoke (per `test_deeplob_models.py`)

For each model:

```python
x_2d = torch.zeros(2, 1, 100, 40)       # for DeepLOB / CNN-I / CNN-II
x_seq = torch.zeros(2, 100, 40)          # for LSTM
x_flat = torch.zeros(2, 4000)            # for MLP
out = model(appropriate_x)
assert out.shape == (2, 3)
assert torch.isfinite(out).all()
assert ((out >= 0) & (out <= 1)).all()
row_sums = out.sum(dim=1)
assert torch.allclose(row_sums, torch.ones(2), atol=1e-6)
```
