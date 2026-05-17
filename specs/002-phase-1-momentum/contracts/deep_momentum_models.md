# Contract: `src/models/deep_momentum.py`

**Importer**: `src/training/train_deep_momentum.py`, `pages/1_📈_Momentum.py`,
`tests/unit/test_checkpoint_smoke.py`.

## Public surface

```python
from src.models.deep_momentum import (
    DeepMomentumMLP,    # 2-layer MLP, hidden=20 per paper Table 3
    DeepMomentumLSTM,   # 1-layer LSTM, hidden=10 per paper Table 3
)
```

## `DeepMomentumMLP`

```python
class DeepMomentumMLP(nn.Module):
    def __init__(self, seq_length: int = 60, n_features: int = 8,
                 hidden_size: int = 20):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_length, n_features)
        # returns: (batch, 1) in (-1, +1) via Softsign
        ...
```

**Architecture** (per brief §13.3 + paper Table 3):

```text
input (B, T, F=8)
  → Flatten              → (B, T*F)
  → Linear(T*F, 64)      → Softsign
  → Linear(64, 32)       → Softsign
  → Linear(32, 1)        → Softsign
  → output (B, 1) in (-1, +1)
```

**Note on hidden sizes**: Brief §13.3 lists 64→32→1 as the source-notebook
starter; the paper's Table 3 reports "hidden ~20" — the source notebook
authors used a deeper / wider MLP than the paper's nominal description.
Phase 1 inherits the brief's starter dimensions (64→32→1) to keep
notebook-to-app parity; the discrepancy with the paper's "~20" is
disclosed in `pages/1_📈_Momentum.py` Tab 3 model-card badge.

## `DeepMomentumLSTM`

```python
class DeepMomentumLSTM(nn.Module):
    def __init__(self, seq_length: int = 60, n_features: int = 8,
                 hidden_size: int = 10, num_layers: int = 1):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_length, n_features)
        # returns: (batch, 1) in (-1, +1) via Softsign on the last time step
        ...
```

**Architecture**:

```text
input (B, T=60, F=8)
  → LSTM(F=8, hidden=10, num_layers=1, batch_first=True)
  → take last-time-step hidden state (B, 10)
  → Linear(10, 1)        → Softsign
  → output (B, 1) in (-1, +1)
```

**Hidden size 10**: matches paper Table 3 exactly.

## Invariants (Constitution Principle II — NON-NEGOTIABLE)

Both classes MUST:

1. Contain NO `.cuda()` calls.
2. Contain NO hardcoded `device='cuda'`.
3. Contain NO `torch.cuda.synchronize` or other GPU-only operations.
4. Inherit device from the input tensor (no `Tensor` creation inside
   `forward` that synthesizes a new device).

Any new tensor allocated inside `forward` (e.g., a zero pad) MUST be
created via `torch.zeros_like(x)` or with explicit `device=x.device`,
never with a hardcoded device.

## Forward-pass smoke test

`tests/unit/test_checkpoint_smoke.py` instantiates each class with default
hyperparameters, loads the corresponding `.pt` from `data/pretrained/`,
and runs:

```python
model.eval()
x = torch.zeros(1, 60, 8)            # CPU; no device kwarg
out = model(x)
assert out.shape == (1, 1)
assert torch.isfinite(out).all()
assert -1 - 1e-6 <= out.item() <= 1 + 1e-6   # Softsign bounds
```

All three asserts MUST pass.

## Parameter counts (informational; verified loosely)

- `DeepMomentumMLP` with (T=60, F=8, hidden cascade 64→32→1): ~ (60*8)*64
  + 64*32 + 32*1 ≈ 33 k params.
- `DeepMomentumLSTM` with (F=8, hidden=10): 4 * (10*8 + 10*10 + 10) ≈ 760
  params (LSTM has 4 gates × W + U + b parameter sets).

Per brief §7.1, the LSTM is the smaller model — counts above are
consistent.
