# Contract: `src/models/deep_portfolio.py`

**Importer**: `src/training/train_deep_portfolio.py`, `pages/2_💼_Portfolio.py`,
`tests/unit/test_softmax_head.py`, `tests/unit/test_portfolio_checkpoint_smoke.py`.

## Public surface

```python
from src.models.deep_portfolio import DeepPortfolioMLP
```

## `DeepPortfolioMLP`

```python
class DeepPortfolioMLP(nn.Module):
    def __init__(self, n_assets: int, lookback: int = 50, hidden_size: int = 64):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, 2*n_assets) — close + return per asset, time-stacked
        # returns: (batch, n_assets) — Softmax weights, sums to 1
        ...
```

## Architecture (per `02_deep_portfolio_optimization.ipynb` + paper §4.3)

```text
input (B, T=50, 2N)
  → Flatten             → (B, T*2N)
  → Linear(T*2N, 64)    → ReLU
  → Linear(64, 64)      → ReLU
  → Linear(64, N)
  → Softmax(dim=-1)
  → output (B, N) — weights in [0, 1] summing to 1
```

## Parameter counts

| Universe | n_assets | Input dim | Total params |
|---|---|---|---|
| etfs    | 4        | 400       | ~30 k |
| 20stock | 20       | 2,000     | ~135 k |

## Invariants (Constitution Principle II — NON-NEGOTIABLE)

1. No `.cuda()` calls, no hardcoded `device='cuda'`, no GPU-only ops.
2. Any new tensor allocated inside `forward` (e.g., padding) MUST use
   `device=x.device`.
3. Same instance MUST work on CPU (page render, tests) and GPU (Modal
   training) without code changes.

## Output invariants (asserted by `test_softmax_head.py`)

For a random input `x ∈ R^(1×50×2N)`:
- `out.shape == (1, n_assets)`
- `torch.isfinite(out).all()`
- `((out >= 0) & (out <= 1)).all()`
- `abs(out.sum().item() - 1.0) < 1e-6`

## Forward-pass smoke test (per `test_portfolio_checkpoint_smoke.py`)

```python
model = DeepPortfolioMLP(n_assets=4)
state = torch.load("data/pretrained/deep_portfolio_etfs.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
x = torch.zeros(1, 50, 8)  # 50 lookback, 2*4 = 8 features for ETF
out = model(x)
assert out.shape == (1, 4)
assert torch.isfinite(out).all()
assert abs(out.sum().item() - 1.0) < 1e-6
```

## Hidden-size attribute

The instance MUST expose `.hidden_size`, `.n_assets`, `.lookback` as
attributes for the training script's sidecar JSON to read.
