"""Deep Portfolio MLP — Softmax-output network (Zhang et al. 2020).

Per paper §4.3 + brief §13 + `notebooks/reference/02_deep_portfolio_optimization.ipynb`:

    input (B, lookback=50, 2*N)
        — features per asset: close, daily return; lookback days deep, flattened
      → Linear(2*N*lookback, 64) → ReLU
      → Linear(64, 64)            → ReLU
      → Linear(64, N)
      → Softmax(dim=-1)
      → output (B, N): non-negative weights summing to 1 (long-only simplex)

Constitution Principle II (NON-NEGOTIABLE): no .cuda() calls, no hardcoded
device. Module device follows input tensor.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DeepPortfolioMLP(nn.Module):
    """MLP with Softmax head producing long-only portfolio weights.

    Args:
        n_assets: portfolio cardinality (4 for ETF basket, 20 for sp500_20).
        lookback: feature window depth (default 50, paper §4.3).
        hidden_size: hidden cascade width (default 64, paper §4.3).
    """

    def __init__(
        self,
        n_assets: int,
        lookback: int = 50,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.lookback = lookback
        self.hidden_size = hidden_size
        in_dim = lookback * 2 * n_assets  # 2 features per asset per step
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_assets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, lookback, 2*n_assets) → (batch, n_assets).

        Output rows sum to 1 (softmax). All entries in [0, 1].
        """
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)
