"""Deep-momentum architectures for the Phase 1 Momentum page.

Implements the two architectures from Lim/Zohren/Roberts 2019 §4.2 +
brief §13.3:

- ``DeepMomentumMLP`` — 2-layer MLP, Flatten → 64 → 32 → 1 → Softsign.
  Hidden cascade (64→32→1) inherited from the brief's source-notebook
  starter (matches `notebooks/reference/03_deep_momentum_strategy.ipynb`).
  Paper Table 3 lists "hidden ~20" — disclosed on Tab 3 model-card badge.
- ``DeepMomentumLSTM`` — single-layer LSTM, hidden=10 (paper Table 3
  exact), → Linear(10, 1) → Softsign on the last time step.

Both end with `nn.Softsign` per research.md §R2 — smoother than tanh,
lighter gradient saturation through the Sharpe loss, and matches
brief §13.3's reference code.

Constitution Principle II (NON-NEGOTIABLE): no `.cuda()` calls, no
hardcoded `device='cuda'`. Module device follows the input tensor.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DeepMomentumMLP(nn.Module):
    """Flatten → Linear stack → Softsign output in (-1, +1).

    Args:
        seq_length: lookback window per sample (default 60, paper §4.3).
        n_features: features per timestep (default 8 = 5 normalized
            return horizons + 3 MACD timescales per FR-008).
        hidden_size: kept for API parity with the LSTM and the trainer's
            CLI; the actual hidden cascade is fixed at (64, 32) per
            brief §13.3. The disclosed paper hidden size is ~20 — Tab 3
            model card surfaces the discrepancy.
        dropout: rate applied to the flattened input and to each hidden
            layer during training. Without it the 480-input network
            memorises the training years (in-sample Sharpe above 10, flat
            out of sample). Kept out of ``self.net`` so the state-dict keys
            are unchanged and older checkpoints still load.
    """

    def __init__(
        self,
        seq_length: int = 60,
        n_features: int = 8,
        hidden_size: int = 20,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        in_dim = seq_length * n_features
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 64),
            nn.Softsign(),
            nn.Linear(64, 32),
            nn.Softsign(),
            nn.Linear(32, 1),
            nn.Softsign(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_length, n_features) → (batch, 1) in (-1, +1)."""
        flatten, fc1, act1, fc2, act2, fc3, out = self.net
        h = self.dropout(flatten(x))
        h = self.dropout(act1(fc1(h)))
        h = self.dropout(act2(fc2(h)))
        return out(fc3(h))


class DeepMomentumLSTM(nn.Module):
    """Single-layer LSTM → Linear → Softsign on the last time step.

    Hidden size 10 matches paper Table 3 exactly. Output of the final
    time step is taken; no attention pooling or sequence-level
    averaging — paper §4.2 specifies "last-step output".
    """

    def __init__(
        self,
        seq_length: int = 60,
        n_features: int = 8,
        hidden_size: int = 10,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softsign(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_length, n_features) → (batch, 1) in (-1, +1).

        Takes the LSTM's hidden state at the final time step
        (``out[:, -1, :]``) and maps to position.
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)
