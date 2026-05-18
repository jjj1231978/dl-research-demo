"""DeepLOB + 4 baseline architectures for FI-2010 mid-price prediction.

Per Zhang, Zohren, Roberts (2019) §III + Tsantekidis et al. 2017
(baselines) + `notebooks/reference/02_predictive_signal_lob.ipynb`.

All five models:
- output a 3-way softmax over (down, stationary, up)
- are device-agnostic (Constitution Principle II — no .cuda() calls)
- expose .arch_name matching the sidecar JSON's `arch` field

Forward input shapes per `contracts/lob_models.md`:
- DeepLOB / CNN-I / CNN-II : (B, 1, T=100, 40)
- LSTM                      : (B, T=100, 40)
- SimpleMLP                 : (B, T*40 = 4000)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DeepLOB(nn.Module):
    """CNN + Inception + LSTM per paper §III. ~140k params."""

    arch_name = "DeepLOB"

    def __init__(self, n_classes: int = 3, lookback: int = 100):
        super().__init__()
        self.lookback = lookback
        self.n_classes = n_classes

        # --- Conv Block 1: volume-imbalance / micro-price features ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2)),  # bid/ask pair
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(),
        )

        # --- Conv Block 2: next-level features ---
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(),
        )

        # --- Conv Block 3: whole-book features ---
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 10)),  # collapse spatial dim
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(),
        )

        # --- Inception module ---
        self.inc_b1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(),
        )
        self.inc_b2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(),
        )
        self.inc_b3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(16, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(),
        )

        # --- LSTM + head ---
        self.lstm = nn.LSTM(input_size=96, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, 40)
        x = self.conv1(x)       # (B, 16, T', 20)
        x = self.conv2(x)       # (B, 16, T', 10)
        x = self.conv3(x)       # (B, 16, T', 1)
        b1 = self.inc_b1(x)     # (B, 32, T', 1)
        b2 = self.inc_b2(x)     # (B, 32, T', 1)
        b3 = self.inc_b3(x)     # (B, 32, T', 1)
        x = torch.cat([b1, b2, b3], dim=1)  # (B, 96, T', 1)
        # Reshape for LSTM: (B, seq_len, features)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, T', 96)
        x, _ = self.lstm(x)
        x = x[:, -1, :]          # last time step → (B, 64)
        logits = self.fc(x)
        return torch.softmax(logits, dim=-1)


class LOBSimpleMLP(nn.Module):
    """Flat MLP baseline (no convolutions). Input: flattened LOB window."""

    arch_name = "MLP"

    def __init__(self, n_classes: int = 3, lookback: int = 100, hidden: int = 64):
        super().__init__()
        self.lookback = lookback
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(lookback * 40, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class LOBCNN_I(nn.Module):
    """Tsantekidis 2017 CNN-I baseline. Conv1d on flattened sequence."""

    arch_name = "CNN1"

    def __init__(self, n_classes: int = 3, lookback: int = 100):
        super().__init__()
        self.lookback = lookback
        self.n_classes = n_classes
        # Treat (T, 40) as (channels=1, height=T, width=40) — same 2D conv as DeepLOB
        # but with simpler architecture: just the first two conv blocks
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class LOBCNN_II(nn.Module):
    """Tsantekidis 2017 CNN-II — adds the third conv block (whole-book)."""

    arch_name = "CNN2"

    def __init__(self, n_classes: int = 3, lookback: int = 100):
        super().__init__()
        self.lookback = lookback
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(1, 10)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class LOBLSTM(nn.Module):
    """LSTM baseline. Input: (B, T, 40) — features per time step."""

    arch_name = "LSTM"

    def __init__(
        self,
        n_classes: int = 3,
        lookback: int = 100,
        hidden_size: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        self.lookback = lookback
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=40, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return torch.softmax(self.fc(last), dim=-1)
