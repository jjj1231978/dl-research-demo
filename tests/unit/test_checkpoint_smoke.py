"""Checkpoint smoke tests for the Phase 1 deep-momentum models (FR-021).

Per `contracts/deep_momentum_models.md` §"Forward-pass smoke test": load
each committed `.pt` on CPU, run a zeros forward pass, and assert the
output shape + finiteness + Softsign bounds.

Tests are *skipped* (not failed) when the corresponding `.pt` file does
not exist. This keeps the suite green between US1 (page MVP — no
checkpoints) and US2 (Modal training — checkpoints land in data/pretrained/).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.deep_momentum import DeepMomentumLSTM, DeepMomentumMLP

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PRETRAINED_DIR = _REPO_ROOT / "data" / "pretrained"


def _smoke_forward(model: torch.nn.Module) -> None:
    model.eval()
    x = torch.zeros(1, 60, 8)  # CPU; no device kwarg
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1), f"unexpected output shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), f"non-finite output {out}"
    val = float(out.item())
    assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6, (
        f"Softsign output {val} outside expected (-1, 1) range"
    )


def test_mlp_checkpoint_loads_and_forwards():
    ckpt = _PRETRAINED_DIR / "mlp_sharpe.pt"
    if not ckpt.exists():
        pytest.skip(
            f"{ckpt} not committed yet — run Phase 1 US2 (Modal training) first."
        )
    model = DeepMomentumMLP(seq_length=60, n_features=8, hidden_size=20)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    _smoke_forward(model)


def test_lstm_checkpoint_loads_and_forwards():
    ckpt = _PRETRAINED_DIR / "lstm_sharpe.pt"
    if not ckpt.exists():
        pytest.skip(
            f"{ckpt} not committed yet — run Phase 1 US2 (Modal training) first."
        )
    model = DeepMomentumLSTM(seq_length=60, n_features=8, hidden_size=10)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    _smoke_forward(model)


# Smoke-tests for the *untrained* models also pass — proves the module
# layout itself is correct independent of US2 checkpoint availability.


def test_untrained_mlp_forward_passes():
    model = DeepMomentumMLP()
    _smoke_forward(model)


def test_untrained_lstm_forward_passes():
    model = DeepMomentumLSTM()
    _smoke_forward(model)
