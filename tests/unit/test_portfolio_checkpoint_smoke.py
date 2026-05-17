"""Checkpoint smoke tests for the Phase 2 Deep Portfolio models (FR-021).

Skipped gracefully (pytest.skip) when the corresponding `.pt` is absent so
the suite stays green between US1 ship and US2 ship.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.deep_portfolio import DeepPortfolioMLP

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PRETRAINED_DIR = _REPO_ROOT / "data" / "pretrained"

_UNIVERSE_TO_NASSETS = {"etfs": 4, "20stock": 20}


def _smoke_forward(model: torch.nn.Module, n_assets: int) -> None:
    model.eval()
    x = torch.zeros(1, 50, n_assets * 2)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, n_assets)
    assert torch.isfinite(out).all()
    assert (out >= 0).all() and (out <= 1).all()
    assert abs(out.sum().item() - 1.0) < 1e-6


@pytest.mark.parametrize("universe,n_assets", list(_UNIVERSE_TO_NASSETS.items()))
def test_checkpoint_loads_and_forwards(universe, n_assets):
    ckpt = _PRETRAINED_DIR / f"deep_portfolio_{universe}.pt"
    if not ckpt.exists():
        pytest.skip(
            f"{ckpt} not committed yet — run Phase 2 US2 (Modal training) first."
        )
    model = DeepPortfolioMLP(n_assets=n_assets)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    _smoke_forward(model, n_assets)


# Untrained-model smoke also runs — proves the module shape is correct
# independent of US2 checkpoint availability.


@pytest.mark.parametrize("n_assets", [4, 20])
def test_untrained_forward_passes(n_assets):
    model = DeepPortfolioMLP(n_assets=n_assets)
    _smoke_forward(model, n_assets)
