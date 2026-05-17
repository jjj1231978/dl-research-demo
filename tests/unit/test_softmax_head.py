"""Softmax invariant test for `src.models.deep_portfolio` (FR-020).

For a random input, the model output must be on the long-only simplex:
sums to 1, all entries in [0, 1].
"""
from __future__ import annotations

import pytest
import torch

from src.models.deep_portfolio import DeepPortfolioMLP


@pytest.mark.parametrize("n_assets", [4, 20])
def test_softmax_output_on_simplex(n_assets):
    torch.manual_seed(0)
    model = DeepPortfolioMLP(n_assets=n_assets)
    model.eval()
    x = torch.randn(8, 50, n_assets * 2)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, n_assets)
    assert torch.isfinite(out).all()
    assert (out >= 0).all() and (out <= 1).all()
    row_sums = out.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(8), atol=1e-6)


def test_softmax_output_on_zeros():
    """Zeros input → equal weights (1/N each) since logits are all 0."""
    model = DeepPortfolioMLP(n_assets=4)
    # Initialize biases to zero so the equal-weight property holds exactly
    for layer in model.net:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 50, 8))
    expected = torch.full((1, 4), 0.25)
    assert torch.allclose(out, expected, atol=1e-6)
