"""Forward-pass smoke tests for the 5 Phase 3 LOB models (FR-018).

Each model:
- accepts the architecture-appropriate input shape per `contracts/lob_models.md`
- returns a (B, 3) softmax tensor that is finite, on [0, 1], and sums to 1.
"""
from __future__ import annotations

import pytest
import torch

from src.models.deeplob import (
    DeepLOB,
    LOBCNN_I,
    LOBCNN_II,
    LOBLSTM,
    LOBSimpleMLP,
)

_BATCH = 2
_LOOKBACK = 100
_FEATS = 40


def _conv2d_input() -> torch.Tensor:
    return torch.randn(_BATCH, 1, _LOOKBACK, _FEATS)


def _seq_input() -> torch.Tensor:
    return torch.randn(_BATCH, _LOOKBACK, _FEATS)


def _flat_input() -> torch.Tensor:
    return torch.randn(_BATCH, _LOOKBACK * _FEATS)


def _assert_softmax_3way(out: torch.Tensor) -> None:
    assert out.shape == (_BATCH, 3), f"expected ({_BATCH}, 3), got {out.shape}"
    assert torch.isfinite(out).all(), "output contains NaN or inf"
    assert ((out >= 0) & (out <= 1)).all(), "output not in [0, 1]"
    row_sums = out.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(_BATCH), atol=1e-6), (
        f"row sums {row_sums.tolist()} != 1"
    )


_MODELS = [
    pytest.param(DeepLOB, _conv2d_input, "DeepLOB", id="DeepLOB"),
    pytest.param(LOBCNN_I, _conv2d_input, "CNN1", id="LOBCNN_I"),
    pytest.param(LOBCNN_II, _conv2d_input, "CNN2", id="LOBCNN_II"),
    pytest.param(LOBLSTM, _seq_input, "LSTM", id="LOBLSTM"),
    pytest.param(LOBSimpleMLP, _flat_input, "MLP", id="LOBSimpleMLP"),
]


@pytest.mark.parametrize("model_cls,input_fn,arch_name", _MODELS)
def test_forward_pass_softmax(model_cls, input_fn, arch_name):
    torch.manual_seed(0)
    model = model_cls()
    model.eval()
    assert model.arch_name == arch_name, (
        f"{model_cls.__name__}.arch_name={model.arch_name!r}, "
        f"expected {arch_name!r}"
    )
    with torch.no_grad():
        out = model(input_fn())
    _assert_softmax_3way(out)


@pytest.mark.parametrize("model_cls,input_fn,arch_name", _MODELS)
def test_forward_zeros_does_not_explode(model_cls, input_fn, arch_name):
    """Zero input must not produce NaN — sanity check on init."""
    model = model_cls()
    model.eval()
    x = torch.zeros_like(input_fn())
    with torch.no_grad():
        out = model(x)
    _assert_softmax_3way(out)
