"""Checkpoint smoke tests for the Phase 3 LOB models (FR-020).

Skips gracefully (pytest.skip) when the corresponding `.pt` is absent so
the suite stays green between US1 ship and US2 ship.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.models.deeplob import (
    DeepLOB,
    LOBCNN_I,
    LOBCNN_II,
    LOBLSTM,
    LOBSimpleMLP,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PRETRAINED_DIR = _REPO_ROOT / "data" / "pretrained"

_ARCH_TO_CLS = {
    "deeplob": DeepLOB,
    "mlp": LOBSimpleMLP,
    "cnn1": LOBCNN_I,
    "cnn2": LOBCNN_II,
    "lstm": LOBLSTM,
}

_ARCH_TO_INPUT_SHAPE = {
    "deeplob": (1, 1, 100, 40),
    "cnn1":    (1, 1, 100, 40),
    "cnn2":    (1, 1, 100, 40),
    "lstm":    (1, 100, 40),
    "mlp":     (1, 4000),
}


@pytest.mark.parametrize("arch_lower", list(_ARCH_TO_CLS))
def test_checkpoint_loads_and_forwards(arch_lower):
    ckpt = _PRETRAINED_DIR / f"{arch_lower}_fi2010_k10.pt"
    if not ckpt.exists():
        pytest.skip(
            f"{ckpt} not committed yet — run Phase 3 US2 "
            f"(`modal run src/training/train_deeplob.py --arch {arch_lower.upper()}`)."
        )
    model = _ARCH_TO_CLS[arch_lower]()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    x = torch.zeros(*_ARCH_TO_INPUT_SHAPE[arch_lower])
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3)
    assert torch.isfinite(out).all()
    row_sum = float(out.sum())
    assert abs(row_sum - 1.0) < 1e-6, f"row sum {row_sum} != 1"


@pytest.mark.parametrize("arch_lower", list(_ARCH_TO_CLS))
def test_sidecar_schema(arch_lower):
    sidecar = _PRETRAINED_DIR / f"{arch_lower}_fi2010_k10.json"
    if not sidecar.exists():
        pytest.skip(f"{sidecar} not committed yet")
    meta = json.loads(sidecar.read_text())
    assert meta["arch"] in ("DeepLOB", "MLP", "CNN1", "CNN2", "LSTM")
    assert meta["arch"].lower() == arch_lower
    assert meta["setup"] == 2
    assert meta["k"] == 10
    assert meta["trained_with"].startswith("Modal "), (
        f"trained_with={meta['trained_with']!r} must start with 'Modal ' "
        f"per Constitution Principle III"
    )
