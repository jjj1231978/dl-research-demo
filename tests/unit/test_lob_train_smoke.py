"""End-to-end CPU smoke test for `src.training.train_deeplob` (FR-021).

Builds a tiny synthetic FI-2010-shaped parquet, calls `train(...)` with
max_epochs=1 on CPU for the cheapest arch (MLP), asserts the return-value
contract and that the `.pt` + `.json` files are written.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.train_deeplob import train


def _make_tiny_fi2010_parquet(path: Path) -> None:
    """A parquet matching the schema from `fetch_lob_fi2010.py`.

    250 ticks train + 250 ticks test, enough for one ~150-window batch on
    a lookback-100 sliding window setup. Synthetic 40-feature noise + random
    labels in {0, 1, 2}.
    """
    rng = np.random.default_rng(42)
    rows = []
    for split, day, n_ticks in (("train", 0, 250), ("test", 8, 250)):
        for tick in range(n_ticks):
            row = {"split": split, "day": day, "tick": tick}
            for i in range(40):
                row[f"f{i:02d}"] = float(rng.standard_normal())
            for k in (10, 20, 30, 50, 100):
                row[f"label_k{k}"] = int(rng.integers(0, 3))
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def test_train_runs_on_cpu_one_epoch_mlp(tmp_path: Path):
    """Cheapest arch (MLP) on CPU, 1 epoch — assert return shape + outputs exist."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ckpt_dir = tmp_path / "pretrained"
    ckpt_dir.mkdir()
    _make_tiny_fi2010_parquet(data_dir / "lob_fi2010.parquet")

    metrics = train(
        data_dir=data_dir,
        arch="MLP",
        device=torch.device("cpu"),
        checkpoint_dir=ckpt_dir,
        max_epochs=1,
    )

    assert isinstance(metrics, dict)
    for k in ("test_accuracy", "test_precision_macro",
              "test_recall_macro", "test_f1_macro"):
        assert k in metrics, f"missing {k!r} in return dict"

    ckpt = ckpt_dir / "mlp_fi2010_k10.pt"
    assert ckpt.exists(), f"no checkpoint written to {ckpt}"
    state = torch.load(ckpt, map_location="cpu")
    assert isinstance(state, dict) and state

    sidecar = ckpt_dir / "mlp_fi2010_k10.json"
    assert sidecar.exists(), f"no sidecar written to {sidecar}"
