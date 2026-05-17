"""End-to-end CPU smoke test for the Modal training body
(Constitution Principle II — same body runs on CPU and GPU).

Builds a tiny `cme_futures.parquet` (5 contracts × ~120 days each) in a
tmp dir and calls `src.training.train_deep_momentum.train(...)` with
arch="MLP", max_epochs=1, device=cpu. Asserts:

- the function returns a dict with the expected `final_metrics` keys
- the `.pt` checkpoint was written and is loadable on CPU
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.train_deep_momentum import train


def _make_tiny_cme_parquet(path: Path, n_contracts: int = 5) -> None:
    """Tiny long-format CME parquet covering both train (≤2019-12-31) and
    test (≥2020-01-01) windows so train() sees non-empty splits.

    Each window needs ≥ 252 (longest return horizon) + 60 (seq_length)
    rows per contract for `_build_features` to emit any samples.
    """
    rng = np.random.default_rng(42)
    rows = []
    contracts = ["CL", "ZC", "GC", "HG", "NG"][:n_contracts]
    train_dates = pd.date_range("2018-01-01", periods=480, freq="B")
    test_dates = pd.date_range("2020-04-01", periods=480, freq="B")
    for ct in contracts:
        for dates in (train_dates, test_dates):
            rets = rng.normal(0, 0.012, len(dates))
            prices = 50.0 * np.cumprod(1 + rets)
            for d, p, r in zip(dates, prices, rets):
                rows.append({
                    "date": d.date(),
                    "contract": ct,
                    "asset_class": "commodity",
                    "price": float(p),
                    "return": float(r),
                })
    df = pd.DataFrame(rows)
    df.to_parquet(path, engine="pyarrow", index=False)


def test_train_runs_on_cpu_one_epoch(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ckpt_dir = tmp_path / "pretrained"
    ckpt_dir.mkdir()
    _make_tiny_cme_parquet(data_dir / "cme_futures.parquet")

    device = torch.device("cpu")
    metrics = train(
        data_dir=data_dir,
        arch="MLP",
        device=device,
        checkpoint_dir=ckpt_dir,
        max_epochs=1,
    )

    # Return-value contract
    assert isinstance(metrics, dict)
    for k in ("test_annual_sharpe", "test_max_drawdown", "test_calmar"):
        assert k in metrics, f"missing key {k!r} in train() return"
        assert np.isfinite(metrics[k]) or metrics[k] == 0, (
            f"non-finite metric {k}={metrics[k]}"
        )

    # Checkpoint side-effect contract
    ckpt = ckpt_dir / "mlp_sharpe.pt"
    assert ckpt.exists(), f"no checkpoint written to {ckpt}"
    state = torch.load(ckpt, map_location="cpu")
    assert isinstance(state, dict) and state, "loaded state_dict is empty"

    sidecar = ckpt_dir / "mlp_sharpe.json"
    assert sidecar.exists(), f"no sidecar written to {sidecar}"
