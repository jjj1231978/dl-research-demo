"""End-to-end CPU smoke test for `src.training.train_deep_portfolio` (FR-021).

Builds a tiny synthetic parquet, calls `train(...)` with max_epochs=1 on
CPU, asserts the return-value contract and that the `.pt` + `.json` files
are written.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.train_deep_portfolio import train


def _make_tiny_etf_parquet(path: Path) -> None:
    """5-asset parquet, single continuous window covering both train (≤2019)
    and test (≥2020) so the trainer's split has data on each side.
    """
    rng = np.random.default_rng(42)
    rows = []
    syms = ["VTI", "AGG", "DBC", "VIXY", "XEXTRA"]  # the trainer derives N from columns
    dates = pd.date_range("2018-01-01", "2021-12-31", freq="B")
    for sym in syms:
        prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
        for d, p in zip(dates, prices):
            rows.append({
                "date": d.date(), "symbol": sym,
                "open": float(p), "high": float(p), "low": float(p),
                "close": float(p), "volume": 1_000_000,
            })
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow")


def test_train_runs_on_cpu_one_epoch(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ckpt_dir = tmp_path / "pretrained"
    ckpt_dir.mkdir()
    # The trainer reads etf_basket.parquet for universe=etfs
    _make_tiny_etf_parquet(data_dir / "etf_basket.parquet")

    metrics = train(
        data_dir=data_dir,
        universe="etfs",
        device=torch.device("cpu"),
        checkpoint_dir=ckpt_dir,
        max_epochs=1,
    )

    assert isinstance(metrics, dict)
    for k in ("test_annual_sharpe", "test_max_drawdown", "test_calmar"):
        assert k in metrics, f"missing {k!r} in return"

    ckpt = ckpt_dir / "deep_portfolio_etfs.pt"
    assert ckpt.exists(), f"no checkpoint written to {ckpt}"
    state = torch.load(ckpt, map_location="cpu")
    assert isinstance(state, dict) and state

    sidecar = ckpt_dir / "deep_portfolio_etfs.json"
    assert sidecar.exists(), f"no sidecar written to {sidecar}"
