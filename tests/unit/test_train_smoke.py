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
    """Tiny long-format CME parquet covering the train (<2018-01-01),
    validation (2018-2019) and test (≥2020-01-01) windows so train() sees
    three non-empty splits.

    The first 252 (longest return horizon) + 60 (seq_length) rows per
    contract are warm-up and emit no samples, so the train window starts
    early enough to leave samples before 2018.
    """
    rng = np.random.default_rng(42)
    rows = []
    contracts = ["CL", "ZC", "GC", "HG", "NG"][:n_contracts]
    train_dates = pd.date_range("2016-01-01", periods=1000, freq="B")
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


def test_reported_test_sharpe_is_from_saved_checkpoint(tmp_path: Path):
    """The sidecar's test Sharpe must describe the weights that ship.

    Regression: the trainer used to score the last epoch's in-memory weights
    while the checkpoint held the best validation epoch, so the sidecar and
    the backtest of that checkpoint disagreed. A stale checkpoint from an
    earlier run must not survive either.
    """
    from src.data.futures import TEST_START
    from src.models.deep_momentum import DeepMomentumMLP
    from src.training.train_deep_momentum import _build_per_day_features

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ckpt_dir = tmp_path / "pretrained"
    ckpt_dir.mkdir()
    _make_tiny_cme_parquet(data_dir / "cme_futures.parquet")
    stale = DeepMomentumMLP()
    torch.nn.init.zeros_(stale.net[1].weight)
    torch.save(stale.state_dict(), ckpt_dir / "mlp_sharpe.pt")

    metrics = train(
        data_dir=data_dir,
        arch="MLP",
        device=torch.device("cpu"),
        checkpoint_dir=ckpt_dir,
        max_epochs=4,
    )

    model = DeepMomentumMLP()
    model.load_state_dict(torch.load(ckpt_dir / "mlp_sharpe.pt", map_location="cpu"))
    model.eval()
    assert model.net[1].weight.abs().sum() > 0, "stale checkpoint was not replaced"

    panel = pd.read_parquet(data_dir / "cme_futures.parquet")
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    X, y, m, dates, _ = _build_per_day_features(panel)
    sel = pd.to_datetime(dates) >= pd.Timestamp(TEST_START)
    X, y, m = (torch.from_numpy(a[sel]) for a in (X, y, m))
    b, c, s, f = X.shape
    with torch.no_grad():
        pos = model(X.reshape(b * c, s, f)).reshape(b, c)
    port = ((pos * y * m.float()).sum(dim=1) / m.sum(dim=1).clamp(min=1)).numpy()
    expected = port.mean() / port.std() * np.sqrt(252)

    assert metrics["test_annual_sharpe"] == pytest.approx(expected, rel=1e-4)


def test_walk_forward_split_schedule():
    """Folds tile the test period with no gaps, and each fold's validation
    slice ends the day before its test window opens."""
    import datetime as dt

    from src.data.futures import WALK_FORWARD_FOLDS, walk_forward_split

    splits = [walk_forward_split(f) for f in WALK_FORWARD_FOLDS]
    for (val_start, train_end, test_start, _), fold in zip(splits, WALK_FORWARD_FOLDS):
        assert val_start < train_end < test_start == dt.date(fold, 1, 1)
        assert test_start - train_end == dt.timedelta(days=1)
    for (_, _, _, end), (_, _, next_start, _) in zip(splits, splits[1:]):
        assert end == next_start
    assert splits[-1][3] is None  # the last fold runs to the end of the data
    with pytest.raises(ValueError):
        walk_forward_split(2021)


def test_train_fold_writes_fold_checkpoint(tmp_path: Path):
    import json

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ckpt_dir = tmp_path / "pretrained"
    _make_tiny_cme_parquet(data_dir / "cme_futures.parquet")

    train(data_dir=data_dir, arch="MLP", device=torch.device("cpu"),
          checkpoint_dir=ckpt_dir, max_epochs=1, fold=2020)

    assert (ckpt_dir / "mlp_sharpe_wf2020.pt").exists()
    assert not (ckpt_dir / "mlp_sharpe.pt").exists()
    sidecar = json.loads((ckpt_dir / "mlp_sharpe_wf2020.json").read_text())
    assert sidecar["split"] == "walk_forward_fold_2020"
    assert sidecar["data_range"]["val_start"] == "2018-01-01"
    assert sidecar["data_range"]["test_end"] < "2023-01-01"
