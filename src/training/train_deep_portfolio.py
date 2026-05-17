"""Modal-hosted trainer for the Phase 2 Deep Portfolio model.

Contract: specs/003-phase-2-portfolio/contracts/train_deep_portfolio_cli.md
Constitution: v1.1.0 §"Training workflow (Modal)".

Two invocation forms (same file, same `train()` body):

    modal run src/training/train_deep_portfolio.py --universe etfs      # T4 GPU
    python -m src.training.train_deep_portfolio --universe etfs --max-epochs 1   # CPU smoke

Mirrors Phase 1's `train_deep_momentum.py` exactly — Modal scaffolding at
top, device-agnostic `train()` body at bottom. Per-day portfolio Sharpe-loss
aggregation per Phase 1 retrospective (commit 93578ca).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ─── Top: Modal scaffolding ──────────────────────────────────────────
try:
    import modal  # type: ignore

    _MODAL_AVAILABLE = True
    app = modal.App("deep-finance-train-portfolio")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_requirements(str(_REPO_ROOT / "requirements-train.txt"))
        .add_local_python_source("src")
    )
    volume = modal.Volume.from_name("dl-research-data", create_if_missing=True)

    @app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=3600)
    def train_remote(universe: str = "etfs", max_epochs: int = 200,
                     resume_from: str | None = None) -> dict:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics = train(
            data_dir=Path("/data"),
            universe=universe,
            device=device,
            checkpoint_dir=Path("/data/pretrained"),
            max_epochs=max_epochs,
            resume_from=resume_from,
            modal_image_id=_modal_image_id_or_local(),
        )
        try:
            volume.commit()
        except Exception:  # noqa: BLE001
            log.warning("volume.commit() failed (non-fatal at this point)")
        return metrics

    @app.local_entrypoint()
    def main(universe: str = "etfs", max_epochs: int = 200,
             resume_from: str | None = None):
        metrics = train_remote.remote(
            universe=universe, max_epochs=max_epochs, resume_from=resume_from,
        )
        print(f"[train_deep_portfolio] universe={universe} metrics={metrics}")
        fname = f"deep_portfolio_{universe}.pt"
        sidecar = f"deep_portfolio_{universe}.json"
        print("To pull the checkpoint locally:")
        print(f"  modal volume get dl-research-data /pretrained/{fname} "
              f"./data/pretrained/{fname}")
        print(f"  modal volume get dl-research-data /pretrained/{sidecar} "
              f"./data/pretrained/{sidecar}")

except ImportError:
    _MODAL_AVAILABLE = False


def _modal_image_id_or_local() -> str:
    """Best-effort Modal image identification — same helper as Phase 1."""
    try:
        import modal  # noqa: F401  type: ignore
        image_id = os.environ.get("MODAL_IMAGE_ID", "")
        return f"Modal T4 (image {image_id})" if image_id else "Modal T4"
    except ImportError:
        return "local CPU smoke run"


# Universe → parquet name mapping (per contract §"Universe → parquet name mapping")
_PARQUET_MAP = {"etfs": "etf_basket", "20stock": "sp500_20"}


# ─── Bottom: device-agnostic training body ────────────────────────────


def _build_per_day_features_portfolio(panel) -> tuple:
    """Build (n_days, lookback=50, 2*N) feature tensors + (n_days, N) targets.

    Per paper §4.3: features per asset per day are (close, daily_return).
    Stacked across assets and time, then flattened by the MLP's Flatten layer.

    Days where any asset has NaN feature in the lookback window are dropped.
    """
    import numpy as np
    import pandas as pd

    lookback = 50

    # Pivot to wide: date × symbol → close
    close_wide = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    close_wide.index = pd.to_datetime(close_wide.index)
    close_wide = close_wide.ffill().dropna(how="all")
    rets = close_wide.pct_change(1)

    assets = list(close_wide.columns)
    n_assets = len(assets)
    dates = close_wide.index.to_numpy()
    n_dates = len(dates)

    # Per-asset (close, return) — shape (n_dates, n_assets, 2)
    close_arr = close_wide.to_numpy(dtype=np.float64)
    ret_arr = rets.to_numpy(dtype=np.float64)
    # Normalise close by its own EWMA to avoid scale-dominance across assets
    close_normed = close_arr / np.where(close_arr > 0,
                                          pd.DataFrame(close_arr).ewm(span=63, adjust=False).mean().to_numpy(),
                                          1.0)
    per_asset = np.stack([close_normed, ret_arr], axis=-1)  # (n_dates, n_assets, 2)

    next_ret = pd.DataFrame(ret_arr).shift(-1).to_numpy()    # (n_dates, n_assets)

    samples_X: list[np.ndarray] = []
    samples_y: list[np.ndarray] = []
    samples_date: list = []

    for i in range(lookback, n_dates - 1):
        window = per_asset[i - lookback: i]              # (lookback, n_assets, 2)
        # Flatten the last two axes → (lookback, 2 * n_assets)
        window = window.reshape(lookback, n_assets * 2)
        target = next_ret[i]                              # (n_assets,)
        if np.isnan(window).any() or np.isnan(target).any():
            continue
        samples_X.append(window)
        samples_y.append(target)
        samples_date.append(dates[i])

    X = np.asarray(samples_X, dtype=np.float32)
    y = np.asarray(samples_y, dtype=np.float32)
    dates_arr = np.asarray(samples_date)
    return X, y, dates_arr, assets


class _PortfolioDataset:
    """torch Dataset returning (X, y) per day. Kept local; MyDataset stays
    canonical (Constitution V).
    """

    def __init__(self, X, y):
        import torch
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train(
    data_dir: Path,
    universe: Literal["etfs", "20stock"],
    device,
    checkpoint_dir: Path,
    max_epochs: int = 200,
    resume_from: str | None = None,
    modal_image_id: str | None = None,
) -> dict[str, float]:
    """Device-agnostic training loop. Importable on CPU; runs unchanged on GPU."""
    import numpy as np
    import pandas as pd
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from src.early_stopper import EarlyStopping
    from src.losses import Neg_Sharpe
    from src.models.deep_portfolio import DeepPortfolioMLP

    torch.manual_seed(42)
    np.random.seed(42)

    if universe not in _PARQUET_MAP:
        raise ValueError(f"universe must be one of {list(_PARQUET_MAP)}; got {universe!r}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / f"deep_portfolio_{universe}.pt"
    sidecar_path = checkpoint_dir / f"deep_portfolio_{universe}.json"

    # ── Data ────────────────────────────────────────────────────────
    parquet_name = _PARQUET_MAP[universe]
    parquet_path = data_dir / f"{parquet_name}.parquet"
    log.info("Loading %s", parquet_path)
    panel = pd.read_parquet(parquet_path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    TEST_START = pd.Timestamp("2020-01-01")
    TRAIN_END = pd.Timestamp("2019-12-31")
    train_panel = panel[panel["date"] <= TRAIN_END]
    test_panel = panel[panel["date"] >= TEST_START]
    log.info("Train rows: %d (… → %s); test rows: %d (%s → ...)",
             len(train_panel), TRAIN_END.date(), len(test_panel), TEST_START.date())

    Xtr, ytr, _dates_tr, assets = _build_per_day_features_portfolio(train_panel)
    Xte, yte, _dates_te, _ = _build_per_day_features_portfolio(test_panel)
    n_assets = len(assets)
    log.info("Train tensors: X=%s y=%s; test: X=%s y=%s; assets=%s",
             Xtr.shape, ytr.shape, Xte.shape, yte.shape, assets)

    train_ds = _PortfolioDataset(Xtr, ytr)
    test_ds = _PortfolioDataset(Xte, yte)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # ── Model ───────────────────────────────────────────────────────
    model = DeepPortfolioMLP(n_assets=n_assets, lookback=50, hidden_size=64).to(device)

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = checkpoint_dir / resume_path
        log.info("Resuming from %s", resume_path)
        model.load_state_dict(torch.load(resume_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    early = EarlyStopping(savepath=str(final_path), patience=25,
                          min_delta=1e-4, verbose=False)

    def _portfolio_returns(xb, yb):
        """xb: (B, T, 2N); yb: (B, N) — returns shape (B,)."""
        weights = model(xb)                       # (B, N) softmax
        return (weights * yb).sum(dim=1)         # (B,)

    # ── Train loop ──────────────────────────────────────────────────
    epochs_run = 0
    val_loss_history: list[float] = []
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        total_train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            port = _portfolio_returns(xb, yb)
            loss = Neg_Sharpe(port)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.detach())
            n_batches += 1
        avg_train_loss = total_train_loss / max(n_batches, 1)

        # Validation (full test window, single Sharpe value for early-stop stability)
        model.eval()
        with torch.no_grad():
            port_segments = []
            for xb, yb in test_loader:
                xb = xb.to(device); yb = yb.to(device)
                port_segments.append(_portfolio_returns(xb, yb))
            port_full = torch.cat(port_segments)
            val_neg_sharpe = float(Neg_Sharpe(port_full))
        val_loss_history.append(val_neg_sharpe)

        log.info("epoch %d  train_loss=%.4f  val_neg_sharpe=%.4f",
                 epoch + 1, avg_train_loss, val_neg_sharpe)

        early(model, val_neg_sharpe)
        if early.early_stop:
            log.info("Early stopping at epoch %d", epoch + 1)
            break

    if not final_path.exists():
        torch.save(model.state_dict(), final_path)

    # ── Final metrics on the per-day portfolio return series ────────
    model.eval()
    with torch.no_grad():
        port_segments = []
        for xb, yb in test_loader:
            xb = xb.to(device); yb = yb.to(device)
            port_segments.append(_portfolio_returns(xb, yb).cpu())
        portfolio_returns_test = torch.cat(port_segments).numpy()
    mu = float(np.mean(portfolio_returns_test))
    sigma = float(np.std(portfolio_returns_test))
    test_annual_sharpe = float(mu / sigma * np.sqrt(252)) if sigma > 0 else 0.0
    cum = np.cumprod(1 + portfolio_returns_test)
    dd = (cum / np.maximum.accumulate(cum)) - 1
    test_mdd = float(-dd.min()) if len(dd) else 0.0
    test_calmar = (mu * 252 / test_mdd) if test_mdd > 0 else 0.0

    # ── Sidecar JSON per data-model §E3 ─────────────────────────────
    import torch as _torch
    sidecar = {
        "trained_on": _dt.date.today().isoformat(),
        "trained_with": modal_image_id or _modal_image_id_or_local(),
        "torch_version": _torch.__version__,
        "modal_app": "deep-finance-train-portfolio",
        "arch": "DeepPortfolioMLP",
        "universe": universe,
        "n_assets": n_assets,
        "data_range": {
            "train_start": str(train_panel["date"].min().date()),
            "train_end": str(TRAIN_END.date()),
            "test_start": str(TEST_START.date()),
            "test_end": str(panel["date"].max().date()),
        },
        "split": "chronological_test_2020",
        "hyperparameters": {
            "hidden_size": model.hidden_size,
            "lr": 1e-3,
            "batch_size_days": 32,
            "epochs_trained": epochs_run,
            "patience": 25,
            "min_delta": 1e-4,
            "lookback": model.lookback,
            "aggregation": "per_day_portfolio_sum",
        },
        "final_metrics": {
            "val_neg_sharpe": val_loss_history[-1] if val_loss_history else None,
            "test_annual_sharpe": test_annual_sharpe,
            "test_max_drawdown": test_mdd,
            "test_calmar": test_calmar,
        },
        "git_commit": os.environ.get("GIT_COMMIT", ""),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    log.info("Wrote sidecar %s", sidecar_path)

    return sidecar["final_metrics"]


# ─── CPU-local CLI ────────────────────────────────────────────────────


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_deep_portfolio",
        description=(
            "Train DeepPortfolioMLP with the Sharpe loss. CPU smoke runs "
            "locally; full training runs on Modal via "
            "`modal run src/training/train_deep_portfolio.py --universe …`."
        ),
    )
    parser.add_argument("--universe", choices=list(_PARQUET_MAP), default="etfs")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Where to read the universe parquet from. Default: data_root().")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=_REPO_ROOT / "data" / "pretrained")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _cpu_main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    import torch
    from src.data import data_root

    data_dir = args.data_dir or data_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = train(
        data_dir=data_dir,
        universe=args.universe,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        max_epochs=args.max_epochs,
        resume_from=args.resume_from,
    )
    print(f"[train_deep_portfolio] universe={args.universe} metrics={metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(_cpu_main())
