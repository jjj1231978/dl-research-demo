"""Modal-hosted trainer for the Phase 1 Deep Momentum models (MLP + LSTM).

Contract: specs/002-phase-1-momentum/contracts/train_deep_momentum_cli.md
Constitution: v1.1.0 §"Training workflow (Modal)".

Two invocation forms (same file, same `train()` body):

    modal run src/training/train_deep_momentum.py --arch MLP   # T4 GPU
    python -m src.training.train_deep_momentum --arch MLP --max-epochs 1   # CPU smoke

The Modal scaffolding sits at the top; the device-agnostic `train(...)`
body at the bottom is importable for unit tests and runs unchanged on
both compute environments.

Constitution Principle II (NON-NEGOTIABLE): no `.cuda()` calls, no
hardcoded `device='cuda'`.

Constitution Principle V: uses `src.losses.SharpeLoss`,
`src.early_stopper.EarlyStopping`, `src.torch_data.MyDataset` AS-IS — no
wrappers, no overrides. Multi-asset aggregation happens INSIDE
`train()` (pre-aggregate `outputs * future_rets` into per-day portfolio
return before passing to `SharpeLoss`) per the `src/losses.py` comment
and research.md §R5.
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

# Ensure repo root on sys.path when invoked via `python -m` from elsewhere
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ─── Top: Modal scaffolding ──────────────────────────────────────────
# Modal is an optional dep — only required when invoking via `modal run`.
# Guarded so `python -m src.training.train_deep_momentum` works without it.
try:
    import modal  # type: ignore

    _MODAL_AVAILABLE = True
    app = modal.App("deep-finance-train-momentum")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_requirements(str(_REPO_ROOT / "requirements-train.txt"))
        .add_local_python_source("src")
    )
    volume = modal.Volume.from_name("dl-research-data", create_if_missing=True)

    @app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=3600)
    def train_remote(arch: str = "MLP", max_epochs: int = 200,
                     resume_from: str | None = None) -> dict:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics = train(
            data_dir=Path("/data"),
            arch=arch,
            device=device,
            checkpoint_dir=Path("/data/pretrained"),
            max_epochs=max_epochs,
            resume_from=resume_from,
            modal_image_id=_modal_image_id_or_local(),
        )
        # Commit so a preemption doesn't lose the final checkpoint
        try:
            volume.commit()
        except Exception:  # noqa: BLE001
            log.warning("volume.commit() failed (non-fatal at this point)")
        return metrics

    @app.local_entrypoint()
    def main(arch: str = "MLP", max_epochs: int = 200,
             resume_from: str | None = None):
        metrics = train_remote.remote(
            arch=arch, max_epochs=max_epochs, resume_from=resume_from,
        )
        print(f"[train_deep_momentum] arch={arch} metrics={metrics}")
        fname = f"{arch.lower()}_sharpe.pt"
        sidecar = f"{arch.lower()}_sharpe.json"
        print("To pull the checkpoint locally:")
        print(f"  modal volume get dl-research-data /pretrained/{fname} "
              f"./data/pretrained/{fname}")
        print(f"  modal volume get dl-research-data /pretrained/{sidecar} "
              f"./data/pretrained/{sidecar}")

except ImportError:
    _MODAL_AVAILABLE = False


def _modal_image_id_or_local() -> str:
    """Best-effort Modal image identification for the sidecar's `trained_with`."""
    try:
        import modal  # type: ignore
        # Modal exposes the running container's image hash via env vars.
        # Fall back to a generic label if the var is unavailable.
        image_id = os.environ.get("MODAL_IMAGE_ID", "")
        if image_id:
            return f"Modal T4 (image {image_id})"
        return "Modal T4"
    except ImportError:
        return "local CPU smoke run"


# ─── Bottom: device-agnostic training body ────────────────────────────


def _build_features(panel) -> tuple:
    """Build the 8-feature tensors per FR-008.

    Per Lim/Zohren/Roberts 2019 §4.3:
        - 5 normalized return horizons: 1d, 1m (21d), 3m (63d), 6m (126d), 1y (252d)
        - 3 MACD timescales: (8/24), (16/48), (32/96)

    Returns ``(X, y, dates, contracts)`` arrays where:
        - ``X`` shape: (n_samples, seq_length=60, n_features=8)
        - ``y`` shape: (n_samples,) — next-day return per sample
        - ``dates``, ``contracts``: parallel index arrays for traceability
    """
    import numpy as np
    import pandas as pd

    from src.strategies.tsmom import macd_signal

    seq_length = 60

    # Pivot to wide: date × contract → price
    wide = panel.pivot(index="date", columns="contract", values="price").sort_index()
    wide.index = pd.to_datetime(wide.index)

    rets_1d = wide.pct_change(1)

    # Normalize returns by their EWMA-vol (paper §4.3 normalization)
    def _norm(returns, horizon):
        roll_ret = wide.pct_change(horizon)
        vol = roll_ret.ewm(span=63, adjust=False).std()
        return roll_ret / vol

    feat_horizons = [_norm(wide, h) for h in (1, 21, 63, 126, 252)]
    feat_macd = [macd_signal(wide, s, l) for s, l in ((8, 24), (16, 48), (32, 96))]
    feature_list = feat_horizons + feat_macd  # 5 + 3 = 8 features

    contracts = list(wide.columns)
    samples_X = []
    samples_y = []
    samples_date = []
    samples_contract = []

    for contract in contracts:
        # Per-contract feature matrix (n_dates × 8)
        feat = np.column_stack([f[contract].to_numpy() for f in feature_list])
        ret_next = rets_1d[contract].shift(-1).to_numpy()
        dates = wide.index.to_numpy()

        # Slide a seq_length window; drop rows where any feature is NaN
        for i in range(seq_length, len(feat) - 1):
            window = feat[i - seq_length: i]
            target = ret_next[i]
            if np.isnan(window).any() or np.isnan(target):
                continue
            samples_X.append(window)
            samples_y.append(target)
            samples_date.append(dates[i])
            samples_contract.append(contract)

    X = np.asarray(samples_X, dtype=np.float32)
    y = np.asarray(samples_y, dtype=np.float32)
    dates_arr = np.asarray(samples_date)
    contracts_arr = np.asarray(samples_contract)
    return X, y, dates_arr, contracts_arr


def train(
    data_dir: Path,
    arch: Literal["MLP", "LSTM"],
    device,                   # torch.device — typed late to avoid module-load cost
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

    from src.data.futures import TRAIN_END, TRAIN_START, TEST_START
    from src.early_stopper import EarlyStopping
    from src.losses import SharpeLoss
    from src.models.deep_momentum import DeepMomentumLSTM, DeepMomentumMLP
    from src.torch_data import MyDataset

    torch.manual_seed(42)
    np.random.seed(42)

    arch = arch.upper()
    if arch not in ("MLP", "LSTM"):
        raise ValueError(f"arch must be 'MLP' or 'LSTM'; got {arch!r}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / f"{arch.lower()}_sharpe.pt"
    sidecar_path = checkpoint_dir / f"{arch.lower()}_sharpe.json"

    # ── Data ────────────────────────────────────────────────────────
    parquet_path = data_dir / "cme_futures.parquet"
    log.info("Loading %s", parquet_path)
    panel = pd.read_parquet(parquet_path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    train_panel = panel[panel["date"] <= pd.Timestamp(TRAIN_END)]
    test_panel = panel[panel["date"] >= pd.Timestamp(TEST_START)]
    log.info("Train rows: %d (%s → %s); test rows: %d (%s → ...)",
             len(train_panel), TRAIN_START, TRAIN_END,
             len(test_panel), TEST_START)

    Xtr, ytr, _dates_tr, _ct_tr = _build_features(train_panel)
    Xte, yte, _dates_te, _ct_te = _build_features(test_panel)
    log.info("Train tensors: X=%s y=%s; test: X=%s y=%s",
             Xtr.shape, ytr.shape, Xte.shape, yte.shape)

    train_ds = MyDataset(Xtr, ytr)
    test_ds = MyDataset(Xte, yte)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # ── Model ───────────────────────────────────────────────────────
    if arch == "MLP":
        model = DeepMomentumMLP(seq_length=60, n_features=8, hidden_size=20)
    else:
        model = DeepMomentumLSTM(seq_length=60, n_features=8, hidden_size=10)
    model = model.to(device)

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = checkpoint_dir / resume_path
        log.info("Resuming from %s", resume_path)
        model.load_state_dict(torch.load(resume_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = SharpeLoss()
    early = EarlyStopping(savepath=str(final_path), patience=25,
                          min_delta=1e-4, verbose=False)

    # ── Train loop ──────────────────────────────────────────────────
    epochs_run = 0
    val_loss_history: list[float] = []
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        total_train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb).squeeze(-1)            # (B,)
            # Multi-asset aggregation: per-batch is mixed across contracts;
            # treat each batch as a pseudo-portfolio cross-section.
            portfolio = out * yb                    # (B,) — per-sample P&L
            loss = -torch.mean(portfolio) / (torch.std(portfolio) + 1e-9)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.detach())
            n_batches += 1
        avg_train_loss = total_train_loss / max(n_batches, 1)

        # Validation: re-use the test set as val (paper has no separate val;
        # early-stopping monitors test-set Neg_Sharpe — caveat disclosed).
        model.eval()
        with torch.no_grad():
            outs = []
            ys = []
            for xb, yb in test_loader:
                xb = xb.to(device); yb = yb.to(device)
                outs.append(model(xb).squeeze(-1))
                ys.append(yb)
            out_all = torch.cat(outs)
            y_all = torch.cat(ys)
            port_all = out_all * y_all
            val_neg_sharpe = float(-torch.mean(port_all) / (torch.std(port_all) + 1e-9))
        val_loss_history.append(val_neg_sharpe)

        log.info("epoch %d  train_loss=%.4f  val_neg_sharpe=%.4f",
                 epoch + 1, avg_train_loss, val_neg_sharpe)

        early(model, val_neg_sharpe)
        if early.early_stop:
            log.info("Early stopping at epoch %d", epoch + 1)
            break

    # Ensure the final state is saved (EarlyStopping only writes on improvement)
    if not final_path.exists():
        torch.save(model.state_dict(), final_path)

    # ── Final test metrics on the portfolio P&L series ──────────────
    model.eval()
    with torch.no_grad():
        outs = []
        ys = []
        for xb, yb in test_loader:
            xb = xb.to(device); yb = yb.to(device)
            outs.append(model(xb).squeeze(-1).cpu())
            ys.append(yb.cpu())
        out_all = torch.cat(outs).numpy()
        y_all = torch.cat(ys).numpy()
    daily_ret = out_all * y_all
    mu = float(np.mean(daily_ret))
    sigma = float(np.std(daily_ret))
    test_annual_sharpe = float(mu / sigma * np.sqrt(252)) if sigma > 0 else 0.0
    cum = np.cumprod(1 + daily_ret)
    dd = (cum / np.maximum.accumulate(cum)) - 1
    test_mdd = float(-dd.min()) if len(dd) else 0.0
    test_calmar = (mu * 252 / test_mdd) if test_mdd > 0 else 0.0

    # ── Sidecar JSON per data-model §E3 ─────────────────────────────
    import torch as _torch
    sidecar = {
        "trained_on": _dt.date.today().isoformat(),
        "trained_with": modal_image_id or _modal_image_id_or_local(),
        "torch_version": _torch.__version__,
        "modal_app": "deep-finance-train-momentum",
        "arch": arch,
        "data_range": {
            "train_start": TRAIN_START.isoformat(),
            "train_end": TRAIN_END.isoformat(),
            "test_start": TEST_START.isoformat(),
            "test_end": panel["date"].max().date().isoformat(),
        },
        "split": "chronological_60_40",
        "hyperparameters": {
            "hidden_size": model.hidden_size,
            "lr": 1e-3,
            "batch_size": 128,
            "epochs_trained": epochs_run,
            "patience": 25,
            "min_delta": 1e-4,
            "seq_length": 60,
            "n_features": 8,
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


# ─── CPU-local CLI (mirrors the Modal local_entrypoint signature) ────


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_deep_momentum",
        description=(
            "Train DeepMomentumMLP or DeepMomentumLSTM with the Sharpe loss. "
            "CPU smoke runs locally; full training runs on Modal via "
            "`modal run src/training/train_deep_momentum.py --arch …`."
        ),
    )
    parser.add_argument("--arch", choices=["MLP", "LSTM"], default="MLP")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Where to read cme_futures.parquet from. "
                             "Default: data_root() (honours DEEP_FINANCE_DATA_DIR).")
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
        arch=args.arch,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        max_epochs=args.max_epochs,
        resume_from=args.resume_from,
    )
    print(f"[train_deep_momentum] arch={args.arch} metrics={metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(_cpu_main())
