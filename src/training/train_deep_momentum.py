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
                     resume_from: str | None = None,
                     fold: int | None = None) -> dict:
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
            fold=fold,
        )
        # Commit so a preemption doesn't lose the final checkpoint
        try:
            volume.commit()
        except Exception:  # noqa: BLE001
            log.warning("volume.commit() failed (non-fatal at this point)")
        return metrics

    @app.local_entrypoint()
    def main(arch: str = "MLP", max_epochs: int = 200,
             resume_from: str | None = None, walk_forward: bool = False):
        if walk_forward:
            # One container per fold, run in parallel.
            from src.data.futures import WALK_FORWARD_FOLDS
            stems = [_checkpoint_stem(arch, f) for f in WALK_FORWARD_FOLDS]
            results = train_remote.starmap(
                [(arch, max_epochs, None, f) for f in WALK_FORWARD_FOLDS]
            )
            for f, metrics in zip(WALK_FORWARD_FOLDS, results):
                print(f"[train_deep_momentum] arch={arch} fold={f} metrics={metrics}")
        else:
            metrics = train_remote.remote(
                arch=arch, max_epochs=max_epochs, resume_from=resume_from,
            )
            print(f"[train_deep_momentum] arch={arch} metrics={metrics}")
            stems = [_checkpoint_stem(arch, None)]
        print("To pull the checkpoints locally:")
        for stem in stems:
            for ext in ("pt", "json"):
                print(f"  modal volume get dl-research-data /pretrained/{stem}.{ext} "
                      f"./data/pretrained/{stem}.{ext}")

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


def _checkpoint_stem(arch: str, fold: int | None) -> str:
    """`mlp_sharpe` for the fixed-split model, `mlp_sharpe_wf2023` for a fold."""
    stem = f"{arch.lower()}_sharpe"
    return stem if fold is None else f"{stem}_wf{fold}"

# L2 penalty on all weights; with the MLP's dropout this keeps the network
# from memorising the training years.
WEIGHT_DECAY = 1e-4


def _build_per_day_features(panel, min_active_contracts: int = 5) -> tuple:
    """Build per-day feature tensors for the Sharpe-loss training objective.

    Per Lim/Zohren/Roberts 2019 §4.3 + research.md §R5: the Sharpe loss
    must be computed on a per-day **portfolio** return series, not on a
    per-sample cross-section. So each training "sample" is one trading
    day with features for every active contract stacked.

    Features per (day, contract) (FR-008):
        - 5 normalized return horizons: 1d, 1m (21d), 3m (63d), 6m (126d), 1y (252d)
        - 3 MACD timescales: (8/24), (16/48), (32/96)

    Returns
    -------
    X      : float32 array shape (n_days, n_contracts, seq_length=60, n_features=8)
    y      : float32 array shape (n_days, n_contracts) — next-day return per asset
    mask   : bool   array shape (n_days, n_contracts) — True where the sample is valid
                     (no NaN in the 60-day window AND a finite next-day return).
                     Used in the per-day portfolio aggregation to ignore missing assets.
    dates  : array shape (n_days,) — the date of each per-day sample
    contracts : array shape (n_contracts,) — the canonical contract ordering

    Days where fewer than ``min_active_contracts`` assets are valid are
    dropped — they'd otherwise give a noisy portfolio return.
    """
    import numpy as np
    import pandas as pd

    from src.strategies.tsmom import macd_signal

    seq_length = 60
    n_features = 8

    wide = panel.pivot(index="date", columns="contract", values="price").sort_index()
    wide.index = pd.to_datetime(wide.index)
    rets_1d = wide.pct_change(1)

    def _norm(horizon):
        roll_ret = wide.pct_change(horizon)
        vol = roll_ret.ewm(span=63, adjust=False).std()
        return roll_ret / vol

    feat_horizons = [_norm(h) for h in (1, 21, 63, 126, 252)]
    feat_macd = [macd_signal(wide, s, l) for s, l in ((8, 24), (16, 48), (32, 96))]
    feature_list = feat_horizons + feat_macd
    assert len(feature_list) == n_features

    contracts = list(wide.columns)
    n_contracts = len(contracts)
    dates = wide.index.to_numpy()
    n_dates = len(dates)

    # Per-contract feature tensors (n_dates, n_features)
    per_contract_feats = np.stack(
        [np.column_stack([f[c].to_numpy() for f in feature_list]) for c in contracts],
        axis=1,
    )  # shape (n_dates, n_contracts, n_features)
    per_contract_next_ret = np.column_stack(
        [rets_1d[c].shift(-1).to_numpy() for c in contracts]
    )  # shape (n_dates, n_contracts)

    samples_X: list[np.ndarray] = []
    samples_y: list[np.ndarray] = []
    samples_mask: list[np.ndarray] = []
    samples_date: list = []

    for i in range(seq_length, n_dates - 1):
        window = per_contract_feats[i - seq_length: i]  # (seq_length, n_contracts, n_features)
        # Reorder to (n_contracts, seq_length, n_features) for downstream model batching
        window = window.transpose(1, 0, 2)
        next_ret = per_contract_next_ret[i]              # (n_contracts,)

        valid_window = ~np.isnan(window).any(axis=(1, 2))   # (n_contracts,)
        valid_target = ~np.isnan(next_ret)                  # (n_contracts,)
        mask = valid_window & valid_target                   # (n_contracts,)
        if mask.sum() < min_active_contracts:
            continue

        # Zero-fill NaNs so the model can run on the full (n_contracts, ...) batch;
        # the mask makes them invisible to the loss aggregator.
        window_safe = np.where(np.isnan(window), 0.0, window)
        next_ret_safe = np.where(np.isnan(next_ret), 0.0, next_ret)

        samples_X.append(window_safe)
        samples_y.append(next_ret_safe)
        samples_mask.append(mask)
        samples_date.append(dates[i])

    X = np.asarray(samples_X, dtype=np.float32)
    y = np.asarray(samples_y, dtype=np.float32)
    mask = np.asarray(samples_mask, dtype=bool)
    dates_arr = np.asarray(samples_date)
    contracts_arr = np.asarray(contracts)
    return X, y, mask, dates_arr, contracts_arr


class _DailyPortfolioDataset:
    """torch Dataset that returns (X, y, mask) triples per day. Defined here
    rather than in src/torch_data.py to keep the canonical MyDataset signature
    intact (Constitution Principle V — pre-existing canon).
    """

    def __init__(self, X, y, mask):
        import torch
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.mask = torch.from_numpy(mask)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]


def train(
    data_dir: Path,
    arch: Literal["MLP", "LSTM"],
    device,                   # torch.device — typed late to avoid module-load cost
    checkpoint_dir: Path,
    max_epochs: int = 200,
    resume_from: str | None = None,
    modal_image_id: str | None = None,
    fold: int | None = None,
) -> dict[str, float]:
    """Device-agnostic training loop. Importable on CPU; runs unchanged on GPU.

    ``fold`` selects a walk-forward fold (its test-window start year, see
    ``src.data.futures.walk_forward_split``) and writes
    ``{arch}_sharpe_wf{fold}.pt``. Without it the fixed 2020 split is used.
    """
    import numpy as np
    import pandas as pd
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from src.data.futures import (
        TEST_START,
        TRAIN_END,
        TRAIN_START,
        VAL_START,
        walk_forward_split,
    )
    from src.early_stopper import EarlyStopping
    from src.losses import Neg_Sharpe  # canonical per Principle V
    from src.models.deep_momentum import DeepMomentumLSTM, DeepMomentumMLP

    torch.manual_seed(42)
    np.random.seed(42)

    arch = arch.upper()
    if arch not in ("MLP", "LSTM"):
        raise ValueError(f"arch must be 'MLP' or 'LSTM'; got {arch!r}")

    if fold is None:
        val_start, train_end, test_start, test_end = (
            VAL_START, TRAIN_END, TEST_START, None)
    else:
        val_start, train_end, test_start, test_end = walk_forward_split(fold)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = _checkpoint_stem(arch, fold)
    final_path = checkpoint_dir / f"{stem}.pt"
    sidecar_path = checkpoint_dir / f"{stem}.json"

    # ── Data ────────────────────────────────────────────────────────
    parquet_path = data_dir / "cme_futures.parquet"
    log.info("Loading %s", parquet_path)
    panel = pd.read_parquet(parquet_path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    # Features look only backwards, so they are built once over the whole
    # panel (as scripts/run_backtests.py does) and the samples are then split
    # by date. Building them per split instead would throw away the first
    # ~312 days of each split to the 252-day horizon + 60-day window warm-up.
    X, y, m, dates, _contracts = _build_per_day_features(panel)
    dates = pd.to_datetime(dates)
    in_train = dates < pd.Timestamp(val_start)
    in_val = (dates >= pd.Timestamp(val_start)) & (dates <= pd.Timestamp(train_end))
    in_test = dates >= pd.Timestamp(test_start)
    if test_end is not None:
        in_test &= dates < pd.Timestamp(test_end)
    # A sample's target is the NEXT day's return, so the last sample of the
    # train and val segments would be scored on the first day of the segment
    # after it. Drop those two boundary samples.
    for seg in (in_train, in_val):
        idx = np.flatnonzero(seg)
        if len(idx):
            seg[idx[-1]] = False
    if not (in_train.any() and in_val.any() and in_test.any()):
        raise ValueError(
            f"Empty split: train={in_train.sum()} val={in_val.sum()} "
            f"test={in_test.sum()} days. Train is {TRAIN_START}..{val_start} "
            f"(exclusive), val {val_start}..{train_end}, test {test_start}"
            f"..{test_end or 'end'}."
        )
    log.info("Days: train=%d (%s → %s), val=%d (%s → %s), test=%d (%s → %s)",
             in_train.sum(), TRAIN_START, val_start, in_val.sum(), val_start,
             train_end, in_test.sum(), test_start, test_end or "end")

    def _loader(sel, shuffle):
        return DataLoader(_DailyPortfolioDataset(X[sel], y[sel], m[sel]),
                          batch_size=32, shuffle=shuffle)

    # Batch size in days. ~1800 train days / 32 ≈ 57 batches/epoch.
    train_loader = _loader(in_train, shuffle=True)
    val_loader = _loader(in_val, shuffle=False)
    test_loader = _loader(in_test, shuffle=False)

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

    # The canonical EarlyStopping only writes on improvement and never on its
    # first call, so a checkpoint left by an earlier run would otherwise
    # survive and be reloaded below as if it were this run's best.
    final_path.unlink(missing_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3,
                           weight_decay=WEIGHT_DECAY)
    early = EarlyStopping(savepath=str(final_path), patience=25,
                          min_delta=1e-4, verbose=False)

    def _portfolio_returns(xb, yb, mb):
        """Forward + per-day portfolio aggregation.

        xb: (B, C, S, F); yb / mb: (B, C). Returns shape (B,).
        Each day's portfolio = masked mean across contracts of position × next-return.
        """
        b, c, s, f = xb.shape
        flat = xb.reshape(b * c, s, f)
        positions = model(flat).squeeze(-1).reshape(b, c)
        masked = positions * yb * mb.float()
        active = mb.sum(dim=1).clamp(min=1).float()
        return masked.sum(dim=1) / active

    # ── Train loop ──────────────────────────────────────────────────
    epochs_run = 0
    val_loss_history: list[float] = []
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        total_train_loss = 0.0
        n_batches = 0
        for xb, yb, mb in train_loader:
            xb = xb.to(device); yb = yb.to(device); mb = mb.to(device)
            optimizer.zero_grad()
            port = _portfolio_returns(xb, yb, mb)         # (B days,)
            loss = Neg_Sharpe(port)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.detach())
            n_batches += 1
        avg_train_loss = total_train_loss / max(n_batches, 1)

        # Validation on the held-out end of the training window. Neg_Sharpe
        # over the FULL validation series for early-stopping signal
        # stability (per-batch Sharpe is noisy). The test window is never
        # looked at until the final scoring below.
        model.eval()
        with torch.no_grad():
            port_segments = []
            for xb, yb, mb in val_loader:
                xb = xb.to(device); yb = yb.to(device); mb = mb.to(device)
                port_segments.append(_portfolio_returns(xb, yb, mb))
            port_full = torch.cat(port_segments)
            val_neg_sharpe = float(Neg_Sharpe(port_full))
        val_loss_history.append(val_neg_sharpe)

        log.info("epoch %d  train_loss=%.4f  val_neg_sharpe=%.4f",
                 epoch + 1, avg_train_loss, val_neg_sharpe)

        early(model, val_neg_sharpe)
        if epoch == 0:
            # EarlyStopping takes the first epoch as its best without saving.
            torch.save(model.state_dict(), final_path)
        if early.early_stop:
            log.info("Early stopping at epoch %d", epoch + 1)
            break

    # Score the checkpoint that ships, i.e. the best validation epoch, not
    # whatever weights the last epoch left in memory.
    model.load_state_dict(torch.load(final_path, map_location=device))
    best_epoch = int(np.argmin(val_loss_history)) + 1

    # ── Final test metrics on the per-day portfolio return series ───
    model.eval()
    with torch.no_grad():
        port_segments = []
        for xb, yb, mb in test_loader:
            xb = xb.to(device); yb = yb.to(device); mb = mb.to(device)
            port_segments.append(_portfolio_returns(xb, yb, mb).cpu())
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
        "modal_app": "deep-finance-train-momentum",
        "arch": arch,
        "data_range": {
            "train_start": TRAIN_START.isoformat(),
            "val_start": val_start.isoformat(),
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": (dates[in_test].max().date().isoformat()),
        },
        "split": ("chronological_train_val_test" if fold is None
                  else f"walk_forward_fold_{fold}"),
        "hyperparameters": {
            "hidden_size": model.hidden_size,
            "lr": 1e-3,
            "batch_size_days": 32,
            "epochs_trained": epochs_run,
            "best_epoch": best_epoch,
            "weight_decay": WEIGHT_DECAY,
            "dropout": getattr(getattr(model, "dropout", None), "p", 0.0),
            "patience": 25,
            "min_delta": 1e-4,
            "seq_length": 60,
            "n_features": 8,
            "aggregation": "per_day_portfolio_mean",
        },
        "final_metrics": {
            "val_neg_sharpe": min(val_loss_history),
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
    parser.add_argument("--fold", type=int, default=None,
                        help="Walk-forward fold (test-window start year, e.g. "
                             "2023). Default: the fixed 2020 split.")
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
        fold=args.fold,
    )
    print(f"[train_deep_momentum] arch={args.arch} metrics={metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(_cpu_main())
