"""Modal-hosted trainer for the Phase 3 LOB models.

Contract: specs/004-phase-3-deeplob/contracts/train_deeplob_cli.md
Constitution: v1.1.0 §"Training workflow (Modal)".

Mirrors Phase 1/2 — Modal scaffolding at top, device-agnostic `train()`
body at bottom. Per-arch dispatch via `--arch`.

    modal run src/training/train_deeplob.py --arch DeepLOB
    modal run src/training/train_deeplob.py --arch MLP
    modal run src/training/train_deeplob.py --arch CNN1
    modal run src/training/train_deeplob.py --arch CNN2
    modal run src/training/train_deeplob.py --arch LSTM

    python -m src.training.train_deeplob --arch DeepLOB --max-epochs 1
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

_LOOKBACK = 100
_N_CLASSES = 3
_FEATURE_COLS = [f"f{i:02d}" for i in range(40)]
_LABEL_COL = "label_k10"

# ─── Top: Modal scaffolding ──────────────────────────────────────────
try:
    import modal  # type: ignore

    _MODAL_AVAILABLE = True
    app = modal.App("deep-finance-train-deeplob")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_requirements(str(_REPO_ROOT / "requirements-train.txt"))
        .pip_install("scikit-learn")  # for metrics
        .add_local_python_source("src")
    )
    volume = modal.Volume.from_name("dl-research-data", create_if_missing=True)

    @app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=3600)
    def train_remote(arch: str = "DeepLOB", max_epochs: int = 100,
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
        try:
            volume.commit()
        except Exception:  # noqa: BLE001
            log.warning("volume.commit() failed (non-fatal)")
        return metrics

    @app.local_entrypoint()
    def main(arch: str = "DeepLOB", max_epochs: int = 100,
             resume_from: str | None = None):
        metrics = train_remote.remote(arch=arch, max_epochs=max_epochs,
                                       resume_from=resume_from)
        print(f"[train_deeplob] arch={arch} metrics={metrics}")
        fname = f"{arch.lower()}_fi2010_k10.pt"
        sidecar = f"{arch.lower()}_fi2010_k10.json"
        print("To pull the checkpoint locally:")
        print(f"  modal volume get dl-research-data /pretrained/{fname} ./data/pretrained/{fname}")
        print(f"  modal volume get dl-research-data /pretrained/{sidecar} ./data/pretrained/{sidecar}")

except ImportError:
    _MODAL_AVAILABLE = False


def _modal_image_id_or_local() -> str:
    try:
        import modal  # noqa: F401  type: ignore
        image_id = os.environ.get("MODAL_IMAGE_ID", "")
        return f"Modal T4 (image {image_id})" if image_id else "Modal T4"
    except ImportError:
        return "local CPU smoke run"


def _shape_input(arch: str, x_seq):
    """Reshape a (B, lookback, 40) tensor to the input shape each arch expects."""
    if arch in ("DeepLOB", "CNN1", "CNN2"):
        return x_seq.unsqueeze(1)         # (B, 1, T, 40)
    if arch == "LSTM":
        return x_seq                       # (B, T, 40) — LSTM consumes as-is
    if arch == "MLP":
        return x_seq.reshape(x_seq.shape[0], -1)  # (B, T*40)
    raise ValueError(arch)


def _build_arch(arch: str):
    """Instantiate the matching model class."""
    from src.models.deeplob import (
        DeepLOB, LOBCNN_I, LOBCNN_II, LOBLSTM, LOBSimpleMLP,
    )
    return {
        "DeepLOB": lambda: DeepLOB(n_classes=_N_CLASSES, lookback=_LOOKBACK),
        "MLP":     lambda: LOBSimpleMLP(n_classes=_N_CLASSES, lookback=_LOOKBACK),
        "CNN1":    lambda: LOBCNN_I(n_classes=_N_CLASSES, lookback=_LOOKBACK),
        "CNN2":    lambda: LOBCNN_II(n_classes=_N_CLASSES, lookback=_LOOKBACK),
        "LSTM":    lambda: LOBLSTM(n_classes=_N_CLASSES, lookback=_LOOKBACK),
    }[arch]()


def _build_windows(panel, split_value: str):
    """Build sliding-window tensors from the long-format parquet.

    Per FR-009: for each starting tick t, window = features[t:t+lookback];
    target = label_k10 at the END of the window. Filter to the requested
    `split`. Drop windows that span split boundaries (shouldn't happen since
    train/test are stored contiguously).

    Returns (X, y) where X shape (n, lookback, 40), y shape (n,).
    """
    import numpy as np
    import pandas as pd

    sub = panel[panel["split"] == split_value].reset_index(drop=True)
    feats = sub[_FEATURE_COLS].to_numpy(dtype=np.float32)
    labels = sub[_LABEL_COL].to_numpy(dtype=np.int64)
    days = sub["day"].to_numpy()
    n = len(sub)

    Xs, ys = [], []
    for i in range(n - _LOOKBACK):
        # Skip windows that cross a day boundary (no temporal continuity)
        if days[i] != days[i + _LOOKBACK - 1]:
            continue
        Xs.append(feats[i: i + _LOOKBACK])
        ys.append(labels[i + _LOOKBACK - 1])
    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int64),
    )


class _SeqDataset:
    """torch Dataset returning (X, y) per sliding window."""

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
    arch: Literal["DeepLOB", "MLP", "CNN1", "CNN2", "LSTM"],
    device,
    checkpoint_dir: Path,
    max_epochs: int = 100,
    resume_from: str | None = None,
    modal_image_id: str | None = None,
) -> dict[str, float]:
    """Device-agnostic training loop."""
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from src.early_stopper import EarlyStopping

    torch.manual_seed(42)
    np.random.seed(42)

    arch = arch
    if arch not in ("DeepLOB", "MLP", "CNN1", "CNN2", "LSTM"):
        raise ValueError(f"unknown arch {arch!r}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / f"{arch.lower()}_fi2010_k10.pt"
    sidecar_path = checkpoint_dir / f"{arch.lower()}_fi2010_k10.json"

    # ── Data ────────────────────────────────────────────────────────
    parquet = data_dir / "lob_fi2010.parquet"
    log.info("Loading %s", parquet)
    panel = pd.read_parquet(parquet)

    Xtr, ytr = _build_windows(panel, "train")
    Xte, yte = _build_windows(panel, "test")
    log.info("Train: X=%s y=%s; test: X=%s y=%s", Xtr.shape, ytr.shape, Xte.shape, yte.shape)
    log.info("Train class balance: %s", np.bincount(ytr, minlength=_N_CLASSES).tolist())
    log.info("Test  class balance: %s", np.bincount(yte, minlength=_N_CLASSES).tolist())

    # 90/10 train/val split (chronological — last 10% as val)
    n_val = max(1, int(0.1 * len(Xtr)))
    Xva, yva = Xtr[-n_val:], ytr[-n_val:]
    Xtr, ytr = Xtr[:-n_val], ytr[:-n_val]
    log.info("After val split: train=%d, val=%d", len(Xtr), len(Xva))

    train_loader = DataLoader(_SeqDataset(Xtr, ytr), batch_size=64, shuffle=True)
    val_loader = DataLoader(_SeqDataset(Xva, yva), batch_size=128, shuffle=False)
    test_loader = DataLoader(_SeqDataset(Xte, yte), batch_size=128, shuffle=False)

    # ── Model ───────────────────────────────────────────────────────
    model = _build_arch(arch).to(device)
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = checkpoint_dir / resume_path
        log.info("Resuming from %s", resume_path)
        model.load_state_dict(torch.load(resume_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Models output softmax (already normalized) — use NLLLoss on log of output
    # OR convert to logits via log. Simpler: use CrossEntropyLoss on the
    # pre-softmax logits. But our models APPLY softmax in forward(). We can
    # use NLLLoss(log(out + eps)) to stay compatible.
    EPS = 1e-9

    def loss_fn(out, target):
        return torch.nn.functional.nll_loss(torch.log(out + EPS), target)

    early = EarlyStopping(savepath=str(final_path), patience=10,
                          min_delta=1e-3, verbose=False)

    # ── Train loop ──────────────────────────────────────────────────
    epochs_run = 0
    val_loss_history: list[float] = []
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb_seq, yb in train_loader:
            xb_seq = xb_seq.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(_shape_input(arch, xb_seq))
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
        avg_train_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xb_seq, yb in val_loader:
                xb_seq = xb_seq.to(device); yb = yb.to(device)
                out = model(_shape_input(arch, xb_seq))
                val_loss += float(loss_fn(out, yb))
                n_val_batches += 1
        val_loss /= max(n_val_batches, 1)
        val_loss_history.append(val_loss)

        log.info("epoch %d  train_loss=%.4f  val_loss=%.4f",
                 epoch + 1, avg_train_loss, val_loss)

        early(model, val_loss)
        if early.early_stop:
            log.info("Early stopping at epoch %d", epoch + 1)
            break

    if not final_path.exists():
        torch.save(model.state_dict(), final_path)

    # ── Test metrics ────────────────────────────────────────────────
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score,
        precision_score, recall_score,
    )

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb_seq, yb in test_loader:
            xb_seq = xb_seq.to(device); yb = yb.to(device)
            out = model(_shape_input(arch, xb_seq))
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    acc = float(accuracy_score(targets, preds))
    prec = float(precision_score(targets, preds, average="macro", zero_division=0))
    rec = float(recall_score(targets, preds, average="macro", zero_division=0))
    f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    cm = confusion_matrix(targets, preds, labels=[0, 1, 2])

    # ── Sidecar JSON per data-model §E3 ─────────────────────────────
    import torch as _torch
    sidecar = {
        "trained_on": _dt.date.today().isoformat(),
        "trained_with": modal_image_id or _modal_image_id_or_local(),
        "torch_version": _torch.__version__,
        "modal_app": "deep-finance-train-deeplob",
        "arch": arch,
        "dataset": "FI-2010",
        "setup": 2,
        "k": 10,
        "n_classes": _N_CLASSES,
        "data_range": {"train_days": [0], "test_days": [8, 9, 10]},
        "hyperparameters": {
            "lookback": _LOOKBACK,
            "batch_size": 64,
            "lr": 1e-3,
            "epochs_trained": epochs_run,
            "patience": 10,
            "min_delta": 1e-3,
        },
        "final_metrics": {
            "val_loss": val_loss_history[-1] if val_loss_history else None,
            "test_accuracy": acc,
            "test_precision_macro": prec,
            "test_recall_macro": rec,
            "test_f1_macro": f1,
            "confusion_matrix": cm.tolist(),
        },
        "git_commit": os.environ.get("GIT_COMMIT", ""),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    log.info("Wrote sidecar %s", sidecar_path)
    return sidecar["final_metrics"]


# ─── CPU-local CLI ────────────────────────────────────────────────────


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_deeplob",
        description="Train one of {DeepLOB, MLP, CNN1, CNN2, LSTM} on FI-2010.",
    )
    parser.add_argument("--arch", choices=["DeepLOB", "MLP", "CNN1", "CNN2", "LSTM"],
                        default="DeepLOB")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
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
    print(f"[train_deeplob] arch={args.arch} metrics={metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(_cpu_main())
