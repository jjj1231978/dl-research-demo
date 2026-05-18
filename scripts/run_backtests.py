#!/usr/bin/env python3
"""Pre-compute backtest panels consumed by the live Streamlit pages.

Phase 1 surface: `--momentum` produces `data/backtests/momentum_results.parquet`
per FR-013 + specs/002-phase-1-momentum/data-model.md §E4. Rows for all
5 strategies × 2 vol-scaling × 18 contracts × ~4000 trading days
(~720 k rows under 50 MB).

Future phases plug additional flags into the same script (e.g.,
`--portfolio` for Phase 2 Zhang/Zohren/Roberts 2020 Table 1 panel).

Pretrained checkpoints (mlp_sharpe.pt, lstm_sharpe.pt) are optional —
if absent, the deep strategies are skipped with a clear warning.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("run_backtests")

REFERENCE_STRATEGIES = ("long_only", "sgn_returns", "macd")
DEEP_STRATEGIES = ("mlp_sharpe", "lstm_sharpe")
ALL_STRATEGIES = REFERENCE_STRATEGIES + DEEP_STRATEGIES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_backtests",
        description=(
            "Pre-compute backtest panels for the live app. Phase 1 owns "
            "the --momentum flag; later phases will add --portfolio etc."
        ),
    )
    parser.add_argument(
        "--momentum", action="store_true",
        help="Produce data/backtests/momentum_results.parquet (Phase 1).",
    )
    parser.add_argument(
        "--portfolio", action="store_true",
        help="Produce data/backtests/portfolio_results.parquet (Phase 2).",
    )
    parser.add_argument(
        "--lob", action="store_true",
        help="Produce data/backtests/lob_results.parquet (Phase 3).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output dir. Default: ./data/backtests/ (repo-relative).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Input dir holding cme_futures.parquet. Default: data_root().",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=_REPO_ROOT / "data" / "pretrained",
        help="Where to find {mlp,lstm}_sharpe.pt. Default: ./data/pretrained/.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _reference_position(strategy: str, prices):
    """Map strategy name → daily position-in-(-1,+1) series per contract."""
    from src.strategies.tsmom import long_only, macd_ensemble, sgn_returns

    if strategy == "long_only":
        return long_only(prices)
    if strategy == "sgn_returns":
        return sgn_returns(prices, lookback_days=252)
    if strategy == "macd":
        return macd_ensemble(prices)
    raise ValueError(strategy)


def _deep_position(strategy: str, prices, checkpoint_path: Path):
    """Run a pretrained deep model over the wide price panel.

    Builds the same 8-feature tensor as
    `src.training.train_deep_momentum._build_features` but in production
    (no train/test split). Returns a wide DataFrame of positions aligned
    to ``prices.index``.
    """
    import numpy as np
    import pandas as pd
    import torch

    from src.models.deep_momentum import DeepMomentumLSTM, DeepMomentumMLP
    from src.strategies.tsmom import macd_signal

    arch = "MLP" if strategy == "mlp_sharpe" else "LSTM"
    model_cls = DeepMomentumMLP if arch == "MLP" else DeepMomentumLSTM
    model = model_cls()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    seq_length = 60
    contracts = list(prices.columns)

    def _norm(_h):
        roll_ret = prices.pct_change(_h)
        vol = roll_ret.ewm(span=63, adjust=False).std()
        return roll_ret / vol
    feats = [_norm(h) for h in (1, 21, 63, 126, 252)]
    feats += [macd_signal(prices, s, l) for s, l in ((8, 24), (16, 48), (32, 96))]

    positions = pd.DataFrame(np.nan, index=prices.index, columns=contracts)
    positions.columns.name = "contract"  # preserve axis-name through stack()
    positions.index.name = "date"
    with torch.no_grad():
        for contract in contracts:
            feat_mat = np.column_stack([f[contract].to_numpy() for f in feats])
            for i in range(seq_length, len(feat_mat)):
                window = feat_mat[i - seq_length: i]
                if np.isnan(window).any():
                    continue
                x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
                p = float(model(x).item())
                positions.iloc[i, positions.columns.get_loc(contract)] = p
    return positions


def _build_momentum_panel(data_dir: Path, checkpoint_dir: Path):
    """Build the long-format E4 panel for all (strategy, vol_scaling, contract)."""
    import pandas as pd

    from src.strategies.vol_targeting import vol_target

    parquet = data_dir / "cme_futures.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"{parquet} missing. Run `python scripts/fetch_futures.py` first."
        )
    panel = pd.read_parquet(parquet)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    wide = panel.pivot(index="date", columns="contract", values="price").sort_index()
    rets = wide.pct_change(1)

    available_deep: list[str] = []
    deep_checkpoints: dict[str, Path] = {}
    for s in DEEP_STRATEGIES:
        ckpt = checkpoint_dir / f"{s.split('_')[0]}_sharpe.pt"
        if ckpt.exists():
            available_deep.append(s)
            deep_checkpoints[s] = ckpt
        else:
            log.warning(
                "Skipping %s — checkpoint %s not found. Train via "
                "`modal run src/training/train_deep_momentum.py --arch %s`.",
                s, ckpt, s.split('_')[0].upper(),
            )
    strategies_to_run = list(REFERENCE_STRATEGIES) + available_deep

    rows: list[pd.DataFrame] = []
    for strategy in strategies_to_run:
        log.info("backtest: %s", strategy)
        if strategy in REFERENCE_STRATEGIES:
            pos = _reference_position(strategy, wide)
        else:
            pos = _deep_position(strategy, wide, deep_checkpoints[strategy])

        # Per-contract daily P&L = position × next-day return
        pnl = pos.shift(1) * rets

        for vol_scaling in (False, True):
            scaled = vol_target(pnl, target_vol=0.15) if vol_scaling else pnl
            stacked = scaled.stack().rename("daily_return").reset_index()
            stacked = stacked.dropna(subset=["daily_return"])
            stacked["strategy"] = strategy
            stacked["vol_scaling"] = vol_scaling
            stacked = stacked[["date", "contract", "strategy", "vol_scaling", "daily_return"]]
            rows.append(stacked)

    if not rows:
        raise RuntimeError("No strategy rows produced — aborting.")

    out = pd.concat(rows, ignore_index=True)
    # Determinism: stable sort by all key columns
    out = out.sort_values(
        ["strategy", "vol_scaling", "contract", "date"], kind="stable"
    ).reset_index(drop=True)
    return out


# ──────────────────────────────────────────────────────────────────────
# Phase 2 — Portfolio panel
# ──────────────────────────────────────────────────────────────────────

PORTFOLIO_UNIVERSES = ("etfs", "20stock")
PORTFOLIO_PARQUET_MAP = {"etfs": "etf_basket", "20stock": "sp500_20"}

# (vol_scaling, cost_rate) panel pairs from paper Table 1
PORTFOLIO_PANELS = (
    (False, 0.0001),
    (True,  0.0001),
    (True,  0.0010),
)

# Methods available in every universe
PORTFOLIO_UNIVERSAL_METHODS = (
    "equal_weight", "min_variance", "max_diversification", "diversity_weighted",
)

# Methods available only for the 4-ETF basket
PORTFOLIO_ETF_ONLY_METHODS = (
    "alloc_25_25_25_25", "alloc_50_10_20_20",
    "alloc_10_50_20_20", "alloc_40_40_10_10",
)

ETF_FIXED_ALLOCATIONS = {
    "alloc_25_25_25_25": {"VTI": 0.25, "AGG": 0.25, "DBC": 0.25, "VIXY": 0.25},
    "alloc_50_10_20_20": {"VTI": 0.50, "AGG": 0.10, "DBC": 0.20, "VIXY": 0.20},
    "alloc_10_50_20_20": {"VTI": 0.10, "AGG": 0.50, "DBC": 0.20, "VIXY": 0.20},
    "alloc_40_40_10_10": {"VTI": 0.40, "AGG": 0.40, "DBC": 0.10, "VIXY": 0.10},
}


def _rolling_classical_weights(close_wide, method, lookback=50):
    """Per day t, compute classical weights from the last `lookback` returns
    up to t-1 (no look-ahead). Returns wide DataFrame (date × asset).
    """
    import numpy as np
    import pandas as pd

    from src.strategies.portfolios import (
        equal_weight, max_diversification, min_variance,
    )

    rets = close_wide.pct_change(1)
    assets = list(close_wide.columns)
    n_assets = len(assets)
    weights = pd.DataFrame(np.nan, index=close_wide.index, columns=assets)
    weights.columns.name = "symbol"
    weights.index.name = "date"

    if method == "equal_weight":
        weights.loc[:, :] = 1.0 / n_assets
        return weights

    # Rolling solver methods
    arr = rets.to_numpy()
    dates = rets.index
    for i in range(lookback, len(dates)):
        window = arr[i - lookback: i]
        if np.isnan(window).any():
            continue
        if method == "min_variance":
            w = min_variance(window)
        elif method == "max_diversification":
            w = max_diversification(window)
        else:
            raise ValueError(f"unknown rolling-classical method: {method}")
        weights.iloc[i] = w
    return weights


def _diversity_weighted_weights(panel, close_wide):
    """Per-day weights from market caps (close × shares_outstanding)."""
    import numpy as np
    import pandas as pd

    from src.strategies.portfolios import diversity_weighted

    if "shares_outstanding" not in panel.columns:
        return None
    so_wide = panel.pivot(index="date", columns="symbol", values="shares_outstanding")
    so_wide.index = pd.to_datetime(so_wide.index)
    so_wide = so_wide.reindex(close_wide.index).ffill()
    # Require ≥ 50% of rows to have a valid market-cap vector
    valid_rows = so_wide.notna().all(axis=1).sum()
    if valid_rows < 0.5 * len(close_wide):
        log.warning(
            "  diversity_weighted: only %d/%d rows have full shares_outstanding "
            "— skipping (FMP coverage limit)", valid_rows, len(close_wide),
        )
        return None

    weights = pd.DataFrame(np.nan, index=close_wide.index, columns=close_wide.columns)
    weights.columns.name = "symbol"
    weights.index.name = "date"
    for i, d in enumerate(close_wide.index):
        mcaps = (close_wide.iloc[i] * so_wide.iloc[i]).to_numpy()
        if np.isnan(mcaps).any() or mcaps.sum() <= 0:
            continue
        try:
            weights.iloc[i] = diversity_weighted(mcaps, p=0.5)
        except ValueError:
            continue
    return weights


def _fixed_allocation_weights(close_wide, alloc_dict):
    """Constant-weight series for paper Table 1's 4 ETF fixed allocations."""
    import numpy as np
    import pandas as pd

    from src.strategies.portfolios import fixed_allocation

    assets = list(close_wide.columns)
    w = fixed_allocation(alloc_dict, assets)
    weights = pd.DataFrame(np.tile(w, (len(close_wide), 1)),
                            index=close_wide.index, columns=assets)
    weights.columns.name = "symbol"
    weights.index.name = "date"
    return weights


def _deep_portfolio_weights(close_wide, checkpoint_path, n_assets):
    """Run the pretrained DeepPortfolioMLP forward over the whole window."""
    import numpy as np
    import pandas as pd
    import torch

    from src.models.deep_portfolio import DeepPortfolioMLP

    lookback = 50
    model = DeepPortfolioMLP(n_assets=n_assets)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    close_arr = close_wide.to_numpy(dtype=np.float64)
    rets_arr = close_wide.pct_change(1).to_numpy()
    close_normed = close_arr / np.where(
        close_arr > 0,
        pd.DataFrame(close_arr).ewm(span=63, adjust=False).mean().to_numpy(),
        1.0,
    )
    per_asset = np.stack([close_normed, rets_arr], axis=-1)  # (T, N, 2)

    weights = pd.DataFrame(np.nan, index=close_wide.index, columns=close_wide.columns)
    weights.columns.name = "symbol"
    weights.index.name = "date"

    with torch.no_grad():
        for i in range(lookback, len(close_wide)):
            window = per_asset[i - lookback: i].reshape(lookback, n_assets * 2)
            if np.isnan(window).any():
                continue
            x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            w = model(x).squeeze(0).numpy()
            weights.iloc[i] = w
    return weights


def _apply_vol_target(daily_returns, target_vol=0.10, ewma_span=60):
    """σ_target rescaling per paper §4.4 (default 10%)."""
    import numpy as np
    from src.strategies.vol_targeting import vol_target
    return vol_target(daily_returns, target_vol=target_vol, ewma_span=ewma_span)


def _build_portfolio_panel(data_dir: Path, checkpoint_dir: Path):
    """Build the long-format E4 panel for all (universe, method, vol_scaling, cost_rate)."""
    import pandas as pd

    rows: list[pd.DataFrame] = []

    for universe in PORTFOLIO_UNIVERSES:
        parquet_name = PORTFOLIO_PARQUET_MAP[universe]
        parquet_path = data_dir / f"{parquet_name}.parquet"
        if not parquet_path.exists():
            log.warning("skip %s — %s missing", universe, parquet_path)
            continue
        log.info("Loading %s for %s", parquet_path, universe)
        panel = pd.read_parquet(parquet_path)
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        close_wide = panel.pivot(index="date", columns="symbol", values="close").sort_index()
        close_wide = close_wide.ffill().dropna(how="all")
        rets_wide = close_wide.pct_change(1)

        # Discover method weight series for this universe
        method_weights: dict = {}
        for m in PORTFOLIO_UNIVERSAL_METHODS:
            if m == "diversity_weighted":
                w = _diversity_weighted_weights(panel, close_wide)
                if w is None:
                    log.warning("  skip diversity_weighted for %s — no shares_outstanding", universe)
                    continue
                method_weights[m] = w
            else:
                method_weights[m] = _rolling_classical_weights(close_wide, m)
        if universe == "etfs":
            for alloc_name, alloc_dict in ETF_FIXED_ALLOCATIONS.items():
                method_weights[alloc_name] = _fixed_allocation_weights(close_wide, alloc_dict)

        # Deep portfolio if checkpoint available
        ckpt = checkpoint_dir / f"deep_portfolio_{universe}.pt"
        if ckpt.exists():
            log.info("  deep portfolio: %s", ckpt)
            method_weights["deep_portfolio"] = _deep_portfolio_weights(
                close_wide, ckpt, n_assets=len(close_wide.columns),
            )
        else:
            log.warning(
                "  skip deep_portfolio for %s — checkpoint %s missing. "
                "Train via `modal run src/training/train_deep_portfolio.py --universe %s`.",
                universe, ckpt, universe,
            )

        # For each (method, vol_scaling, cost_rate) emit per-day portfolio P&L
        for method, weights in method_weights.items():
            log.info("  %s × %s", universe, method)
            # Per-day raw P&L = (lagged weights * today's returns).sum across assets
            pnl_raw = (weights.shift(1) * rets_wide).sum(axis=1)
            # Per-day turnover cost = cost_rate * |Δw|.sum
            turnover = (weights.shift(1) - weights).abs().sum(axis=1).fillna(0)

            for vol_scaling, cost_rate in PORTFOLIO_PANELS:
                pnl_net = pnl_raw - cost_rate * turnover
                if vol_scaling:
                    pnl_net = _apply_vol_target(pnl_net, target_vol=0.10, ewma_span=60)
                df = pnl_net.dropna().rename("portfolio_return").reset_index()
                df["universe"] = universe
                df["method"] = method
                df["vol_scaling"] = vol_scaling
                df["cost_rate"] = float(cost_rate)
                df = df[["date", "universe", "method", "vol_scaling", "cost_rate", "portfolio_return"]]
                rows.append(df)

    if not rows:
        raise RuntimeError("No portfolio rows produced — check parquet availability.")

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(
        ["universe", "method", "vol_scaling", "cost_rate", "date"], kind="stable"
    ).reset_index(drop=True)
    return out


# ──────────────────────────────────────────────────────────────────────
# Phase 3 — LOB metrics panel
# ──────────────────────────────────────────────────────────────────────

LOB_ARCHS = ("DeepLOB", "MLP", "CNN1", "CNN2", "LSTM")

# Paper-reported numbers from Zhang et al. 2019 Table II (Setup 2, k=10).
# These are surfaced in Tab 4 with a "(paper-reported)" badge for the
# baselines we don't reproduce locally.
LOB_PAPER_REPORTED = (
    # (method, accuracy, precision, recall, f1)  — all percentages
    ("SVM", 0.486, 0.491, 0.486, 0.487),
    ("BoF", 0.572, 0.490, 0.460, 0.460),
    ("MCSDA", 0.737, 0.460, 0.479, 0.467),
    ("B(TABL)", 0.788, 0.789, 0.788, 0.785),
    ("C(TABL)", 0.842, 0.851, 0.842, 0.844),
)


def _build_lob_panel(data_dir: Path, checkpoint_dir: Path):
    """Produce per-method metrics + confusion matrices for FI-2010 Setup 2 k=10."""
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score,
        precision_score, recall_score,
    )

    from src.models.deeplob import (
        DeepLOB, LOBCNN_I, LOBCNN_II, LOBLSTM, LOBSimpleMLP,
    )
    from src.strategies.lob_classical import fit_lda, predict_lda

    parquet = data_dir / "lob_fi2010.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"{parquet} missing. Run `python scripts/fetch_lob_fi2010.py` first."
        )
    panel = pd.read_parquet(parquet)
    log.info("LOB panel loaded: %d rows", len(panel))

    # Build sliding-window arrays for train (LDA) and test (eval)
    from src.training.train_deeplob import _build_windows, _shape_input
    Xtr, ytr = _build_windows(panel, "train")
    Xte, yte = _build_windows(panel, "test")
    log.info("train windows: %s; test windows: %s", Xtr.shape, Xte.shape)

    rows = []

    # ── Deep models ────────────────────────────────────────────────
    arch_map = {
        "DeepLOB": DeepLOB,
        "MLP": LOBSimpleMLP,
        "CNN1": LOBCNN_I,
        "CNN2": LOBCNN_II,
        "LSTM": LOBLSTM,
    }
    for arch_name, cls in arch_map.items():
        ckpt = checkpoint_dir / f"{arch_name.lower()}_fi2010_k10.pt"
        if not ckpt.exists():
            log.warning(
                "Skipping %s — checkpoint %s missing. Train via "
                "`modal run src/training/train_deeplob.py --arch %s`.",
                arch_name, ckpt, arch_name,
            )
            continue
        log.info("Evaluating %s on test set …", arch_name)
        model = cls()
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        preds = []
        BATCH = 256
        with torch.no_grad():
            for i in range(0, len(Xte), BATCH):
                xb = torch.from_numpy(Xte[i: i + BATCH])
                out = model(_shape_input(arch_name, xb))
                preds.append(out.argmax(dim=1).numpy())
        preds = np.concatenate(preds)
        cm = confusion_matrix(yte, preds, labels=[0, 1, 2])
        row = {
            "method": arch_name.lower(),
            "k": 10,
            "accuracy": float(accuracy_score(yte, preds)),
            "precision_macro": float(precision_score(yte, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(yte, preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(yte, preds, average="macro", zero_division=0)),
            "source": "reproduced_here",
        }
        for i in range(3):
            for j in range(3):
                row[f"cm_{i}{j}"] = int(cm[i, j])
        rows.append(row)

    # ── LDA classical (sklearn, no Modal) ───────────────────────────
    # LDA scales O(min(n,p)^3) and we have p = 100*40 = 4000 features, so
    # we subsample training windows to keep the fit tractable on CPU
    # (full ~254k windows × 4000 features = 4 GB and ~hours of SVD).
    _LDA_TRAIN_SAMPLES = 30_000
    log.info("Fitting LDA on flattened LOB features …")
    Xtr_flat = Xtr.reshape(Xtr.shape[0], -1).astype(np.float32)
    Xte_flat = Xte.reshape(Xte.shape[0], -1).astype(np.float32)
    if len(Xtr_flat) > _LDA_TRAIN_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(Xtr_flat), size=_LDA_TRAIN_SAMPLES, replace=False)
        Xtr_flat = Xtr_flat[idx]
        ytr_sub = ytr[idx]
        log.info("  LDA subsample: %d train windows (of %d)", _LDA_TRAIN_SAMPLES, len(Xtr))
    else:
        ytr_sub = ytr
    try:
        lda = fit_lda(Xtr_flat, ytr_sub)
        preds = predict_lda(lda, Xte_flat)
        cm = confusion_matrix(yte, preds, labels=[0, 1, 2])
        row = {
            "method": "lda",
            "k": 10,
            "accuracy": float(accuracy_score(yte, preds)),
            "precision_macro": float(precision_score(yte, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(yte, preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(yte, preds, average="macro", zero_division=0)),
            "source": "reproduced_here",
        }
        for i in range(3):
            for j in range(3):
                row[f"cm_{i}{j}"] = int(cm[i, j])
        rows.append(row)
        log.info("LDA F1=%.3f", row["f1_macro"])
    except Exception as exc:  # noqa: BLE001
        log.warning("LDA fit failed: %s", exc)

    # ── Paper-reported baselines (no confusion matrix) ──────────────
    for method, acc, prec, rec, f1 in LOB_PAPER_REPORTED:
        row = {
            "method": method.lower(),
            "k": 10,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
            "source": "paper_reported",
        }
        for i in range(3):
            for j in range(3):
                row[f"cm_{i}{j}"] = -1  # sentinel: no CM available
        rows.append(row)

    if not rows:
        raise RuntimeError("No LOB rows produced.")
    out = pd.DataFrame(rows)
    out = out.sort_values(["source", "f1_macro"], ascending=[True, False]).reset_index(drop=True)
    return out


def _atomic_write_parquet(df, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp, engine="pyarrow", index=False)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.data import data_root

    if not (args.momentum or args.portfolio or args.lob):
        print("nothing to do — pass --momentum (P1), --portfolio (P2), or --lob (P3).")
        return 0

    data_dir = args.data_dir or data_root()
    out_dir = args.out_dir or (_REPO_ROOT / "data" / "backtests")

    if args.momentum:
        panel = _build_momentum_panel(data_dir, args.checkpoint_dir)
        target = out_dir / "momentum_results.parquet"
        _atomic_write_parquet(panel, target)
        print(
            f"[run_backtests] momentum_results.parquet: {len(panel)} rows, "
            f"{panel['strategy'].nunique()} strategies, "
            f"{panel['contract'].nunique()} contracts → {target}"
        )

    if args.portfolio:
        panel = _build_portfolio_panel(data_dir, args.checkpoint_dir)
        target = out_dir / "portfolio_results.parquet"
        _atomic_write_parquet(panel, target)
        print(
            f"[run_backtests] portfolio_results.parquet: {len(panel)} rows, "
            f"{panel['universe'].nunique()} universes, "
            f"{panel['method'].nunique()} methods → {target}"
        )

    if args.lob:
        panel = _build_lob_panel(data_dir, args.checkpoint_dir)
        target = out_dir / "lob_results.parquet"
        _atomic_write_parquet(panel, target)
        print(
            f"[run_backtests] lob_results.parquet: {len(panel)} rows, "
            f"{panel['method'].nunique()} methods "
            f"({(panel['source']=='reproduced_here').sum()} reproduced, "
            f"{(panel['source']=='paper_reported').sum()} paper-reported) → {target}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
