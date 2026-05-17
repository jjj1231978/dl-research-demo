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

    if not args.momentum:
        print("nothing to do — pass --momentum (Phase 1) to produce a panel.")
        return 0

    data_dir = args.data_dir or data_root()
    out_dir = args.out_dir or (_REPO_ROOT / "data" / "backtests")
    panel = _build_momentum_panel(data_dir, args.checkpoint_dir)

    target = out_dir / "momentum_results.parquet"
    _atomic_write_parquet(panel, target)
    print(
        f"[run_backtests] momentum_results.parquet: {len(panel)} rows, "
        f"{panel['strategy'].nunique()} strategies, "
        f"{panel['contract'].nunique()} contracts → {target}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
