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

    if not (args.momentum or args.portfolio):
        print("nothing to do — pass --momentum (Phase 1) or --portfolio (Phase 2).")
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
