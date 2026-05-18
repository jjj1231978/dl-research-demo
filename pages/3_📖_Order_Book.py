"""Page 3 — Order Book (Zhang, Zohren, Roberts 2019).

Spec: specs/004-phase-3-deeplob/spec.md
Contract: specs/004-phase-3-deeplob/contracts/order_book_page_ui.md

Replicates Table II of the paper on FI-2010 Setup 2, k=10. The page reads
a small demo slice (`data/lob_fi2010_demo.parquet`, ~10 MB) plus a
pre-computed metrics panel (`data/backtests/lob_results.parquet`).
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import render_data_status_sidebar
from src.models.deeplob import (
    DeepLOB,
    LOBCNN_I,
    LOBCNN_II,
    LOBLSTM,
    LOBSimpleMLP,
)
from src.strategies.lob_classical import fit_lda, predict_lda

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRETRAINED_DIR = _REPO_ROOT / "data" / "pretrained"
_DEMO_PARQUET = _REPO_ROOT / "data" / "lob_fi2010_demo.parquet"

_FEATURE_COLS = [f"f{i:02d}" for i in range(40)]
_HORIZONS = (10, 20, 30, 50, 100)
_LOOKBACK = 100
_CLASS_NAMES = ("down", "stationary", "up")
# In-page LDA fit + DeepLOB inference must fit in HF CPU Basic's 16 GB RAM
# AND complete in a few seconds. The full demo slice (~55k ticks) blows up to
# ~880 MB of windows and takes 240+ s for the LDA SVD; we cap to the first
# _IN_PAGE_TICK_CAP ticks for in-page work. Tab 4A still reports the full-set
# numbers from `data/backtests/lob_results.parquet`.
_IN_PAGE_TICK_CAP = 3000

# Per FI-2010 NoAuction_DecPre_CF spec: features are ordered
# (ask_p_1, ask_v_1, bid_p_1, bid_v_1, ask_p_2, ask_v_2, …) for 10 levels.
_ASK_PRICE_IDX = [4 * i + 0 for i in range(10)]
_ASK_VOL_IDX = [4 * i + 1 for i in range(10)]
_BID_PRICE_IDX = [4 * i + 2 for i in range(10)]
_BID_VOL_IDX = [4 * i + 3 for i in range(10)]

_ARCH_TO_CLS = {
    "deeplob": DeepLOB,
    "mlp": LOBSimpleMLP,
    "cnn1": LOBCNN_I,
    "cnn2": LOBCNN_II,
    "lstm": LOBLSTM,
}

_ARCH_LABEL = {
    "deeplob": "DeepLOB",
    "mlp": "MLP",
    "cnn1": "CNN-I",
    "cnn2": "CNN-II",
    "lstm": "LSTM",
    "lda": "LDA",
    "svm": "SVM",
    "bof": "BoF",
    "mcsda": "MCSDA",
    "b(tabl)": "B(TABL)",
    "c(tabl)": "C(TABL)",
}


st.set_page_config(
    page_title="Order Book — Deep Finance Showcase",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _backtests_dir() -> Path:
    raw = os.environ.get("DEEP_FINANCE_BACKTESTS_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return _REPO_ROOT / "data" / "backtests"


def _demo_path() -> Path:
    raw = os.environ.get("DEEP_FINANCE_LOB_DEMO_PARQUET")
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEMO_PARQUET


def _pretrained_dir() -> Path:
    raw = os.environ.get("DEEP_FINANCE_PRETRAINED_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return _PRETRAINED_DIR


@st.cache_data(show_spinner=False)
def _load_demo(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def _load_panel(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def _checkpoint_present(arch_lower: str) -> bool:
    return (_pretrained_dir() / f"{arch_lower}_fi2010_k10.pt").exists()


def _load_sidecar(arch_lower: str) -> dict | None:
    p = _pretrained_dir() / f"{arch_lower}_fi2010_k10.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _shape_for_arch(arch_lower: str, x_seq: "np.ndarray | torch.Tensor"):
    import torch as _torch
    if not isinstance(x_seq, _torch.Tensor):
        x_seq = _torch.from_numpy(np.asarray(x_seq, dtype=np.float32))
    if arch_lower in ("deeplob", "cnn1", "cnn2"):
        return x_seq.unsqueeze(1) if x_seq.dim() == 3 else x_seq.unsqueeze(0).unsqueeze(0)
    if arch_lower == "lstm":
        return x_seq if x_seq.dim() == 3 else x_seq.unsqueeze(0)
    if arch_lower == "mlp":
        return x_seq.reshape(x_seq.shape[0], -1) if x_seq.dim() == 3 else x_seq.reshape(1, -1)
    raise ValueError(arch_lower)


@st.cache_data(show_spinner=False)
def _build_windows_capped(demo_path_str: str, label_col: str, cap: int):
    """Cached sliding-window builder.

    Reads the demo parquet, keeps the first `cap` ticks, returns
    (X, y) where X has shape (N-T+1, T, 40), y shape (N-T+1,).
    Key on (demo_path, label_col, cap) so multiple horizons share the parquet read.
    """
    p = Path(demo_path_str)
    if not p.exists():
        return np.zeros((0, _LOOKBACK, 40), dtype=np.float32), np.zeros(0, dtype=np.int64)
    df = pd.read_parquet(p)
    if len(df) > cap:
        df = df.iloc[:cap]
    feats = df[_FEATURE_COLS].to_numpy(dtype=np.float32)
    labels = df[label_col].to_numpy(dtype=np.int64)
    n = len(df)
    if n < _LOOKBACK + 1:
        return np.zeros((0, _LOOKBACK, 40), dtype=np.float32), np.zeros(0, dtype=np.int64)
    n_windows = n - _LOOKBACK + 1
    X = np.stack([feats[i: i + _LOOKBACK] for i in range(n_windows)])
    y = labels[_LOOKBACK - 1: _LOOKBACK - 1 + n_windows]
    return X, y


@st.cache_data(show_spinner=False)
def _fit_lda_cached(demo_path_str: str, label_col: str, cap: int):
    """Fit LDA on the first 70% of the capped demo windows. Return (preds, yte).

    Cached so the ~5-10 s SVD only runs once per (path, horizon) per session.
    """
    X, y = _build_windows_capped(demo_path_str, label_col, cap)
    if len(X) < 20:
        return None, None
    n_split = int(0.7 * len(X))
    Xtr = X[:n_split].reshape(n_split, -1).astype(np.float32)
    Xte = X[n_split:].reshape(len(X) - n_split, -1).astype(np.float32)
    ytr, yte = y[:n_split], y[n_split:]
    try:
        lda = fit_lda(Xtr, ytr)
        preds = predict_lda(lda, Xte)
    except Exception:  # noqa: BLE001
        return None, None
    return preds, yte


@st.cache_resource(show_spinner=False)
def _load_deeplob_cached(ckpt_path_str: str):
    """Load the DeepLOB checkpoint once per session."""
    import torch as _torch
    model = DeepLOB()
    model.load_state_dict(_torch.load(ckpt_path_str, map_location="cpu"))
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def _deeplob_infer_cached(ckpt_path_str: str, demo_path_str: str,
                            label_col: str, cap: int):
    """Cached DeepLOB inference over the capped demo windows. Returns (probs, y)."""
    import torch as _torch
    X, y = _build_windows_capped(demo_path_str, label_col, cap)
    if len(X) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int64)
    model = _load_deeplob_cached(ckpt_path_str)
    BATCH = 256
    probs_chunks = []
    with _torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = _torch.from_numpy(X[i: i + BATCH]).unsqueeze(1)
            probs_chunks.append(model(xb).numpy())
    return np.concatenate(probs_chunks, axis=0), y


def _mid_price_proxy(demo: pd.DataFrame) -> np.ndarray:
    """Mid-price proxy from the normalized features.

    Features are z-score normalized so absolute price is not recoverable;
    we use (ask_price_1 + bid_price_1)/2 as a normalized mid-price series
    which still reflects the local trend.
    """
    ap1 = demo["f00"].to_numpy(dtype=np.float64)
    bp1 = demo["f02"].to_numpy(dtype=np.float64)
    return 0.5 * (ap1 + bp1)


# ── Sidebar ───────────────────────────────────────────────────────────

st.sidebar.title("📖 Limit Order Book")
st.sidebar.markdown("**Zhang, Zohren, Roberts (2019)** — FI-2010 Setup 2.")

horizon_k = st.sidebar.selectbox(
    "Prediction horizon (k)",
    options=list(_HORIZONS),
    index=0,
    key="lob_horizon_k",
    help="Only k=10 is reproduced locally; others are paper-reported only.",
)

classical_choice = st.sidebar.selectbox(
    "Classical baseline",
    options=["LDA"],
    key="lob_classical",
)

demo_present = _demo_path().exists()
demo = _load_demo(str(_demo_path())) if demo_present else pd.DataFrame()
n_demo = len(demo)
mid_tick = max(0, n_demo // 2)
tick_max = max(0, n_demo - 1)

tick_idx = st.sidebar.slider(
    "Tick index in demo slice",
    min_value=0,
    max_value=max(1, tick_max),
    value=min(mid_tick, tick_max) if tick_max else 0,
    key="lob_tick",
    disabled=not demo_present,
)

st.sidebar.divider()
render_data_status_sidebar(st.sidebar)


# ── Header ────────────────────────────────────────────────────────────

st.title("📖 Order Book — Zhang, Zohren, Roberts (2019)")
st.markdown(
    "*DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*, "
    "IEEE TSP — [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)"
)
st.markdown(
    "> A CNN + Inception + LSTM that learns universal LOB microstructure "
    "features for 3-class mid-price-movement prediction."
)


# ── Data preflight ───────────────────────────────────────────────────

panel = _load_panel(str(_backtests_dir() / "lob_results.parquet"))


if not demo_present:
    st.warning(
        "**Demo slice missing — page is in skeleton mode.** "
        "Run `python scripts/fetch_lob_fi2010.py -v` to produce "
        "`data/lob_fi2010_demo.parquet` (and the full `lob_fi2010.parquet` "
        "on your machine for training). The page will render skeleton "
        "tabs without crashing, but Tabs 1/2/3 need the demo slice to "
        "show LOB snapshots and predictions."
    )


# ── Tabs ──────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Problem & Data", "2. Classical Baselines",
     "3. DeepLOB", "4. Table II Replication"]
)


with tab1:
    st.subheader("Why limit-order-book prediction?")
    st.markdown(
        "A limit order book (LOB) records all outstanding buy and sell "
        "orders at each price level. Microstructure features — bid/ask "
        "imbalance, queue dynamics, level-2 depth — carry short-horizon "
        "predictive signal about the next mid-price move.\n\n"
        "Zhang, Zohren, Roberts (2019) frame this as a 3-class classification "
        "(down / stationary / up at k ticks ahead) and learn the features "
        "end-to-end with a CNN + Inception + LSTM stack — no hand-engineered "
        "imbalance ratios. We replicate Table II of their paper on the "
        "**FI-2010 benchmark** (Ntakaris et al. 2017): 10 days × 5 Nasdaq "
        "Nordic stocks, train on days 1–7, test on days 8–10."
    )

    if demo_present:
        c1, c2, c3 = st.columns(3)
        c1.metric("Demo ticks", f"{n_demo:,}")
        c2.metric("Day", int(demo["day"].iloc[0]))
        c3.metric("Stock", "1 (KESBV)")

        feats_now = demo.iloc[tick_idx][_FEATURE_COLS].to_numpy(dtype=np.float64)
        ask_p = feats_now[_ASK_PRICE_IDX]
        ask_v = feats_now[_ASK_VOL_IDX]
        bid_p = feats_now[_BID_PRICE_IDX]
        bid_v = feats_now[_BID_VOL_IDX]
        snap = pd.DataFrame({
            "level": list(range(1, 11)) * 2,
            "side": ["ask"] * 10 + ["bid"] * 10,
            "price": np.concatenate([ask_p, bid_p]),
            "volume_z": np.concatenate([ask_v, bid_v]),
        })
        st.markdown(f"**LOB snapshot — tick {tick_idx}** "
                    "(features are z-score normalized per FI-2010)")
        fig_snap = px.bar(
            snap, x="level", y="volume_z", color="side", barmode="group",
            color_discrete_map={"ask": "#d62728", "bid": "#2ca02c"},
            height=300,
        )
        fig_snap.update_layout(yaxis_title="Volume (z-scored)",
                                xaxis_title="Depth level")
        st.plotly_chart(fig_snap, width="stretch")

        # Class-balance per horizon
        st.markdown("**Class balance across horizons (demo slice)**")
        bal_rows = []
        for k in _HORIZONS:
            col = f"label_k{k}"
            if col not in demo.columns:
                continue
            counts = demo[col].value_counts().sort_index()
            for cls in (0, 1, 2):
                bal_rows.append({
                    "k": k,
                    "class": _CLASS_NAMES[cls],
                    "count": int(counts.get(cls, 0)),
                })
        if bal_rows:
            bal = pd.DataFrame(bal_rows)
            fig_bal = px.bar(
                bal, x="k", y="count", color="class", barmode="stack",
                color_discrete_map={"down": "#d62728", "stationary": "#bbbbbb",
                                       "up": "#2ca02c"},
                height=280,
            )
            fig_bal.update_layout(xaxis_title="Prediction horizon k (ticks)")
            st.plotly_chart(fig_bal, width="stretch")

        # Smoothed-label visualization: mid-price proxy with up/down shading
        st.markdown(
            f"**Smoothed labels at k={horizon_k} (mid-price proxy with class shading)**"
        )
        mid = _mid_price_proxy(demo)
        sample = min(2000, len(demo))
        idx = np.linspace(0, len(demo) - 1, sample, dtype=int)
        labels_col = f"label_k{horizon_k}"
        lab = demo[labels_col].to_numpy()
        fig_smooth = go.Figure()
        fig_smooth.add_trace(go.Scatter(
            x=idx, y=mid[idx], mode="lines",
            name="Mid-price proxy (z-scored)",
            line={"color": "#1f77b4"},
        ))
        for cls, color in (
            (0, "rgba(214,39,40,0.20)"),
            (2, "rgba(44,160,44,0.20)"),
        ):
            mask = lab[idx] == cls
            fig_smooth.add_trace(go.Scatter(
                x=idx[mask], y=mid[idx][mask], mode="markers",
                name=f"label = {_CLASS_NAMES[cls]}",
                marker={"color": color, "size": 4},
            ))
        fig_smooth.update_layout(height=340, xaxis_title="Tick (demo slice)",
                                   yaxis_title="Mid-price proxy")
        st.plotly_chart(fig_smooth, width="stretch")
    else:
        st.info("LOB snapshot, class-balance, and smoothed-label visualizations require the demo slice.")

    st.divider()
    st.markdown(
        "### Substrate disclosure\n"
        "We replicate **Zhang et al. 2019 Table II** on the **FI-2010** "
        "benchmark (Ntakaris et al. 2017, *Forecasting stock prices from "
        "the limit order book using convolutional neural networks*). "
        "FI-2010 is the standard public LOB dataset: 10 trading days × 5 "
        "Nasdaq Nordic stocks, z-score-normalized (`NoAuction_DecPre_CF`), "
        "with pre-computed 3-class labels at k ∈ {10, 20, 30, 50, 100} ticks. "
        "Only Setup 2 (train days 1–7, test days 8–10) and k=10 are "
        "reproduced here; other horizons + baselines (SVM, BoF, MCSDA, "
        "TABL variants) appear in Tab 4 with a **(paper-reported)** badge."
    )


with tab2:
    st.subheader(f"Classical baseline — {classical_choice}")
    if not demo_present:
        st.info(
            "The classical baseline runs on the demo slice. "
            "Produce `data/lob_fi2010_demo.parquet` first."
        )
    elif n_demo < _LOOKBACK + 1:
        st.warning(
            f"Demo slice has {n_demo} ticks but a single window requires "
            f"{_LOOKBACK + 1} — increase the slice size to enable classical predictions."
        )
    else:
        cap = min(_IN_PAGE_TICK_CAP, n_demo)
        preds, yte = _fit_lda_cached(str(_demo_path()), f"label_k{horizon_k}", cap)
        if preds is None:
            st.warning("LDA fit failed on this slice — see logs.")
        elif len(yte):
            from sklearn.metrics import (
                accuracy_score, f1_score,
                precision_score, recall_score,
            )
            acc = accuracy_score(yte, preds)
            f1 = f1_score(yte, preds, average="macro", zero_division=0)
            prec = precision_score(yte, preds, average="macro", zero_division=0)
            rec = recall_score(yte, preds, average="macro", zero_division=0)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc * 100:.1f}%")
            c2.metric("Precision (macro)", f"{prec * 100:.1f}%")
            c3.metric("Recall (macro)", f"{rec * 100:.1f}%")
            c4.metric("F1 (macro)", f"{f1 * 100:.1f}%")

            # Predicted vs realized labels over the test portion
            x_axis = np.arange(len(yte))
            sample = min(1500, len(yte))
            sub = np.linspace(0, len(yte) - 1, sample, dtype=int)
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=x_axis[sub], y=yte[sub], mode="markers",
                name="Realized", marker={"color": "#1f77b4", "size": 4},
            ))
            fig_pred.add_trace(go.Scatter(
                x=x_axis[sub], y=preds[sub], mode="markers",
                name=f"{classical_choice}",
                marker={"color": "#ff7f0e", "size": 4, "symbol": "x"},
            ))
            fig_pred.update_layout(height=320, yaxis_title="Class (0=down, 1=stat, 2=up)",
                                     xaxis_title="Test window index")
            st.plotly_chart(fig_pred, width="stretch")

            st.caption(
                f"In-page LDA fit on the first {cap} ticks of the demo slice "
                f"(70/30 train/test split) and evaluated on {len(yte)} test "
                f"windows. Tab 4A reports the Table II numbers computed via "
                f"`scripts/run_backtests.py --lob` on the full FI-2010 split."
            )


with tab3:
    sub3a, sub3b, sub3c = st.tabs(
        ["3A. Architecture", "3B. Live Prediction Demo", "3C. Train your own"]
    )

    with sub3a:
        st.subheader("DeepLOB — CNN + Inception + LSTM")
        st.markdown(
            "DeepLOB threads three conv blocks through a parallel-filter "
            "Inception module into a final LSTM. Each conv block has a "
            "specific purpose:"
        )
        st.code(
            """
input (B, 1, T=100, 40)
                ↓
Conv-Block-1   pairs bid/ask price-volume across each level
                Conv2d(1→16, k=(1,2), s=(1,2))   → (B,16,T,20)
                Conv2d(16→16, k=(4,1)) × 2
                ↓
Conv-Block-2   joins adjacent depth levels
                Conv2d(16→16, k=(1,2), s=(1,2))  → (B,16,T,10)
                Conv2d(16→16, k=(4,1)) × 2
                ↓
Conv-Block-3   collapses the spatial axis to 1
                Conv2d(16→16, k=(1,10))           → (B,16,T,1)
                Conv2d(16→16, k=(4,1)) × 2
                ↓
Inception      three parallel branches
                (1×1 → 3×1) + (1×1 → 5×1) + (MaxPool → 1×1)
                concat along channel dim          → (B,96,T,1)
                ↓
LSTM(96→64, batch_first=True), take last time-step
Linear(64 → 3), Softmax → (B, 3)
""",
            language="text",
        )
        st.caption(
            "~140k parameters total. The Inception module is the "
            "Network-in-Network idea from Szegedy et al. 2015: parallel "
            "filters of different widths capture multi-scale temporal "
            "patterns before the LSTM."
        )

    with sub3b:
        st.subheader("DeepLOB prediction on the demo slice")
        ckpt = _pretrained_dir() / "deeplob_fi2010_k10.pt"
        sidecar = _load_sidecar("deeplob")
        if sidecar:
            fm = sidecar.get("final_metrics", {})
            st.info(
                f"**DeepLOB** — trained `{sidecar.get('trained_on', '?')}` on "
                f"`{sidecar.get('trained_with', '?')}` — "
                f"test F1 **{fm.get('test_f1_macro', float('nan')):.2f}** — "
                f"epochs {sidecar.get('hyperparameters', {}).get('epochs_trained', '?')}"
            )
        else:
            st.warning(
                "**DeepLOB checkpoint missing.** Run "
                "`modal run src/training/train_deeplob.py --arch DeepLOB` "
                "to train it, then `modal volume get dl-research-data "
                "/pretrained/deeplob_fi2010_k10.pt ./data/pretrained/` "
                "to commit."
            )

        if demo_present and ckpt.exists() and n_demo >= _LOOKBACK + 1:
            try:
                cap_3b = min(_IN_PAGE_TICK_CAP, n_demo)
                probs, y_demo = _deeplob_infer_cached(
                    str(ckpt), str(_demo_path()),
                    f"label_k{horizon_k}", cap_3b,
                )
                preds = probs.argmax(axis=1)

                running_acc = float((preds == y_demo).mean())
                st.metric(
                    f"Running accuracy on first {cap_3b}-tick window",
                    f"{running_acc * 100:.1f}%",
                )

                end = min(tick_idx + 1, len(probs) - 1)
                start = max(0, end - 600)
                idx = np.arange(start, end)
                fig_live = go.Figure()
                for cls, color in (
                    (0, "#d62728"), (1, "#888888"), (2, "#2ca02c"),
                ):
                    fig_live.add_trace(go.Scatter(
                        x=idx, y=probs[idx, cls], mode="lines",
                        name=f"P({_CLASS_NAMES[cls]})",
                        line={"color": color},
                    ))
                fig_live.add_trace(go.Scatter(
                    x=idx, y=y_demo[idx], mode="markers",
                    name="Realized class", yaxis="y2",
                    marker={"size": 3, "color": "#000000"},
                ))
                fig_live.update_layout(
                    height=360,
                    yaxis={"title": "Probability", "range": [0, 1]},
                    yaxis2={"title": "Realized class", "overlaying": "y",
                              "side": "right", "range": [-0.5, 2.5]},
                    xaxis_title="Window index (recent 600 around slider)",
                )
                st.plotly_chart(fig_live, width="stretch")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Live prediction failed: {exc}")
        else:
            st.caption(
                "Live prediction requires both the demo slice and the "
                "`deeplob_fi2010_k10.pt` checkpoint."
            )

    with sub3c:
        st.subheader("Train your own DeepLOB checkpoint")
        st.markdown(
            "Training runs on Modal (Constitution Principle III — the live "
            "app does not invoke GPUs). The CPU smoke path is a 1-epoch "
            "synthetic-data test only."
        )
        st.code(
            "# Full training on a Modal T4 (~60 min, ~$0.50)\n"
            "modal run src/training/train_deeplob.py --arch DeepLOB\n\n"
            "# Pull the checkpoint\n"
            "modal volume get dl-research-data "
            "/pretrained/deeplob_fi2010_k10.pt ./data/pretrained/\n"
            "modal volume get dl-research-data "
            "/pretrained/deeplob_fi2010_k10.json ./data/pretrained/\n\n"
            "# CPU smoke (synthetic data, 1 epoch)\n"
            "python -m src.training.train_deeplob --arch DeepLOB --max-epochs 1 -v\n",
            language="bash",
        )


with tab4:
    st.subheader("Table II replication — FI-2010 Setup 2, k=10")
    if panel.empty:
        st.info(
            "Run `python scripts/run_backtests.py --lob` to seed "
            "`data/backtests/lob_results.parquet` (paper-reported rows "
            "will appear immediately; reproduced rows require the .pt "
            "checkpoints)."
        )
    else:
        sub4a, sub4b, sub4c = st.tabs(
            ["4A. Table II", "4B. Confusion Matrices", "4C. Per-class F1"]
        )

        with sub4a:
            view = panel.copy()
            view["Method"] = view["method"].apply(
                lambda m: _ARCH_LABEL.get(m, m.upper())
            )
            view["Source"] = view["source"].map({
                "reproduced_here": "reproduced here",
                "paper_reported": "paper-reported",
            })
            view["Accuracy"] = (view["accuracy"] * 100).round(1)
            view["Precision"] = (view["precision_macro"] * 100).round(1)
            view["Recall"] = (view["recall_macro"] * 100).round(1)
            view["F1"] = (view["f1_macro"] * 100).round(1)
            display = view[["Method", "Source", "Accuracy",
                              "Precision", "Recall", "F1"]]
            display = display.sort_values("F1", ascending=False).reset_index(drop=True)
            st.dataframe(display, hide_index=True, width="stretch")
            st.caption(
                "All values × 100. Macro-averaged Precision / Recall / F1. "
                "Best F1 row is at the top. Paper-reported entries are "
                "from Zhang et al. 2019 Table II (Setup 2, k=10)."
            )

        with sub4b:
            reproduced = panel[panel["source"] == "reproduced_here"].copy()
            if reproduced.empty:
                st.info(
                    "No reproduced confusion matrices yet — train at least "
                    "one of {DeepLOB, MLP, CNN1, CNN2, LSTM} via Modal."
                )
            else:
                top = reproduced.nlargest(3, "f1_macro")
                cols = st.columns(min(3, len(top)))
                for col, (_, row) in zip(cols, top.iterrows()):
                    with col:
                        cm = np.array([
                            [row[f"cm_{i}{j}"] for j in range(3)]
                            for i in range(3)
                        ])
                        fig_cm = px.imshow(
                            cm, text_auto=True, aspect="auto",
                            x=list(_CLASS_NAMES), y=list(_CLASS_NAMES),
                            color_continuous_scale="Blues",
                            labels={"x": "predicted", "y": "actual"},
                        )
                        fig_cm.update_layout(
                            title=f"{_ARCH_LABEL.get(row['method'], row['method'])}"
                                    f" (F1={row['f1_macro']:.2f})",
                            height=320,
                        )
                        st.plotly_chart(fig_cm, width="stretch")

        with sub4c:
            reproduced = panel[panel["source"] == "reproduced_here"].copy()
            if reproduced.empty:
                st.info("No per-class F1 rows yet (no .pt checkpoints).")
            else:
                rows = []
                for _, row in reproduced.iterrows():
                    cm = np.array([
                        [row[f"cm_{i}{j}"] for j in range(3)]
                        for i in range(3)
                    ], dtype=np.float64)
                    for cls in (0, 1, 2):
                        tp = cm[cls, cls]
                        fp = cm[:, cls].sum() - tp
                        fn = cm[cls, :].sum() - tp
                        denom = 2 * tp + fp + fn
                        f1_cls = (2 * tp / denom) if denom > 0 else 0.0
                        rows.append({
                            "Method": _ARCH_LABEL.get(row["method"], row["method"]),
                            "Class": _CLASS_NAMES[cls],
                            "F1": round(f1_cls * 100, 1),
                        })
                f1_df = pd.DataFrame(rows)
                fig_perc = px.bar(
                    f1_df, x="Method", y="F1", color="Class", barmode="group",
                    color_discrete_map={"down": "#d62728", "stationary": "#888888",
                                          "up": "#2ca02c"},
                    height=360,
                )
                fig_perc.update_layout(yaxis_title="Per-class F1 (×100)")
                st.plotly_chart(fig_perc, width="stretch")


with st.expander("📐 Math", expanded=False):
    st.markdown("**Mid-price** (paper Eq. 1)")
    st.latex(r"p_t = \tfrac{1}{2}\bigl(p_t^{\text{ask},1} + p_t^{\text{bid},1}\bigr)")

    st.markdown("**Smoothed-label rule** (paper Eq. 4)")
    st.latex(
        r"m_-(t) = \tfrac{1}{k}\sum_{i=0}^{k-1} p_{t-i},\quad"
        r"m_+(t) = \tfrac{1}{k}\sum_{i=1}^{k} p_{t+i}"
    )
    st.latex(
        r"\ell_t = \frac{m_+(t) - m_-(t)}{m_-(t)},\quad"
        r"y_t = \begin{cases} 0 & \ell_t < -\alpha \\ "
        r"2 & \ell_t > \alpha \\ 1 & \text{otherwise} \end{cases}"
    )
    st.caption(r"$\alpha$ is the FI-2010 paper's classification threshold; "
                "labels at k ∈ {10, 20, 30, 50, 100} are pre-computed in the dataset.")

    st.markdown("**Gated activation** in DeepLOB's first conv block")
    st.latex(
        r"\sigma(x) = \text{LeakyReLU}_{\alpha=0.01}(x),\quad"
        r"\text{the }(1,2)\text{-stride pair-wise conv produces "
        r"micro-price / volume-imbalance features.}"
    )

    st.markdown("**Inception module** — parallel filters")
    st.latex(
        r"x_{\text{out}} = \text{concat}\Bigl["
        r"\,\text{conv}_{3\times1}(\text{conv}_{1\times1}(x)),\;"
        r"\text{conv}_{5\times1}(\text{conv}_{1\times1}(x)),\;"
        r"\text{conv}_{1\times1}(\text{maxpool}(x))\Bigr]"
    )
    st.caption("Three branches over the same input → channel-concat → "
                 "captures multi-scale temporal patterns before the LSTM.")


with st.expander("💻 Code", expanded=False):
    st.markdown("**`src.models.deeplob.DeepLOB`**")
    st.code(inspect.getsource(DeepLOB), language="python")
    st.markdown("**`src.models.deeplob.LOBSimpleMLP`**")
    st.code(inspect.getsource(LOBSimpleMLP), language="python")


st.divider()
st.caption(
    "📄 [arXiv:1808.03668](https://arxiv.org/abs/1808.03668) · "
    "💻 [pages/3_📖_Order_Book.py on GitHub]"
    "(https://github.com/jjj1231978/dl-research-demo/blob/main/pages/3_%F0%9F%93%96_Order_Book.py) · "
    "Constitution v1.1.0."
)
st.code(
    """@article{zhang2019deeplob,
  title={DeepLOB: Deep convolutional neural networks for limit order books},
  author={Zhang, Zihao and Zohren, Stefan and Roberts, Stephen},
  journal={IEEE Transactions on Signal Processing},
  year={2019}
}""",
    language="bibtex",
)
