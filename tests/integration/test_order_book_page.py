"""Integration tests for the Phase 3 Order Book page (FR-022).

Four parametrised cases per `contracts/order_book_page_ui.md` §"Render
assertions": (demo_slice × checkpoints) = {absent, present}^2, plus the
two copy-text invariants (setup-instruction banner + substrate disclosure).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from streamlit.testing.v1 import AppTest

from src.models.deeplob import (
    DeepLOB,
    LOBCNN_I,
    LOBCNN_II,
    LOBLSTM,
    LOBSimpleMLP,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAGE_FILE = str(REPO_ROOT / "pages" / "3_📖_Order_Book.py")

_LOOKBACK = 100
_FEATURE_COLS = [f"f{i:02d}" for i in range(40)]

_ARCH_TO_CLS = {
    "deeplob": DeepLOB,
    "mlp": LOBSimpleMLP,
    "cnn1": LOBCNN_I,
    "cnn2": LOBCNN_II,
    "lstm": LOBLSTM,
}


def _make_demo_parquet(path: Path) -> None:
    """Tiny synthetic demo slice: ~250 ticks of f00..f39 + labels."""
    rng = np.random.default_rng(7)
    n = 250
    rows = []
    for tick in range(n):
        row = {"split": "test", "day": 8, "tick": tick}
        for i in range(40):
            row[f"f{i:02d}"] = float(rng.standard_normal())
        for k in (10, 20, 30, 50, 100):
            row[f"label_k{k}"] = int(rng.integers(0, 3))
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def _make_backtest_parquet(path: Path, reproduced: bool = False) -> None:
    """A small lob_results.parquet — always 5 paper-reported, optionally
    add 5 reproduced (with confusion-matrix cells) so Tab 4A has 6+ rows."""
    rows = []
    paper = (
        ("svm",     0.486, 0.491, 0.486, 0.487),
        ("bof",     0.572, 0.490, 0.460, 0.460),
        ("mcsda",   0.737, 0.460, 0.479, 0.467),
        ("b(tabl)", 0.788, 0.789, 0.788, 0.785),
        ("c(tabl)", 0.842, 0.851, 0.842, 0.844),
    )
    for m, a, p, r, f in paper:
        row = {
            "method": m, "k": 10, "accuracy": a, "precision_macro": p,
            "recall_macro": r, "f1_macro": f, "source": "paper_reported",
        }
        for i in range(3):
            for j in range(3):
                row[f"cm_{i}{j}"] = -1
        rows.append(row)
    if reproduced:
        # Fake reproduced rows for 5 methods + LDA
        for m, f1 in (("deeplob", 0.78), ("lstm", 0.70), ("cnn2", 0.65),
                       ("cnn1", 0.62), ("mlp", 0.58), ("lda", 0.46)):
            row = {
                "method": m, "k": 10, "accuracy": f1 + 0.05,
                "precision_macro": f1 + 0.02, "recall_macro": f1 + 0.01,
                "f1_macro": f1, "source": "reproduced_here",
            }
            for i in range(3):
                for j in range(3):
                    row[f"cm_{i}{j}"] = 100 if i == j else 10
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow")


def _make_fake_checkpoint(arch_lower: str, path: Path, sidecar_path: Path) -> None:
    """Write a fake but valid checkpoint + sidecar so the page can load it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model = _ARCH_TO_CLS[arch_lower]()
    torch.save(model.state_dict(), path)
    sidecar = {
        "trained_on": "2026-05-17",
        "trained_with": "Modal T4 (image im-test)",
        "torch_version": torch.__version__,
        "modal_app": "deep-finance-train-deeplob",
        "arch": arch_lower.upper() if arch_lower != "deeplob" else "DeepLOB",
        "dataset": "FI-2010",
        "setup": 2,
        "k": 10,
        "n_classes": 3,
        "hyperparameters": {"lookback": 100, "batch_size": 64, "lr": 1e-3,
                              "epochs_trained": 1, "patience": 10, "min_delta": 1e-3},
        "final_metrics": {
            "val_loss": 0.94, "test_accuracy": 0.78,
            "test_precision_macro": 0.78, "test_recall_macro": 0.78,
            "test_f1_macro": 0.78, "confusion_matrix": [[100, 10, 10],
                                                          [10, 100, 10],
                                                          [10, 10, 100]],
        },
        "git_commit": "test",
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))


@pytest.fixture
def page_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
              request: pytest.FixtureRequest) -> dict:
    """Parametrized by (demo_present, checkpoints_present).

    Sets DEEP_FINANCE_LOB_DEMO_PARQUET + DEEP_FINANCE_BACKTESTS_DIR +
    DEEP_FINANCE_PRETRAINED_DIR to tmp_path subdirs, optionally seeds them.
    """
    demo_present, ckpts_present = request.param

    demo_path = tmp_path / "lob_fi2010_demo.parquet"
    backtests_dir = tmp_path / "backtests"
    pretrained_dir = tmp_path / "pretrained"
    backtests_dir.mkdir()
    pretrained_dir.mkdir()

    if demo_present:
        _make_demo_parquet(demo_path)
    # Always seed the lob_results panel (paper rows show even without
    # reproduced ones). When checkpoints are present, also seed reproduced rows.
    _make_backtest_parquet(backtests_dir / "lob_results.parquet",
                             reproduced=ckpts_present)

    if ckpts_present:
        for arch_lower in _ARCH_TO_CLS:
            _make_fake_checkpoint(
                arch_lower,
                pretrained_dir / f"{arch_lower}_fi2010_k10.pt",
                pretrained_dir / f"{arch_lower}_fi2010_k10.json",
            )

    monkeypatch.setenv("DEEP_FINANCE_LOB_DEMO_PARQUET", str(demo_path))
    monkeypatch.setenv("DEEP_FINANCE_BACKTESTS_DIR", str(backtests_dir))
    monkeypatch.setenv("DEEP_FINANCE_PRETRAINED_DIR", str(pretrained_dir))
    return {"demo_present": demo_present, "ckpts_present": ckpts_present,
              "tmp_path": tmp_path}


def _run_app() -> AppTest:
    at = AppTest.from_file(PAGE_FILE, default_timeout=60)
    at.run()
    return at


def _all_markdown(at: AppTest) -> str:
    parts = [md.value for md in at.main.markdown]
    for tab in getattr(at, "tabs", []) or []:
        parts.extend(md.value for md in tab.markdown)
        for inner in getattr(tab, "tabs", []) or []:
            parts.extend(md.value for md in inner.markdown)
    return "\n".join(parts)


def _all_warnings(at: AppTest) -> str:
    parts = [w.value for w in at.warning]
    for tab in getattr(at, "tabs", []) or []:
        parts.extend(w.value for w in tab.warning)
        for inner in getattr(tab, "tabs", []) or []:
            parts.extend(w.value for w in inner.warning)
    return "\n".join(parts)


def _all_dataframes(at: AppTest):
    out = list(at.dataframe)
    for tab in getattr(at, "tabs", []) or []:
        out.extend(tab.dataframe)
        for inner in getattr(tab, "tabs", []) or []:
            out.extend(inner.dataframe)
    return out


_MATRIX = [
    pytest.param((False, False), id="A_demo_absent_ckpts_absent"),
    pytest.param((False, True),  id="B_demo_absent_ckpts_present"),
    pytest.param((True,  False), id="C_demo_present_ckpts_absent"),
    pytest.param((True,  True),  id="D_demo_present_ckpts_present"),
]


@pytest.mark.parametrize("page_env", _MATRIX, indirect=True)
def test_order_book_page_renders(page_env):
    """All 4 cases: AppTest.run() raises no exception (FR-022)."""
    at = _run_app()
    assert not at.exception, (
        f"Order Book page raised in case "
        f"(demo={page_env['demo_present']}, "
        f"ckpts={page_env['ckpts_present']}): "
        f"{[str(e) for e in at.exception]}"
    )


@pytest.mark.parametrize(
    "page_env",
    [pytest.param((False, False), id="demo_absent")],
    indirect=True,
)
def test_setup_instruction_banner(page_env):
    """When demo is absent, the setup banner mentions both the parquet
    filename and the fetch script."""
    at = _run_app()
    assert not at.exception, str([str(e) for e in at.exception])
    haystack = _all_markdown(at) + "\n" + _all_warnings(at)
    for needle in ("lob_fi2010.parquet", "scripts/fetch_lob_fi2010.py"):
        assert needle in haystack, (
            f"Setup-instruction banner missing {needle!r}.\n"
            f"Markdown+Warnings:\n{haystack[:1500]}"
        )


@pytest.mark.parametrize(
    "page_env",
    [pytest.param((True, True), id="full")],
    indirect=True,
)
def test_substrate_disclosure_visible(page_env):
    """Tab 1 contains the substrate-disclosure copy-text invariants."""
    at = _run_app()
    assert not at.exception, str([str(e) for e in at.exception])
    md = _all_markdown(at)
    for needle in ("FI-2010", "Ntakaris et al."):
        assert needle in md, (
            f"Substrate disclosure missing {needle!r}.\n"
            f"Markdown (truncated):\n{md[:1500]}"
        )


@pytest.mark.parametrize(
    "page_env",
    [pytest.param((True, True), id="full")],
    indirect=True,
)
def test_table_ii_has_six_plus_rows(page_env):
    """Tab 4A renders a dataframe with ≥ 6 rows when both demo + ckpts present."""
    at = _run_app()
    assert not at.exception, str([str(e) for e in at.exception])
    dfs = _all_dataframes(at)
    expected_cols = {"Method", "Source", "Accuracy", "Precision",
                      "Recall", "F1"}
    table = next(
        (d for d in dfs
         if d.value is not None and expected_cols.issubset(set(d.value.columns))),
        None,
    )
    assert table is not None, (
        f"No Table II dataframe with cols {expected_cols} found. "
        f"Rendered df cols: "
        f"{[list(d.value.columns) if d.value is not None else None for d in dfs]}"
    )
    assert len(table.value) >= 6, (
        f"Table II has {len(table.value)} rows; expected ≥ 6 "
        f"(5 paper-reported + 1+ reproduced)."
    )


@pytest.mark.parametrize(
    "page_env",
    [pytest.param((True, False), id="demo_present_ckpts_absent")],
    indirect=True,
)
def test_missing_checkpoint_warning_visible(page_env):
    """Case C: when demo is present but DeepLOB checkpoint absent,
    Tab 3B shows a 'missing checkpoint' warning."""
    at = _run_app()
    assert not at.exception, str([str(e) for e in at.exception])
    warns = _all_warnings(at)
    assert "DeepLOB checkpoint missing" in warns, (
        f"Tab 3B missing-checkpoint warning not found.\n"
        f"Warnings (truncated):\n{warns[:1500]}"
    )
