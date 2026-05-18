#!/usr/bin/env python3
"""Download FI-2010 from Kaggle, parse into parquet.

Contract: specs/004-phase-3-deeplob/contracts/fetch_lob_fi2010_cli.md

Source: Kaggle `praanj/limit-orderbook-data` — redistribution of
Ntakaris et al. 2017 FI-2010 benchmark. 4 files (~940 MB raw):
  Train_Dst_NoAuction_DecPre_CF_7.txt   — 7 train days, all 5 stocks
  Test_Dst_NoAuction_DecPre_CF_7.txt    — test day 8 (1-indexed; "CF_7"
                                          is the file index in the
                                          standard release)
  Test_Dst_NoAuction_DecPre_CF_8.txt    — test day 9
  Test_Dst_NoAuction_DecPre_CF_9.txt    — test day 10

Each file: 149 rows × N columns (sample-per-column).
- Rows 1-40: 40 LOB features (z-score normalized, DecPre / NoAuction / CF)
- Rows 41-144: hand-crafted features (NOT used — DeepLOB uses raw LOB only)
- Rows 145-149: 5 labels at k ∈ {10, 20, 30, 50, 100} ticks, values 1/2/3
  → mapped to 0/1/2 = (down, stationary, up)

Output:
  ${DEEP_FINANCE_DATA_DIR}/lob_fi2010.parquet   — full benchmark
  data/lob_fi2010_demo.parquet                  — 100k-tick slice of
                                                  test day 8 for HF Space
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path

# Repo root on sys.path so `from src.* import ...` works
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("fetch_lob_fi2010")

EXIT_OK = 0
EXIT_AUTH = 1
EXIT_DOWNLOAD = 2
EXIT_PARSE = 3
EXIT_WRITE = 4

KAGGLE_DATASET = "praanj/limit-orderbook-data"
TRAIN_FILE = "Train_Dst_NoAuction_DecPre_CF_7.txt"
TEST_FILES = (
    ("Test_Dst_NoAuction_DecPre_CF_7.txt", 8),
    ("Test_Dst_NoAuction_DecPre_CF_8.txt", 9),
    ("Test_Dst_NoAuction_DecPre_CF_9.txt", 10),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_lob_fi2010",
        description="Kaggle FI-2010 download + parquet build.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output dir for lob_fi2010.parquet. Default: data_root().",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path.home() / "data_lake" / "fi2010",
        help="Where to download/cache the 4 Kaggle .txt files.",
    )
    parser.add_argument(
        "--demo-only", action="store_true",
        help="Skip the full parquet; only build data/lob_fi2010_demo.parquet.",
    )
    parser.add_argument(
        "--no-redownload", action="store_true",
        help="Skip the Kaggle download (use cached files in --raw-dir).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _kaggle_download(raw_dir: Path) -> int:
    """Download via the Kaggle Python API. Returns exit code."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    needed = [TRAIN_FILE] + [f for f, _ in TEST_FILES]
    if all((raw_dir / f).exists() for f in needed):
        log.info("All 4 raw files already present in %s — skipping download.",
                 raw_dir)
        return EXIT_OK

    try:
        import kaggle  # type: ignore
        api = kaggle.api
        api.authenticate()
    except Exception as exc:  # noqa: BLE001
        log.error("Kaggle auth failed (%s). Check ~/.kaggle/kaggle.json.", exc)
        return EXIT_AUTH

    log.info("Kaggle: downloading %s → %s (unzipping)", KAGGLE_DATASET, raw_dir)
    try:
        api.dataset_download_files(KAGGLE_DATASET, path=str(raw_dir), unzip=True, quiet=False)
    except Exception as exc:  # noqa: BLE001
        log.error("Kaggle download failed: %s", exc)
        return EXIT_DOWNLOAD
    return EXIT_OK


def _parse_fi2010_file(path: Path):
    """Parse one of the 4 FI-2010 .txt files into (features, labels).

    Returns:
        features: (n_ticks, 40) float32
        labels:   (n_ticks, 5)  int8 — k=10,20,30,50,100 horizon labels in {0,1,2}
    """
    import numpy as np

    log.info("Parsing %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    arr = np.loadtxt(path, dtype=np.float32)
    # File layout: 149 rows × N samples (samples-per-column)
    if arr.shape[0] != 149:
        raise ValueError(
            f"{path.name}: expected 149 rows, got {arr.shape[0]} "
            f"(does this match the FI-2010 NoAuction_DecPre_CF format?)"
        )

    features = arr[:40, :].T  # (N, 40) — z-score-normalized LOB
    labels = arr[144:149, :].T.astype(np.int8) - 1  # (N, 5) — {1,2,3} → {0,1,2}
    if labels.min() < 0 or labels.max() > 2:
        raise ValueError(
            f"{path.name}: labels out of expected {{0,1,2}} range — "
            f"min={labels.min()}, max={labels.max()}"
        )
    return features, labels


def _build_long_format(features, labels, day: int, split: str):
    """Convert (features, labels) arrays to a long-format DataFrame row block."""
    import numpy as np
    import pandas as pd

    n = features.shape[0]
    cols = {
        "split": np.full(n, split, dtype=object),
        "day": np.full(n, day, dtype=np.int8),
        "tick": np.arange(n, dtype=np.int32),
    }
    # 40 LOB features: alternating (price, volume) per level, bid 1..10 then ask 1..10
    # Standard FI-2010 layout: cols 0..3 = (best ask price, vol, best bid price, vol),
    # then deeper levels. We name them generically as f00..f39 to match the file's
    # native order; the page interprets them per the FI-2010 spec.
    for i in range(40):
        cols[f"f{i:02d}"] = features[:, i].astype(np.float32)
    for j, k in enumerate((10, 20, 30, 50, 100)):
        cols[f"label_k{k}"] = labels[:, j]
    return pd.DataFrame(cols)


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

    import pandas as pd

    from src.data import data_root

    raw_dir: Path = args.raw_dir.expanduser().resolve()
    out_dir: Path = (args.out_dir or data_root()).expanduser().resolve()

    # 1. Kaggle download
    if not args.no_redownload:
        rc = _kaggle_download(raw_dir)
        if rc != 0:
            return rc

    # 2. Parse each file → DataFrame blocks
    blocks: list = []
    try:
        if not args.demo_only:
            train_path = raw_dir / TRAIN_FILE
            if not train_path.exists():
                log.error("Missing train file: %s", train_path)
                return EXIT_PARSE
            tr_feat, tr_lab = _parse_fi2010_file(train_path)
            blocks.append(_build_long_format(tr_feat, tr_lab, day=0, split="train"))

        for fname, day in TEST_FILES:
            test_path = raw_dir / fname
            if not test_path.exists():
                log.error("Missing test file: %s", test_path)
                return EXIT_PARSE
            te_feat, te_lab = _parse_fi2010_file(test_path)
            blocks.append(_build_long_format(te_feat, te_lab, day=day, split="test"))
    except (ValueError, OSError) as exc:
        log.error("parse error: %s", exc)
        return EXIT_PARSE

    panel = pd.concat(blocks, ignore_index=True)
    log.info("Combined panel: %d rows × %d cols", *panel.shape)

    # 3. Demo slice: first 100k ticks of test day 8
    demo = panel[(panel["split"] == "test") & (panel["day"] == 8)].iloc[:100_000].copy()
    log.info("Demo slice: %d rows", len(demo))

    try:
        # 4. Atomic-write outputs
        if not args.demo_only:
            target = out_dir / "lob_fi2010.parquet"
            _atomic_write_parquet(panel, target)
            log.info("Wrote %s (%.1f MB)", target, target.stat().st_size / 1e6)

            sidecar = {
                "dataset": "FI-2010",
                "kaggle_source": KAGGLE_DATASET,
                "fetched_at": _dt.datetime.now(_dt.timezone.utc)
                              .isoformat(timespec="seconds").replace("+00:00", "Z"),
                "row_count": int(len(panel)),
                "days": sorted(int(d) for d in panel["day"].unique()),
                "horizons": [10, 20, 30, 50, 100],
            }
            target.with_suffix(target.suffix + ".json").write_text(
                json.dumps(sidecar, indent=2)
            )

        # Demo slice always written (it's the committed-to-git artifact)
        demo_target = _REPO_ROOT / "data" / "lob_fi2010_demo.parquet"
        _atomic_write_parquet(demo, demo_target)
        log.info("Wrote %s (%.1f MB)", demo_target, demo_target.stat().st_size / 1e6)
    except OSError as exc:
        log.error("write error: %s", exc)
        return EXIT_WRITE

    if args.demo_only:
        print(f"[fetch_lob_fi2010] demo slice: {len(demo)} rows → {demo_target}")
    else:
        print(
            f"[fetch_lob_fi2010] full panel: {len(panel)} rows "
            f"(train: {(panel['split']=='train').sum()}, "
            f"test: {(panel['split']=='test').sum()}); "
            f"demo slice: {len(demo)} rows"
        )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
