"""Regression test for FR-022 — DEEP_FINANCE_DATA_DIR env-var resolution.

Verifies:
- data_root() defaults to ./data/ when env var unset
- data_root() honours DEEP_FINANCE_DATA_DIR when set (with ~ expansion)
- BUNDLED_CSV_DIR is constant — does NOT change when DEEP_FINANCE_DATA_DIR
  is set (FR-022 — CSV fallbacks always resolve relative to repo root)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data import BUNDLED_CSV_DIR, data_root


def test_data_root_default_when_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DEEP_FINANCE_DATA_DIR", raising=False)
    result = data_root()
    assert result == Path("data").resolve(), (
        f"Default data_root expected to resolve to ./data/, got {result}"
    )


def test_data_root_honours_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("DEEP_FINANCE_DATA_DIR", str(tmp_path))
    assert data_root() == tmp_path.resolve()


def test_data_root_expands_tilde(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEP_FINANCE_DATA_DIR", "~/test-deep-finance-data")
    result = data_root()
    assert "~" not in str(result), f"~ should be expanded, got {result}"
    assert result == Path.home() / "test-deep-finance-data"


def test_bundled_csv_dir_does_not_honour_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """FR-022: BUNDLED_CSV_DIR is fixed at the repo root — overriding
    DEEP_FINANCE_DATA_DIR MUST NOT redirect CSV-fallback resolution
    (otherwise pointing at a directory without CSVs breaks first-clone
    usability)."""
    from src import data as data_module
    original_bundled = BUNDLED_CSV_DIR
    monkeypatch.setenv("DEEP_FINANCE_DATA_DIR", str(tmp_path))
    # BUNDLED_CSV_DIR is a module-level constant set at import time. Setting
    # the env var must NOT mutate it.
    assert data_module.BUNDLED_CSV_DIR == original_bundled, (
        f"BUNDLED_CSV_DIR changed under env override: was {original_bundled}, "
        f"now {data_module.BUNDLED_CSV_DIR}"
    )
    # Verify the original bundled CSVs are still reachable from BUNDLED_CSV_DIR
    assert (data_module.BUNDLED_CSV_DIR / "aapl.csv").exists()
    assert (data_module.BUNDLED_CSV_DIR / "portfolio_data.csv").exists()
