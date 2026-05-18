<!-- SPECKIT START -->
**Active feature**: `004-phase-3-deeplob` (Phase 3 — Order Book / DeepLOB, Zhang et al. 2019)

**Spec**: [specs/004-phase-3-deeplob/spec.md](specs/004-phase-3-deeplob/spec.md) · [plan.md](specs/004-phase-3-deeplob/plan.md) · [tasks.md](specs/004-phase-3-deeplob/tasks.md)

**Phase 2 reference** (live on HF Space, merged into main): [specs/003-phase-2-portfolio/spec.md](specs/003-phase-2-portfolio/spec.md), [tasks.md](specs/003-phase-2-portfolio/tasks.md)

**Phase 1 reference** (merged into main): [specs/002-phase-1-momentum/spec.md](specs/002-phase-1-momentum/spec.md), [tasks.md](specs/002-phase-1-momentum/tasks.md)

**Phase 0 reference** (merged into main): [specs/001-phase-0-skeleton-data/plan.md](specs/001-phase-0-skeleton-data/plan.md)

**Project brief** (multi-phase roadmap): [Project_brief.md](Project_brief.md)

**Constitution**: [.specify/memory/constitution.md](.specify/memory/constitution.md) v1.1.0 — six principles. The non-negotiable ones for Phase 0:
- II. Device-Agnostic Torch — no `.cuda()` calls in `src/`. **GPU host is Modal serverless containers** (changed from Colab Pro in v1.1.0); training scripts in Phases 1–3 live under `src/training/` with `@app.local_entrypoint()` decorators, no Colab notebooks.
- V. Pre-existing Canon — `src/losses.py`, `src/early_stopper.py`, `src/torch_data.py` are user-authored and copied verbatim from `Utilis/`; `src/metrics.py` is extended per brief §13.7 but the three original keys (`annual_ret`, `annual_std`, `annual_sharpe`) MUST stay byte-identical against a fixed-seed series.

**Note**: `Project_brief.md` (referenced above) was written assuming Colab Pro. The constitution at v1.1.0 supersedes that choice — when the brief mentions "Colab", read it as "Modal" until the brief is reconciled in a later edit.

When taking on Phase 3 work, read `specs/004-phase-3-deeplob/plan.md` first, then `quickstart.md` for the operator walkthrough, then the four contract files for module-level interface contracts (`lob_models.md`, `train_deeplob_cli.md`, `fetch_lob_fi2010_cli.md`, `order_book_page_ui.md`). Phase 3 introduces the FI-2010 benchmark + the Kaggle `praanj/limit-orderbook-data` fetch path; the full parquet is dev-machine + Modal Volume only — only `data/lob_fi2010_demo.parquet` (~2 MB) ships in git.
<!-- SPECKIT END -->
