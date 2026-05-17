<!-- SPECKIT START -->
**Active feature**: `002-phase-1-momentum` (Phase 1 — Momentum Page, Lim et al. 2019)

**Plan**: [specs/002-phase-1-momentum/plan.md](specs/002-phase-1-momentum/plan.md)

**Spec**: [specs/002-phase-1-momentum/spec.md](specs/002-phase-1-momentum/spec.md)

**Phase 0 reference** (merged into this branch's history): [specs/001-phase-0-skeleton-data/plan.md](specs/001-phase-0-skeleton-data/plan.md)

**Project brief** (multi-phase roadmap): [Project_brief.md](Project_brief.md)

**Constitution**: [.specify/memory/constitution.md](.specify/memory/constitution.md) v1.1.0 — six principles. The non-negotiable ones for Phase 0:
- II. Device-Agnostic Torch — no `.cuda()` calls in `src/`. **GPU host is Modal serverless containers** (changed from Colab Pro in v1.1.0); training scripts in Phases 1–3 live under `src/training/` with `@app.local_entrypoint()` decorators, no Colab notebooks.
- V. Pre-existing Canon — `src/losses.py`, `src/early_stopper.py`, `src/torch_data.py` are user-authored and copied verbatim from `Utilis/`; `src/metrics.py` is extended per brief §13.7 but the three original keys (`annual_ret`, `annual_std`, `annual_sharpe`) MUST stay byte-identical against a fixed-seed series.

**Note**: `Project_brief.md` (referenced above) was written assuming Colab Pro. The constitution at v1.1.0 supersedes that choice — when the brief mentions "Colab", read it as "Modal" until the brief is reconciled in a later edit.

When taking on Phase 0 work, read `specs/001-phase-0-skeleton-data/plan.md` first, then `quickstart.md` for the operator walkthrough, then the four files in `contracts/` for module-level interface contracts.
<!-- SPECKIT END -->
