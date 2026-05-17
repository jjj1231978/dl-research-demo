"""Modal-hosted training scripts for the Deep Finance Showcase.

Each module follows the canonical Modal-app pattern (constitution v1.1.0
§"Training workflow (Modal)"): top of file declares `App` / `Image` /
`Volume` / `@app.function(gpu=...)` + `@app.local_entrypoint()`; bottom
declares a device-agnostic `def train(...)` body that runs unchanged on
CPU (local smoke) or GPU (Modal container).

Phase 1 modules:
    train_deep_momentum — MLP and LSTM trainer (FR-009/010/011/012)
"""
