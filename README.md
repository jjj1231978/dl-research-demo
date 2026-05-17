# Deep Finance Showcase

Interactive Streamlit application showcasing three Oxford-Man Institute papers
applying deep learning to canonical quantitative-finance problems
(time-series momentum, portfolio optimization, limit order books).

> **Phase 0 minimal README** — full README with hero screenshot, live-demo
> badge, citation block, and deployment instructions lands in Phase 4. For
> now, see:
>
> - [`Project_brief.md`](Project_brief.md) — multi-phase roadmap (the source-of-truth design doc)
> - [`specs/`](specs/) — per-phase specifications, plans, and tasks
> - [`.specify/memory/constitution.md`](.specify/memory/constitution.md) — non-negotiable principles

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at <http://localhost:7860>. Works on a fresh clone with no API key
(falls back to bundled CSVs); see the Data status sidebar.

## License

MIT — see [`LICENSE`](LICENSE).
