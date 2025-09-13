# Contributing

Thanks for your interest in improving the Raptor Sighting Model! Please follow these guidelines to keep contributions smooth and consistent.

## Getting Started
- Create a virtual env and install dev deps:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements-dev.txt`
- Run tests: `pytest -q` (or target a subset: `pytest -q tests/test_factors.py`).
- Try the CLI: `python3 raptor_model_advanced.py --help`.

## Coding Standards
- Python 3.9+, 4-space indentation, UTF-8.
- Naming: `snake_case` (functions/vars), `CAPS` (constants), `CamelCase` (classes).
- Type hints for public functions; concise docstrings.
- Keep CLI help clear and validate input ranges.
- Format and lint before PRs: `black -l 88 .` and `ruff .`.

## Commits & Pull Requests
- Use Conventional Commits where practical (e.g., `feat: add sensitivity analysis`, `fix: clamp precipitation bounds`).
- PR checklist:
  - Description of changes and rationale.
  - Example CLI command(s) used and short output summary.
  - Screenshots for plots when applicable.
  - Note any CLI flag changes and link related issues.

## Tests
- Prefer deterministic checks using `--seed` and modest `--n_sims`.
- Add tests under `tests/` as `test_*.py` to cover new logic (parsing, factor math, I/O).

## Where to Look
- Core file: `raptor_model_advanced.py` (model + CLI).
- Contributor guide: `AGENTS.md`.
- Architecture and math overview: `README.md`.

## Security
- No secrets or network keys required. Avoid introducing network calls.
- Default outputs to local paths; document any new files.

