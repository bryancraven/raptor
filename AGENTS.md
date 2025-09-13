# Repository Guidelines

## Project Structure & Module Organization
- Source: `raptor_model_advanced.py` (single-file Python CLI + model).
- Outputs: plots (`raptor_prob_dist.png`, `raptor_contributions.png`) and optional reports via `--report_json`, `--report_csv`.
- Tests: none yet. If you add modules, create `tests/` with `test_*.py` and keep all code under `./` or a new `raptor/` package as the code grows.

## Build, Test, and Development Commands
- Create venv: 
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  ```
- Install deps:
  ```bash
  pip install numpy matplotlib
  ```
  Or use repo pins:
  ```bash
  pip install -r requirements.txt          # runtime
  pip install -r requirements-dev.txt      # + tests, lint, formatters
  ```
- CLI help / run:
  ```bash
  python3 raptor_model_advanced.py --help
  python3 raptor_model_advanced.py --month 8 --time_of_day dawn --plot \
    --report_json out.json --report_csv species.csv
  ```
- Smoke test (deterministic):
  ```bash
  python3 raptor_model_advanced.py --seed 123 --n_sims 5000
  ```

## Coding Style & Naming Conventions
- Python 3.9+, UTF-8, 4-space indentation.
- Naming: `snake_case` (functions/vars), `CAPS` (constants), `CamelCase` (classes).
- Use type hints and concise docstrings for public functions.
- CLI: group related flags and validate ranges; keep `--help` informative.
- Formatting (recommended):
  ```bash
  black -l 88 .
  ruff .
  ```

## Testing Guidelines
- Prefer deterministic checks with `--seed` and smaller `--n_sims`.
- If adding modules, use `pytest`; place tests in `tests/` as `test_*.py`.
- Run tests (if present):
  ```bash
  pytest -q
  ```
 - Target a subset:
  ```bash
  pytest -q tests/test_factors.py
  ```

## Continuous Integration
- GitHub Actions workflow runs `pytest` on Python 3.9–3.12 (`.github/workflows/ci.yml`).

## Commit & Pull Request Guidelines
- Commits: use Conventional Commits where practical (e.g., `feat: add sensitivity analysis`, `fix: clamp precipitation bounds`).
- PRs must include: clear description, example command(s) used, observed output summary, and screenshots of plots if applicable. Link related issues and note any CLI flag changes.

## Security & Configuration Tips
- No secrets or network keys required; avoid introducing network calls.
- Default outputs to local paths; document any new files.
- If dependencies grow, add `requirements.txt` or `pyproject.toml` and pin versions for reproducibility.

## Architecture Overview
- Hazard-based Monte Carlo model that combines species priors with time, weather, habitat, season, party size, and day-quality multipliers. The CLI orchestrates inputs, simulation, summaries, and optional plots/reports.
