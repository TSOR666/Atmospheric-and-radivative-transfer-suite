# Contributing Guide

Thanks for your interest in contributing! This guide covers local setup, coding style, tests, and CI behavior so you can get productive quickly.

## Environments

This repo contains two main components. We recommend separate virtual environments:

- STRATUS: radiative-transfer toolkit (package in `STRATUS/`)
- MDNO/RTNO: coupled dynamics + radiative transfer (package in `MDNO/mdno_rtno/`)

### STRATUS quickstart

- Create env and install dev extras:
  - `cd STRATUS`
  - `python -m venv .venv && . .venv/Scripts/Activate.ps1` (PowerShell) or `. .venv/bin/activate` (bash)
  - `pip install -e .[dev]`
- Run tests and tooling:
  - `pytest -q`
  - `ruff check stratus`
  - `black --check stratus`
  - `mypy stratus`

### MDNO/RTNO quickstart

- Create env and install requirements:
  - `cd MDNO/mdno_rtno`
  - `python -m venv .venv && . .venv/Scripts/Activate.ps1` (PowerShell) or `. .venv/bin/activate`
  - `pip install -r requirements.txt`
- Sanity checks:
  - Syntax-only: `python -m compileall .`
  - Tests (optional, may require heavier deps): `pytest -q`

## Code Style

- Lint: Ruff (config in `STRATUS/pyproject.toml`) - `ruff check stratus`
- Format: Black, line length 100 - `black --check stratus`
- Types: mypy with loose settings - `mypy stratus`
- Keep changes minimal, focused, and consistent with the local style. Prefer clarity over cleverness.

## Tests

- STRATUS has unit/integration tests under `STRATUS/tests`. Aim to keep tests fast and CPU-friendly.
- MDNO/RTNO has tests under `MDNO/mdno_rtno/tests`. These may be heavier; keep optional/hardware-heavy paths guarded and skip gracefully when optional deps are missing.
- For new features, add targeted tests near the code under test.

## Data Contract & Validators

- See `docs/data_schema.md` for CF/NetCDF expectations.
- Root tools install (exposes CLI): `pip install -e .`
- Validate a dataset with CLI: `psuite-validate <dataset.nc>`
- Or run directly via Python: `python validators/cf_schema.py <dataset.nc>`
- Physics invariants stub for quick checks: `validators/invariants.py` (positivity/CFL placeholders).

## CI

CI runs on push and PR (see `.github/workflows/ci.yml`):

- Matrix: Ubuntu, Windows, macOS; Python 3.10 and 3.11
- Repo-wide syntax: `python -m compileall .`
- STRATUS lane:
  - Installs `STRATUS` with dev + extras
  - Runs Ruff, Black (check), mypy, and `pytest -q`
- MDNO/RTNO lane (optional):
  - Installs `MDNO/mdno_rtno/requirements.txt`
  - Runs `pytest -q`
  - Triggered automatically if MDNO files changed in the PR, or if the PR has label `mdno-ci`

To force MDNO tests on a PR, add the label: `mdno-ci`.

## Contribution Flow

- Fork (if external) and create a topic branch: `feature/...` or `fix/...`
- Keep commits focused; reference issues where applicable.
- Open a PR; CI must be green on STRATUS. MDNO tests run per the rules above.
- Be responsive to review feedback; keep changes minimal and scoped.

## Questions

- For issues specific to STRATUS (physics, RT solvers), open issues under STRATUS paths.
- For MDNO/RTNO (dynamics operator, API service), open issues under MDNO paths.
- If unsure, open an issue and tag with a brief description; we'll triage.

