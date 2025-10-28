# Atmospheric and Radiative Transfer Suite

[![CI](https://github.com/tsor/atmospheric-radiative-transfer-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/tsor/atmospheric-radiative-transfer-suite/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tsor/atmospheric-radiative-transfer-suite/branch/main/graph/badge.svg)](https://codecov.io/gh/tsor/atmospheric-radiative-transfer-suite)

See the contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)

The Atmospheric and Radiative Transfer Suite bundles two complementary toolkits:

- **MDNO/RTNO** - a coupled multi-scale diffusion neural operator (MDNO v5.3) and radiative-transfer neural operator (RTNO v4.3) with a FastAPI service.
- **STRATUS** - a modular radiative-transfer framework featuring volumetric ray marching, polarization (Stokes vectors, Mueller calculus), Monte Carlo transport, spherical harmonics, and a physics-informed neural network solver.

Use this repository when you need both advanced atmospheric dynamics powered by neural operators and a configurable radiative-transfer engine.

> **📖 New to the physics?** See the [Model Physics and Mathematics Wiki](../../wiki/Model-Physics-and-Mathematics) for detailed mathematical derivations covering the Boltzmann equation, Navier-Stokes dynamics, radiative transfer theory, polarization, and numerical methods.

## Repository Layout

- `MDNO/` - entry point for the MDNO/RTNO stack
  - `MDNO/README.md` - overview and references
  - `MDNO/mdno_rtno/` - main package and service
    - `mdno/` - MDNO (multi-scale atmospheric dynamics)
    - `rtno/` - RTNO (radiative-transfer physics)
    - `physics/` - unified facade (`create_model("mdno"|"rtno")`)
    - `api.py` - FastAPI inference surface
    - `requirements.txt` - pinned dependencies
    - `deploy.sh`, `k8s.yaml`, `pre_deploy_validation.sh` - operations artefacts
    - `README.md` - detailed setup and usage
- `STRATUS/` - refactored STRATUS package
  - `stratus/` - library modules (config, ray marching, data, polarization, Monte Carlo, PINN, export, validation)
  - `pyproject.toml` - build and dependency metadata
  - `README.md` - physics background and examples
  - `tests/` - unit and integration tests
- `validators/` - CF/NetCDF schema and physics invariants CLI (`psuite-validate`)
- `adapters/` - parity adapters for RRTMGP and libRadtran/DISORT
- `configs/` - training recipes and partner blueprints
- `docs/` - data schema and supporting documentation
  - `wiki/` - comprehensive mathematical documentation (also available as [GitHub Wiki](../../wiki))

## System Requirements

- Python 3.9 or newer (3.10+ recommended)
- PyTorch (CPU or GPU builds)
- Linux, macOS, or Windows
- Optional extras: `onnx`, `h5py`, `netCDF4`, `psutil`, `GPUtil`, `scipy` (used by STRATUS extras and MDNO tooling)

> Tip: MDNO/RTNO and STRATUS declare separate dependency sets; use two virtual environments during development.

## Quick Start

### Option A - STRATUS

1. Create a virtual environment and install STRATUS in editable mode:

   ```bash
   cd STRATUS
   python -m venv .venv
   . .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e .[dev]
   ```

2. Run a quick smoke test and the test suite:

   ```bash
   python stratus_refactored.py
   pytest -q
   ```

3. Minimal usage example:

   ```python
   from stratus.config import StratusConfig
   from stratus.model import StratusRadianceModel

   config = StratusConfig(grid_shape=(64, 64, 32), n_bands=8)
   model = StratusRadianceModel(config)

   import torch

   kappa = torch.rand(1, config.n_stokes, *config.grid_shape, config.n_bands)
   source = torch.rand(1, config.n_stokes, *config.grid_shape, config.n_bands)
   output = model(kappa, source)
   ```

### Option B - MDNO/RTNO

```bash
cd MDNO/mdno_rtno
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                       # optional: requires PyTorch and dependencies
uvicorn api:main --reload       # launch inference service
```

## Spec-First Positioning

This repository publishes a specification and reference implementation for:

- MDNO v5.3 (multi-scale atmospheric dynamics operator)
- RTNO v4.3 (radiative-transfer operator)
- STRATUS (validator and label generator)

It is designed for partners who bring their own data and compute. No datasets or model weights are redistributed.

### Data Stays On-Premises

- CF/NetCDF contract described in `docs/data_schema.md`
- `psuite-validate` CLI performs schema and physics checks on sample files

### Ready-to-Train Test Suite

- `validators/` - schema validation, positivity, CFL metrics
- `adapters/` - optional parity bridges to RRTMGP (scalar labels) and libRadtran/DISORT (polarised labels)
- `configs/recipes/` - blueprints (e.g. ERA5 + RRTMGP parity, CAMS geometry + polarised columns, CERES flux closure)

Optional dependencies are detected automatically; tests skip gracefully when tools are absent.

### Partner Engagement

We are looking for collaborators to:

1. Train at scale on proprietary archives
2. Run ablations (scattering order, spectral banding, spherical-harmonic order)
3. Upstream improvements and validation artefacts

In return we offer co-authorship opportunities (arXiv/JOSS), joint announcements, and module maintainership.

## Typical Workflows

- Radiative-transfer research: work inside `STRATUS/`, iterate quickly, benchmark, export models (ONNX/TorchScript).
- Coupled dynamics + radiative transfer: run `MDNO/mdno_rtno/`, deploy the FastAPI service, or import `physics.create_model`.

## Development Tasks

- STRATUS:
  - `ruff check .`
  - `black --check .`
  - `mypy stratus`
  - `pytest`
  - Optional extras: `pip install -e .[dataset,export,monitoring,mie,dev]`
- MDNO/RTNO:
  - `pip install -r MDNO/mdno_rtno/requirements.txt`
  - `python -m compileall MDNO/mdno_rtno`
  - `pytest -q`

## Reproducibility

- Seed all stochastic libraries (e.g. `torch.manual_seed(...)`, `numpy.random.seed(...)`) before running experiments; STRATUS components respect the active global RNG state.
- Configure `MonteCarloConfig.seed` to keep the radiative Monte Carlo solver deterministic across runs and deployments.
- Pin dependency versions via `pyproject.toml`/`requirements.txt` or a lock file when training; CI exercises the physics regression suite (`pytest -q`) to guard energy conservation, PINN multi-Stokes support, and ray-marching accuracy.
- When distributing checkpoints, record the STRATUS/MDNO configuration snapshots alongside the seeds used for generation.

## CLI Usage

- Install root tools: `pip install -e .`
- Validate a NetCDF sample: `psuite-validate <dataset.nc>`

## Continuous Integration

- GitHub Actions matrix on Ubuntu, Windows, macOS with Python 3.10 and 3.11
- Repository-wide syntax check via `python -m compileall .`
- STRATUS lane: install extras, run Ruff, Black (check), mypy, and `pytest -q`
- MDNO/RTNO lane: optional; runs when MDNO files change or the `mdno-ci` label is set
- Coverage uploads routed to Codecov

## Documentation

- **[Wiki: Model Physics and Mathematics](../../wiki/Model-Physics-and-Mathematics)** - comprehensive mathematical formulations, physical equations, and computational methods for MDNO v5.3, RTNO v4.3, and STRATUS
- `STRATUS/README.md` - physics derivations, configuration, end-to-end examples, Monte Carlo and PINN variants
- `MDNO/mdno_rtno/README.md` - unified framework, mathematical background, programmatic usage, deployment artefacts
- `docs/data_schema.md` - CF/NetCDF expectations and invariants

## Citation

If you use this project, please credit:

- Thierry Silvio Claude Soreze

Cite STRATUS and/or MDNO/RTNO according to your venue's guidelines.

