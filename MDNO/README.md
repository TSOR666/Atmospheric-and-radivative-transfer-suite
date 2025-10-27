# MDNO Atmospheric Modeling Suite

The MDNO Atmospheric Modeling Suite delivers the production implementations of:

- **MDNO v5.3** - Multi-scale Diffusion Neural Operator for atmospheric dynamics
- **RTNO v4.3** - Radiative Transfer Neural Operator

All source files live under [`mdno_rtno/`](./mdno_rtno/), which bundles:

- `mdno/` - kinetic, meso-scale, and macro-scale physics modules
- `rtno/` - radiative-transfer solvers and neural corrections
- `physics/` - factory helpers to instantiate MDNO or RTNO models
- `api.py` - FastAPI service for inference
- Deployment utilities (`deploy.sh`, `k8s.yaml`, `pre_deploy_validation.sh`)
- Tests (`mdno_rtno/tests/`)

## Key Features

- Multi-scale physics: Boltzmann micro-physics, primitive-equation meso dynamics, Hamiltonian macro coupling
- Physics-informed neural operators (FNO-based) with conservation enforcement
- Integrated turbulence, cloud microphysics, and chemistry toggles
- Optional RTNO coupling for radiative-transfer feedback
- Monitoring, validation, and trainer utilities for large-scale runs

## Getting Started

```bash
cd MDNO/mdno_rtno
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q                         # optional: requires PyTorch
uvicorn api:main --reload         # launch the FastAPI inference service
```

## MDNO Overview

- **Configuration**: `mdno/config.py` defines `MDNOConfig` (grid shapes, time step, physics toggles).
- **Model**: `mdno/model.py` assembles FNO blocks, Boltzmann solver, primitive equations, turbulence, microphysics, chemistry, and Hamiltonian coupling.
- **Physics Modules**: `mdno/physics.py` holds detailed implementations for kinetic transport, spectral advection, turbulence parameterisation, microphysics, chemistry, and scale bridging.
- **Training**: `mdno/trainer.py` provides an `MDNOTrainer` with mixed-precision support, gradient clipping, cosine restarts, and checkpointing.
- **Validation**: `mdno/validation.py` exposes automated conservation and stability checks.

Typical workflow:

```python
from mdno.config import MDNOConfig
from mdno.model import EnhancedMDNO_v53_Complete

config = MDNOConfig(use_radiative_transfer=True)
model = EnhancedMDNO_v53_Complete(config)
```

## RTNO Overview

- **Configuration**: `rtno/config.py` defines `RTNOConfig` (spectral bands, scattering options, discrete ordinates).
- **Model**: `rtno/model.py` combines deterministic radiative-transfer solvers with neural residual corrections.
- **Solver**: `rtno/solver.py` implements multi-stream radiative transfer with support for polarization and gas absorption.
- **Monitoring**: `rtno/monitoring.py` tracks inference timing and GPU usage.

Usage example:

```python
from rtno import RTNOConfig, EnhancedRTNO_v43

config = RTNOConfig(n_stokes=4, use_polarization=True)
model = EnhancedRTNO_v43(config)
```

## Unified Physics Interface

`physics/__init__.py` exposes:

```python
from physics import create_model

mdno_model = create_model("mdno")
rtno_model = create_model("rtno")
```

This ensures consistent configuration and device management between MDNO and RTNO.

## Tests

- Unit and integration tests live in `mdno_rtno/tests/`:
  - `test_mdno.py`
  - `test_rtno.py`
  - `test_api.py`
- Run with `pytest -q` once PyTorch and FastAPI dependencies are installed.

## Deployment

- `deploy.sh` - orchestrates build, test, and deployment steps
- `k8s.yaml` - reference Kubernetes manifests
- `pre_deploy_validation.sh` - smoke tests before rollout

## Additional Resources

- `README_AUDIT.md` - audit notes and optimisation history
- `OPTIMIZATION_CHANGELOG.md` - performance improvements across versions
- `mdno/demo.py` and `rtno/demo.py` - end-to-end smoke tests

For further detail consult the in-package docstrings and comments; every module documents the governing equations, numerical schemes, and configuration options in plain ASCII.

