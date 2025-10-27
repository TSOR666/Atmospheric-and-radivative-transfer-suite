# MDNO/RTNO Unified Physics Framework

This package merges the production-ready MDNO (Multi-scale Diffusion Neural Operator) and RTNO (Radiative Transfer Neural Operator) implementations. It exposes:

- `mdno/` - atmospheric dynamics modules (Boltzmann solver, primitive equations, turbulence, microphysics, chemistry, Hamiltonian coupling)
- `rtno/` - radiative-transfer modules (solver, residual network, polarization, spectroscopy helpers)
- `physics/` - a unified factory to create MDNO or RTNO instances
- `api.py` - FastAPI service for hosting inference endpoints
- `tests/` - API, MDNO, and RTNO test suites
- Deployment scripts (`deploy.sh`, `k8s.yaml`, `pre_deploy_validation.sh`)

## Installation

```bash
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: run tests (requires PyTorch):

```bash
pytest -q
```

## Quick Usage

```python
from physics import create_model

mdno = create_model("mdno", config_kwargs={"use_radiative_transfer": True})
rtno = create_model("rtno")
```

Launch the FastAPI service:

```bash
uvicorn api:main --reload
```

## MDNO Highlights

- Configurable via `MDNOConfig` (`mdno/config.py`)
- FNO-based operators for micro, meso, and macro grids (`mdno/model.py`)
- Boltzmann solver with moment conservation and entropic projection
- Scale bridging between kinetic and continuum regimes
- Optional cloud microphysics, chemistry, turbulence, and radiative coupling
- Trainer with AMP support, gradient clipping, cosine restarts (`mdno/trainer.py`)
- Validation utilities for conservation and stability checks (`mdno/validation.py`)

## RTNO Highlights

- Configurable via `RTNOConfig` (`rtno/config.py`)
- Multi-stream radiative-transfer solver with polarization (`rtno/solver.py`)
- Neural residual network for sub-grid corrections (`rtno/model.py`)
- Heating-rate, net-flux, and diagnostic outputs
- Monitoring instrumentation (`rtno/monitoring.py`)

## API Service

`api.py` wraps both models behind a FastAPI application:

- `/predict` endpoint selecting `mdno` or `rtno`
- Prometheus metrics with multiprocess support
- Structured logging middleware and request validation

## Testing

```
pytest -q mdno_rtno/tests
```

Tests cover:

- MDNO conservation checks (`test_mdno.py`)
- RTNO radiance outputs (`test_rtno.py`)
- API error handling (`test_api.py`)

## Deployment

- `deploy.sh` - build, test, deploy pipeline
- `pre_deploy_validation.sh` - smoke tests for CLI and inference
- `k8s.yaml` - reference Kubernetes manifests

## Additional Documentation

- `README_AUDIT.md` - audit summary and remediation history
- `OPTIMIZATION_CHANGELOG.md` - performance improvements between releases
- `mdno/demo.py`, `rtno/demo.py` - runnable demos

All documentation and code comments are ASCII-only to ease reuse in automated pipelines.

