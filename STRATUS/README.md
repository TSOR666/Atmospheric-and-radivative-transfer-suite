# STRATUS Radiative-Transfer Toolkit

STRATUS is a modular radiative-transfer framework designed for atmospheric science and computer graphics workloads. It combines deterministic solvers, Monte Carlo transport, and physics-informed neural networks within a unified PyTorch codebase.

## Features

- Fourier neural operators (TrueFNO3D) for spatial transport
- Volumetric ray marching with adaptive step control
- Polarization support (Stokes vectors, Mueller matrices)
- Monte Carlo solver with configurable scattering order
- Physics-informed neural network (PINN) solver
- Multiscale feature encoding and spectral processing helpers
- Export utilities (TorchScript, ONNX) and validation harnesses

## Installation

```bash
cd STRATUS
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

Run tests:

```bash
pytest -q
ruff check stratus
black --check stratus
mypy stratus
```

## Quick Example

```python
from stratus.config import StratusConfig
from stratus.model import StratusRadianceModel

config = StratusConfig(grid_shape=(32, 32, 16), n_bands=4)
model = StratusRadianceModel(config)

import torch

kappa = torch.rand(1, config.n_stokes, *config.grid_shape, config.n_bands)
source = torch.zeros_like(kappa)
result = model(kappa, source)
print(result["radiance"].shape)
```

## Project Structure

- `stratus/config.py` - configuration objects and enums
- `stratus/model.py` - high-level radiance model orchestrating all subcomponents
- `stratus/raymarch.py` - volumetric ray marching utilities
- `stratus/polarization.py` - Mueller matrix algebra
- `stratus/monte_carlo.py` - Monte Carlo transport
- `stratus/pinn.py` - physics-informed radiative-transfer solver
- `stratus/multiscale.py` - multiscale feature encoders
- `stratus/export.py` - exporters and CLI helpers
- `stratus/validation.py` - radiative validation suite
- `stratus/tests/` - unit and integration tests

## Optional Components

- Monte Carlo radiative transfer (`MonteCarloRadiativeTransfer`)
- Physics-informed RTE (`PhysicsInformedRTE`)
- Dataset utilities (`StratusDataset`)
- Performance monitoring (`PerformanceMonitor`)

Optional extras are exposed via `pip install -e .[dataset,export,monitoring,mie,dev]`.

## Validation

Use the built-in validator to run a uniform slab test:

```python
from stratus.validation import RadiativeValidator

validator = RadiativeValidator(model, config)
result = validator.uniform_slab_test()
print("PASS" if result.passed else "FAIL", result.metrics)
```

## Export

```python
from stratus.export import StratusExporter, ExportInputs

exporter = StratusExporter(model, config)
inputs = ExportInputs(kappa=kappa, source=source)
exporter.export_onnx(inputs, "stratus.onnx")
```

## License

STRATUS is distributed under the Apache 2.0 License as part of the Atmospheric and Radiative Transfer Suite.

