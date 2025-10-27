# MDNO/RTNO Audit Summary

This document records the major audit findings and remediation steps undertaken while preparing MDNO v5.3 and RTNO v4.3 for release.

## Timeline

- **2025-08** - Initial refactor of legacy MDNO scripts into package form
- **2025-09** - Integration of RTNO residual network and diagnostics
- **2025-10** - Comprehensive audit across physics modules, training loops, and deployment artefacts

## Key Fixes

- Corrected Boltzmann solver moment projection and FNO channel alignment
- Hardened spectral advection to avoid mixed-precision FFT issues
- Added Hamiltonian energy conservation checks on macro grids
- Guarded `GradScaler` usage for CPU-only environments
- Added scheduler stepping and checkpoint tracking within `MDNOTrainer`
- Validated radiative-transfer residual network shape assumptions
- Enabled Prometheus multi-process metrics in the FastAPI service

## Outstanding Work

- Extend parity adapters with full libRadtran and RRTMGP integrations
- Automate large-scale benchmark suite for MDNO + RTNO coupling
- Expand API load testing and include GPU utilisation alarms

## Testing

- `pytest -q mdno_rtno/tests`
- `python -m mdno.demo` and `python -m rtno.demo` for smoke tests
- `python -m compileall mdno_rtno` for syntax validation

## Contact

Questions or follow-up items should be filed via GitHub issues or directed to the maintainers listed in `CITATION.cff`.

