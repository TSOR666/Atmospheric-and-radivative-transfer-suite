# MDNO/RTNO Optimisation Log

## v5.3.1 / v4.3.1 (2025-10-15)

- Vectorised Boltzmann spectral advection (10x-100x speedup depending on velocity resolution)
- Added batched moment calculations and reduced intermediate tensor allocations
- Introduced channel adapters for macro FNO operators to avoid redundant reshaping
- Enabled gradient checkpointing toggles for memory-constrained runs
- Cached radiative-transfer lookup tables to avoid recomputation

## v5.3.0 / v4.3.0 (2025-09-10)

- Ported legacy MDNO and RTNO code to modular packages (`mdno/`, `rtno/`)
- Replaced custom FFT wrappers with native PyTorch implementations
- Added monitoring hooks for wall-clock timing and GPU usage
- Implemented residual connections in the RTNO neural corrector

## Planned Improvements

- Kernel fusion for repeated stencil operators
- Optional CUDA graph capture for inference workloads
- Asynchronous data prefetch utilities for large ensemble runs

