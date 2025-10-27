# Release Notes - v0.1.0

Date: 2025-10-22

## Highlights

- Spec-first positioning: the repo is framed as a specification and reference implementation for MDNO (v5.3) and RTNO (v4.3), with STRATUS acting as validator and label generator.
- Partner readiness: added a CF/NetCDF data schema stub, the `psuite-validate` CLI, parity adapters, and recipe stubs.
- License and citation: Apache-2.0 at the root, `CITATION.cff` included.
- CI: multi-OS, multi-Python testing for STRATUS; repo-wide syntax checks; optional MDNO tests gated by label or path.

## Changes

- Renamed `MDNO/MDNO v2/` to `MDNO/mdno_rtno/` and updated references.
- Updated the root README with spec-first language, on-prem data stance, partner RFP, and refreshed structure.
- Clarified the STRATUS README positioning.
- Added `docs/data_schema.md` describing CF/NetCDF dimensions, units, and required fields.
- Added validator and adapter stubs under `validators/` and `adapters/`.
- Added training recipe stubs under `configs/recipes/`.
- Replaced "demos" wording with "smoke tests".
- Added `LICENSE`, `CITATION.cff`, `CHANGELOG.md`, and `.zenodo.json`.
- CI workflow now runs Ruff, Black (check), mypy, and pytest for STRATUS on Ubuntu/Windows/macOS and optionally runs MDNO tests.
- Added `CONTRIBUTING.md`, plus PR and issue templates.

## Tooling

- Install STRATUS dev dependencies: `cd STRATUS && pip install -e .[dev]`
- Install root tools: `pip install -e .` (exposes `psuite-validate`)
- Validate a dataset: `psuite-validate <dataset.nc>`

## Partner Engagement

We are seeking labs and industry teams to:
1. Train at scale on their archives
2. Run ablations (scattering order, banding, spherical-harmonic order)
3. Upstream improvements

In return we offer co-authorship (arXiv/JOSS), joint announcements, and module maintainership options.

## Notes

- Update the CI badge in `README.md` to match your GitHub organisation and repository.
- Optional parity adapters (RRTMGP/libRadtran) remain optional; tests skip gracefully if tools are not installed.
