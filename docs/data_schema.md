# Data Schema (CF/NetCDF)

- Dimensions: `time`, `level`, `lat`, `lon` (optional: `band`, `mu`)
- Units: SI (K, kg/kg, Pa, m/s, ...); use CF `standard_name` where applicable
- Required core fields (examples): `u`, `v`, `w`, `T`, `q`, `surface_pressure`, `cloud_fraction`, hydrometeor mass-mixing ratios, aerosol modes
- Radiative-transfer fields (examples): gas mixing ratios (H2O, CO2, O3, ...), optical properties (extinction, single-scattering albedo, asymmetry factor `g`), surface albedo/emissivity, geometry (solar zenith angle, view zenith angle, relative azimuth)
- Conventions: latitude within [-90, 90], longitude within [0, 360) or [-180, 180), monotonic `level` with metadata, and `time` as UTC ISO 8601
