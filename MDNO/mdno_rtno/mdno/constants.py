from dataclasses import dataclass


@dataclass(frozen=True)
class AtmosphericConstants:
    """Comprehensive atmospheric physics constants."""

    # Fundamental
    GRAVITY: float = 9.80665
    EARTH_RADIUS: float = 6371000.0
    EARTH_ROTATION_RATE: float = 7.2921e-5

    # Thermodynamic
    DRY_AIR_GAS_CONSTANT: float = 287.058
    WATER_VAPOR_GAS_CONSTANT: float = 461.495
    CP_DRY_AIR: float = 1005.0
    CP_WATER_VAPOR: float = 1850.0
    CV_DRY_AIR: float = 718.0
    LATENT_HEAT_VAPORIZATION: float = 2.5e6
    LATENT_HEAT_FUSION: float = 3.34e5
    LATENT_HEAT_SUBLIMATION: float = 2.834e6

    # Reference values
    STANDARD_PRESSURE: float = 101325.0
    STANDARD_TEMPERATURE: float = 288.15
    STANDARD_DENSITY: float = 1.225
    TROPOPAUSE_HEIGHT: float = 11000.0

    # Turbulence
    VON_KARMAN_CONSTANT: float = 0.41
    PRANDTL_NUMBER: float = 0.71

    # Cloud physics
    CRITICAL_RELATIVE_HUMIDITY: float = 0.8
    CLOUD_DROPLET_RADIUS: float = 10e-6
    RAIN_DROP_RADIUS: float = 1e-3

    # Molecular
    BOLTZMANN: float = 1.380649e-23
    PLANCK: float = 6.62607015e-34
    SPEED_OF_LIGHT: float = 299792458.0
    AVOGADRO: float = 6.02214076e23


CONSTANTS = AtmosphericConstants()
