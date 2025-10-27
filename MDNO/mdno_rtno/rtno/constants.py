from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class PhysicsConstants:
    """Comprehensive physical constants for radiative transfer."""

    # Fundamental
    SPEED_OF_LIGHT: float = 299792458.0  # m/s
    BOLTZMANN: float = 1.380649e-23  # J/K
    PLANCK: float = 6.62607015e-34  # J*s
    STEFAN_BOLTZMANN: float = 5.670374419e-8  # W*m-^2*K-^4
    AVOGADRO: float = 6.02214076e23  # mol-^1

    # Atmospheric
    DRY_AIR_GAS_CONSTANT: float = 287.058  # J/(kg*K)
    WATER_VAPOR_GAS_CONSTANT: float = 461.495  # J/(kg*K)
    STANDARD_PRESSURE: float = 101325.0  # Pa
    STANDARD_TEMPERATURE: float = 288.15  # K
    EARTH_RADIUS: float = 6371000.0  # m
    GRAVITY: float = 9.80665  # m/s^2

    # Optical
    RAYLEIGH_DEPOLARIZATION: float = 0.0279
    AIR_REFRACTIVE_INDEX: float = 1.000293
    RAYLEIGH_CROSS_SECTION_550NM: float = 5.8e-31  # m^2

    # Absorption bands
    O3_BANDS: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (254, 1.15e-17),
            (280, 5.0e-18),
            (310, 1.9e-19),
            (600, 5.0e-21),
        ]
    )
    H2O_BANDS: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (940, 2.5e-23),
            (1130, 8.2e-24),
            (1380, 1.8e-22),
            (1880, 8e-24),
        ]
    )
    CO2_BANDS: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (1400, 3.5e-25),
            (1600, 7.8e-26),
            (2000, 4.2e-24),
            (2700, 1.2e-24),
        ]
    )

    # Numerical safeguards
    CONVERGENCE_TOL: float = 1e-7
    MAX_ITERATIONS: int = 200
    OPTICAL_DEPTH_THRESHOLD: float = 20.0
    SMALL_TAU_THRESHOLD: float = 1e-4


CONSTANTS = PhysicsConstants()

__all__ = ["PhysicsConstants", "CONSTANTS"]
