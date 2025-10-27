"""Physics validation helpers for MDNO."""
from typing import Dict, Tuple
import torch


class PhysicsValidator:
    """Validate physical correctness of model outputs."""

    @staticmethod
    def check_conservation(
        state_before: Dict[str, torch.Tensor],
        state_after: Dict[str, torch.Tensor],
        dV: float,
        tolerance: float = 1e-3
    ) -> Dict[str, Tuple[float, bool]]:
        """Check conservation of mass, momentum, and energy.

        Args:
            state_before: Initial state dict with 'rho', 'u', 'v', 'w', 'T'
            state_after: Final state dict
            dV: Volume element (dx * dy * dz)
            tolerance: Relative error tolerance

        Returns:
            Dict with conservation errors and pass/fail status
        """
        results = {}

        # Mass conservation
        mass_before = torch.sum(state_before['rho']) * dV
        mass_after = torch.sum(state_after['rho']) * dV
        mass_error = torch.abs(mass_after - mass_before) / (torch.abs(mass_before) + 1e-10)
        results['mass'] = (mass_error.item(), mass_error.item() < tolerance)

        # Momentum conservation (total)
        def total_momentum(state):
            return (
                torch.sum(state['rho'] * state['u']) +
                torch.sum(state['rho'] * state['v']) +
                torch.sum(state['rho'] * state['w'])
            ) * dV

        mom_before = total_momentum(state_before)
        mom_after = total_momentum(state_after)
        mom_error = torch.abs(mom_after - mom_before) / (torch.abs(mom_before) + 1e-10)
        results['momentum'] = (mom_error.item(), mom_error.item() < tolerance)

        # Energy conservation (kinetic + internal)
        def total_energy(state):
            KE = 0.5 * state['rho'] * (state['u']**2 + state['v']**2 + state['w']**2)
            IE = state['rho'] * 717.0 * state['T']  # cv * T
            return torch.sum(KE + IE) * dV

        energy_before = total_energy(state_before)
        energy_after = total_energy(state_after)
        energy_error = torch.abs(energy_after - energy_before) / (torch.abs(energy_before) + 1e-10)
        results['energy'] = (energy_error.item(), energy_error.item() < tolerance)

        return results

    @staticmethod
    def check_physical_bounds(state: Dict[str, torch.Tensor]) -> Dict[str, bool]:
        """Check if state variables are within physical bounds.

        Returns:
            Dict with boolean flags for each check
        """
        checks = {}

        # Temperature: 100K - 400K (atmospheric range)
        checks['temperature_positive'] = torch.all(state['T'] > 0).item()
        checks['temperature_realistic'] = torch.all(
            (state['T'] >= 100) & (state['T'] <= 400)
        ).item()

        # Pressure: must be positive
        checks['pressure_positive'] = torch.all(state['p'] > 0).item()
        checks['pressure_realistic'] = torch.all(
            (state['p'] >= 100) & (state['p'] <= 110000)
        ).item()

        # Density: must be positive
        checks['density_positive'] = torch.all(state['rho'] > 0).item()
        checks['density_realistic'] = torch.all(
            (state['rho'] >= 0.01) & (state['rho'] <= 2.0)
        ).item()

        # Humidity: 0-1 range
        if 'q' in state:
            checks['humidity_valid'] = torch.all(
                (state['q'] >= 0) & (state['q'] <= 1)
            ).item()

        # No NaNs or Infs
        for var_name, var in state.items():
            checks[f'{var_name}_no_nan'] = not torch.isnan(var).any().item()
            checks[f'{var_name}_no_inf'] = not torch.isinf(var).any().item()

        return checks

    @staticmethod
    def check_radiative_balance(
        radiance: torch.Tensor,
        irradiance: torch.Tensor,
        heating_rate: torch.Tensor
    ) -> Dict[str, Tuple[float, bool]]:
        """Check radiative transfer consistency.

        Returns:
            Dict with validation results
        """
        results = {}

        # Radiance should be non-negative
        results['radiance_positive'] = (
            torch.min(radiance).item(),
            torch.all(radiance >= 0).item()
        )

        # Irradiance should be bounded
        results['irradiance_reasonable'] = (
            torch.max(irradiance).item(),
            torch.max(irradiance).item() < 1500  # Max solar constant
        )

        # Heating rate should be realistic (K/day)
        max_heating = torch.max(torch.abs(heating_rate)).item()
        results['heating_rate_realistic'] = (
            max_heating,
            max_heating < 100.0  # Max ~100 K/day
        )

        return results

    @staticmethod
    def validate_cfl_condition(
        velocity: torch.Tensor,
        dx: float,
        dy: float,
        dz: float,
        dt: float,
        max_cfl: float = 1.0
    ) -> Tuple[float, bool]:
        """Validate CFL condition.

        Returns:
            (cfl_number, is_stable)
        """
        u = velocity[:, 0] if velocity.dim() > 1 else velocity
        v = velocity[:, 1] if velocity.shape[1] > 1 else torch.zeros_like(u)
        w = velocity[:, 2] if velocity.shape[1] > 2 else torch.zeros_like(u)

        cfl = (
            torch.abs(u).max() * dt / dx +
            torch.abs(v).max() * dt / dy +
            torch.abs(w).max() * dt / dz
        )

        return cfl.item(), cfl.item() <= max_cfl


def run_comprehensive_validation(
    model,
    inputs: Dict[str, torch.Tensor],
    config,
    verbose: bool = True
) -> Dict[str, any]:
    """Run comprehensive physics validation on model.

    Args:
        model: MDNO model instance
        inputs: Input tensors
        config: Model configuration
        verbose: Print results

    Returns:
        Dict with all validation results
    """
    results = {'overall_pass': True}

    with torch.no_grad():
        outputs = model(inputs)

    # Test conservation laws
    if 'meso' in inputs and 'meso' in outputs:
        state_in = model._tensor_to_state_dict(inputs['meso'])
        state_out = model._tensor_to_state_dict(outputs['meso'])
        dV = config.dx * config.dy * config.dz

        conservation = PhysicsValidator.check_conservation(
            state_in, state_out, dV
        )
        results['conservation'] = conservation

        if verbose:
            print("\n=== Conservation Laws ===")
            for quantity, (error, passed) in conservation.items():
                status = "PASS" if passed else "FAIL"
                print(f"  {quantity.capitalize()}: {error:.3e} {status}")

        results['overall_pass'] &= all(p for _, p in conservation.values())

    # Test physical bounds
    if 'meso' in outputs:
        state_out = model._tensor_to_state_dict(outputs['meso'])
        bounds = PhysicsValidator.check_physical_bounds(state_out)
        results['bounds'] = bounds

        if verbose:
            print("\n=== Physical Bounds ===")
            failed_checks = [k for k, v in bounds.items() if not v]
            if failed_checks:
                print(f"  FAILed checks: {', '.join(failed_checks)}")
                results['overall_pass'] = False
            else:
                print("  All bounds checks passed")

    # Test CFL condition
    if 'meso' in outputs:
        state_out = model._tensor_to_state_dict(outputs['meso'])
        velocity = torch.stack([state_out['u'], state_out['v'], state_out['w']], dim=1)
        cfl, stable = PhysicsValidator.validate_cfl_condition(
            velocity, config.dx, config.dy, config.dz, config.dt
        )
        results['cfl'] = (cfl, stable)

        if verbose:
            print(f"\n=== CFL Condition ===")
            status = "STABLE" if stable else "UNSTABLE"
            print(f"  CFL number: {cfl:.3f} {status}")

        results['overall_pass'] &= stable

    # Test radiative transfer if present
    if 'radiative_transfer' in outputs:
        rad_results = PhysicsValidator.check_radiative_balance(
            outputs['radiative_transfer'],
            outputs.get('radiative_irradiance', torch.zeros(1)),
            outputs.get('radiative_heating_rate', torch.zeros(1))
        )
        results['radiative'] = rad_results

        if verbose:
            print("\n=== Radiative Transfer ===")
            for check, (value, passed) in rad_results.items():
                status = "PASS" if passed else "FAIL"
                print(f"  {check}: {value:.3e} {status}")

        results['overall_pass'] &= all(p for _, p in rad_results.values())

    if verbose:
        print("\n" + "="*50)
        overall_status = "ALL CHECKS PASSED" if results['overall_pass'] else "SOME CHECKS FAILED"
        print(f"Overall: {overall_status}")
        print("="*50)

    return results


__all__ = ['PhysicsValidator', 'run_comprehensive_validation']





