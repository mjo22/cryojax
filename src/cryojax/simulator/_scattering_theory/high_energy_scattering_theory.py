from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, PRNGKeyArray

from ...image import fftn, irfftn
from .._instrument_config import InstrumentConfig
from .._potential_integrator import AbstractPotentialIntegrator
from .._solvent import AbstractIce
from .._structural_ensemble import AbstractStructuralEnsemble
from .._transfer_theory import WaveTransferTheory
from .base_scattering_theory import AbstractWaveScatteringTheory
from .common_functions import compute_phase_shifts_from_integrated_potential


class HighEnergyScatteringTheory(AbstractWaveScatteringTheory, strict=True):
    """Scattering theory in the high-energy approximation (eikonal approximation).

    This is the simplest model for multiple scattering events.

    **References:**

    - For the definition of the exit wave in the eikonal approximation, see Chapter 69,
      Page 2012, from *Hawkes, Peter W., and Erwin Kasper. Principles of Electron
      Optics, Volume 4: Advanced Wave Optics. Academic Press, 2022.*
    """

    structural_ensemble: AbstractStructuralEnsemble
    potential_integrator: AbstractPotentialIntegrator
    transfer_theory: WaveTransferTheory
    solvent: Optional[AbstractIce]

    @override
    def compute_fourier_wavefunction_at_exit_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        # Compute the phase shifts in the exit plane
        potential = self.structural_ensemble.get_potential_in_lab_frame()
        fourier_phase_shifts_at_exit_plane = (
            compute_phase_shifts_from_integrated_potential(
                self.potential_integrator.compute_fourier_integrated_potential(
                    potential, instrument_config
                ),
                instrument_config.wavelength_in_angstroms,
            )
        )

        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                fourier_phase_shifts_at_exit_plane = (
                    self.solvent.compute_fourier_phase_shifts_with_ice(
                        rng_key, fourier_phase_shifts_at_exit_plane, instrument_config
                    )
                )

        return fftn(
            jnp.exp(
                1.0j
                * irfftn(
                    fourier_phase_shifts_at_exit_plane, s=instrument_config.padded_shape
                )
            )
        )