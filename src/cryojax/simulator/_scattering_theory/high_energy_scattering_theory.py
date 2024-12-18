from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, PRNGKeyArray

from ...image import ifftn, irfftn
from .._instrument_config import InstrumentConfig
from .._potential_integrator import AbstractPotentialIntegrator
from .._solvent import AbstractIce
from .._structural_ensemble import AbstractStructuralEnsemble
from .._transfer_theory import WaveTransferTheory
from .base_scattering_theory import AbstractWaveScatteringTheory
from .common_functions import convert_units_of_integrated_potential


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

    def __init__(
        self,
        structural_ensemble: AbstractStructuralEnsemble,
        potential_integrator: AbstractPotentialIntegrator,
        transfer_theory: WaveTransferTheory,
        solvent: Optional[AbstractIce] = None,
    ):
        """**Arguments:**

        - `structural_ensemble`: The structural ensemble of scattering potentials.
        - `potential_integrator`: The method for integrating the scattering potential.
        - `transfer_theory`: The wave transfer theory.
        - `solvent`: The model for the solvent.
        """
        self.structural_ensemble = structural_ensemble
        self.potential_integrator = potential_integrator
        self.transfer_theory = transfer_theory
        self.solvent = solvent

    @override
    def compute_wavefunction_at_exit_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        # Compute the object spectrum in the exit plane
        potential = self.structural_ensemble.get_potential_in_lab_frame()
        if self.potential_integrator.is_projection_approximation:
            object_spectrum_at_exit_plane = convert_units_of_integrated_potential(
                self.potential_integrator.compute_fourier_integrated_potential(
                    potential, instrument_config
                ),
                instrument_config.wavelength_in_angstroms,
            )

            if rng_key is not None:
                # Get the potential of the specimen plus the ice
                if self.solvent is not None:
                    object_spectrum_at_exit_plane = (
                        self.solvent.compute_object_spectrum_with_ice(
                            rng_key,
                            object_spectrum_at_exit_plane,
                            instrument_config,
                            is_hermitian_symmetric=True,
                        )
                    )

            return jnp.exp(
                1.0j
                * irfftn(object_spectrum_at_exit_plane, s=instrument_config.padded_shape)
            )
        else:
            object_spectrum_at_exit_plane = convert_units_of_integrated_potential(
                self.potential_integrator.compute_fourier_integrated_potential(
                    potential, instrument_config
                ),
                instrument_config.wavelength_in_angstroms,
            )

            if rng_key is not None:
                # Get the potential of the specimen plus the ice
                if self.solvent is not None:
                    object_spectrum_at_exit_plane = (
                        self.solvent.compute_object_spectrum_with_ice(
                            rng_key,
                            object_spectrum_at_exit_plane,
                            instrument_config,
                            is_hermitian_symmetric=False,
                        )
                    )

            return jnp.exp(
                1.0j
                * ifftn(object_spectrum_at_exit_plane, s=instrument_config.padded_shape)
            )
