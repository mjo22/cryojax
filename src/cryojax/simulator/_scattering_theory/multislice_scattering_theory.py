from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, PRNGKeyArray

from ...image import fftn
from .._instrument_config import InstrumentConfig
from .._multislice_integrator import AbstractMultisliceIntegrator
from .._structural_ensemble import AbstractStructuralEnsemble
from .._transfer_theory import WaveTransferTheory
from .base_scattering_theory import AbstractWaveScatteringTheory


class MultisliceScatteringTheory(AbstractWaveScatteringTheory, strict=True):
    """A scattering theory using the multislice method."""

    structural_ensemble: AbstractStructuralEnsemble
    multislice_integrator: AbstractMultisliceIntegrator
    transfer_theory: WaveTransferTheory

    @override
    def compute_fourier_wavefunction_at_exit_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        # Compute the wavefunction in the exit plane
        fourier_wavefunction_at_exit_plane = (
            _compute_fourier_wavefunction_from_scattering_potential(
                self.structural_ensemble, self.multislice_integrator, instrument_config
            )
        )

        return fourier_wavefunction_at_exit_plane


def _compute_fourier_wavefunction_from_scattering_potential(
    structural_ensemble, multislice_integrator, instrument_config
):
    # Get potential in the lab frame
    potential = structural_ensemble.get_potential_in_lab_frame()
    # Compute the wavefunction in the exit plane
    wavefunction_in_exit_plane = multislice_integrator.compute_wavefunction_at_exit_plane(
        potential, instrument_config
    )
    # Compute in-plane translation through fourier phase shifts
    translational_phase_shifts = structural_ensemble.pose.compute_shifts(
        instrument_config.padded_frequency_grid_in_angstroms
    )
    return fftn(wavefunction_in_exit_plane) * jnp.exp(1.0j * translational_phase_shifts)
