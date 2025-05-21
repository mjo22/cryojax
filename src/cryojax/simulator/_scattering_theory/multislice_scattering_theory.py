from typing import Optional
from typing_extensions import override

from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ...internal import error_if_not_fractional
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
    amplitude_contrast_ratio: Float[Array, ""]

    def __init__(
        self,
        structural_ensemble: AbstractStructuralEnsemble,
        multislice_integrator: AbstractMultisliceIntegrator,
        transfer_theory: WaveTransferTheory,
        amplitude_contrast_ratio: float | Float[Array, ""] = 0.1,
    ):
        """**Arguments:**

        - `structural_ensemble`: The structural ensemble of scattering potentials.
        - `multislice_integrator`: The multislice method.
        - `transfer_theory`: The wave transfer theory.
        - `amplitude_contrast_ratio`: The amplitude contrast ratio.
        """
        self.structural_ensemble = structural_ensemble
        self.multislice_integrator = multislice_integrator
        self.transfer_theory = transfer_theory
        self.amplitude_contrast_ratio = error_if_not_fractional(amplitude_contrast_ratio)

    @override
    def compute_wavefunction_at_exit_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        # Get potential in the lab frame
        potential = self.structural_ensemble.get_potential_in_transformed_frame(
            apply_translation=False
        )
        # Compute the wavefunction in the exit plane
        wavefunction_at_exit_plane = (
            self.multislice_integrator.compute_wavefunction_at_exit_plane(
                potential, instrument_config, self.amplitude_contrast_ratio
            )
        )

        return wavefunction_at_exit_plane
