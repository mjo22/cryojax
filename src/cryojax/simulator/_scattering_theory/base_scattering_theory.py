from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, PRNGKeyArray

from ...image import fftn, ifftn, rfftn
from .._instrument_config import InstrumentConfig
from .._structural_ensemble import AbstractStructuralEnsemble
from .._transfer_theory import WaveTransferTheory


class AbstractScatteringTheory(eqx.Module, strict=True):
    """Base class for a scattering theory."""

    @abstractmethod
    def compute_contrast_spectrum_at_detector_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        raise NotImplementedError

    @abstractmethod
    def compute_intensity_spectrum_at_detector_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        raise NotImplementedError


class AbstractWaveScatteringTheory(AbstractScatteringTheory, strict=True):
    """Base class for a wave-based scattering theory."""

    structural_ensemble: eqx.AbstractVar[AbstractStructuralEnsemble]
    transfer_theory: eqx.AbstractVar[WaveTransferTheory]

    @abstractmethod
    def compute_wavefunction_at_exit_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        raise NotImplementedError

    @override
    def compute_intensity_spectrum_at_detector_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        # ... compute the exit wave
        fourier_wavefunction_at_exit_plane = fftn(
            self.compute_wavefunction_at_exit_plane(instrument_config, rng_key)
        )
        # ... propagate to the detector plane
        fourier_wavefunction_at_detector_plane = self.transfer_theory(
            fourier_wavefunction_at_exit_plane,
            instrument_config,
        )
        wavefunction_at_detector_plane = ifftn(fourier_wavefunction_at_detector_plane)
        # ... get the squared wavefunction and return to fourier space
        intensity_spectrum_at_detector_plane = rfftn(
            (
                wavefunction_at_detector_plane * jnp.conj(wavefunction_at_detector_plane)
            ).real
        )
        # ... apply translation
        translational_phase_shifts = self.structural_ensemble.pose.compute_shifts(
            instrument_config.padded_frequency_grid_in_angstroms
        )

        return translational_phase_shifts * intensity_spectrum_at_detector_plane

    @override
    def compute_contrast_spectrum_at_detector_plane(
        self,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Compute the contrast at the detector plane, given the squared wavefunction."""
        # ... compute the exit wave
        fourier_wavefunction_at_exit_plane = fftn(
            self.compute_wavefunction_at_exit_plane(instrument_config, rng_key)
        )
        # ... propagate to the detector plane
        fourier_wavefunction_at_detector_plane = self.transfer_theory(
            fourier_wavefunction_at_exit_plane,
            instrument_config,
        )
        wavefunction_at_detector_plane = ifftn(fourier_wavefunction_at_detector_plane)
        # ... get the squared wavefunction
        squared_wavefunction_at_detector_plane = (
            wavefunction_at_detector_plane * jnp.conj(wavefunction_at_detector_plane)
        ).real
        # ... compute the contrast directly from the squared wavefunction
        # as C = -1 + psi^2 / 1 + psi^2
        contrast_spectrum_at_detector_plane = rfftn(
            (-1 + squared_wavefunction_at_detector_plane)
            / (1 + squared_wavefunction_at_detector_plane)
        )
        # ... apply translation
        translational_phase_shifts = self.structural_ensemble.pose.compute_shifts(
            instrument_config.padded_frequency_grid_in_angstroms
        )

        return translational_phase_shifts * contrast_spectrum_at_detector_plane
