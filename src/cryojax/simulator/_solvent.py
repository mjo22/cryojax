"""
Abstraction of the ice in a cryo-EM image.
"""

from abc import abstractmethod
from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from equinox import Module
from jaxtyping import Array, Complex, PRNGKeyArray

from ..image.operators import FourierOperatorLike
from ._instrument_config import InstrumentConfig
from ._scattering_theory import compute_phase_shifts_from_integrated_potential


class AbstractIce(Module, strict=True):
    """Base class for an ice model."""

    @abstractmethod
    def sample_fourier_phase_shifts_from_ice(
        self, key: PRNGKeyArray, instrument_config: InstrumentConfig
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Sample a stochastic realization of the phase shifts due to the ice
        at the exit plane."""
        raise NotImplementedError

    def compute_fourier_phase_shifts_with_ice(
        self,
        key: PRNGKeyArray,
        fourier_phase_at_exit_plane: Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}",
        ],
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Compute the combined phase of the ice and the specimen."""
        # Sample the realization of the phase due to the ice.
        fourier_ice_phase_at_exit_plane = self.sample_fourier_phase_shifts_from_ice(
            key, instrument_config
        )

        return fourier_phase_at_exit_plane + fourier_ice_phase_at_exit_plane


class GaussianIce(AbstractIce, strict=True):
    r"""Ice modeled as gaussian noise.

    **Attributes:**

    - `variance_function` :
        A function that computes the variance
        of the ice, modeled as colored gaussian noise.
        The dimensions of this function are the square
        of the dimensions of an integrated potential.
    """

    variance_function: FourierOperatorLike

    def __init__(self, variance_function: FourierOperatorLike):
        self.variance_function = variance_function

    @override
    def sample_fourier_phase_shifts_from_ice(
        self, key: PRNGKeyArray, instrument_config: InstrumentConfig
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Sample a realization of the ice phase shifts as colored gaussian noise."""
        N_pix = np.prod(instrument_config.padded_shape)
        frequency_grid_in_angstroms = instrument_config.padded_frequency_grid_in_angstroms
        # Compute standard deviation, scaling up by the variance by the number
        # of pixels to make the realization independent pixel-independent in real-space.
        std = jnp.sqrt(N_pix * self.variance_function(frequency_grid_in_angstroms))
        ice_integrated_potential_at_exit_plane = std * jr.normal(
            key,
            shape=frequency_grid_in_angstroms.shape[0:-1],
            dtype=complex,
        ).at[0, 0].set(0.0)
        ice_phase_shifts_at_exit_plane = compute_phase_shifts_from_integrated_potential(
            ice_integrated_potential_at_exit_plane,
            instrument_config.wavelength_in_angstroms,
        )

        return ice_phase_shifts_at_exit_plane
