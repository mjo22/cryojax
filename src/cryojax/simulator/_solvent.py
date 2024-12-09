"""
Abstraction of the ice in a cryo-EM image.
"""

from abc import abstractmethod
from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
from equinox import Module
from jaxtyping import Array, Complex, PRNGKeyArray

from ..image import fftn, irfftn
from ..image.operators import FourierOperatorLike
from ._instrument_config import InstrumentConfig
from ._scattering_theory import convert_units_of_integrated_potential


class AbstractIce(Module, strict=True):
    """Base class for an ice model."""

    @abstractmethod
    def sample_ice_spectrum(
        self,
        key: PRNGKeyArray,
        instrument_config: InstrumentConfig,
        apply_hermitian_symmetry: bool = True,
    ) -> (
        Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}",
        ]
        | Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}",
        ]
    ):
        """Sample a stochastic realization of the phase shifts due to the ice
        at the exit plane."""
        raise NotImplementedError

    def compute_object_spectrum_with_ice(
        self,
        key: PRNGKeyArray,
        object_spectrum_at_exit_plane: (
            Complex[
                Array,
                "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}",
            ]
            | Complex[
                Array,
                "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}",
            ]
        ),
        instrument_config: InstrumentConfig,
        is_hermitian_symmetric: bool = True,
    ) -> (
        Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}",
        ]
        | Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}",
        ]
    ):
        """Compute the combined spectrum of the ice and the specimen."""
        # Sample the realization of the phase due to the ice.
        ice_spectrum_at_exit_plane = self.sample_ice_spectrum(
            key, instrument_config, apply_hermitian_symmetry=is_hermitian_symmetric
        )

        return object_spectrum_at_exit_plane + ice_spectrum_at_exit_plane


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
    def sample_ice_spectrum(
        self,
        key: PRNGKeyArray,
        instrument_config: InstrumentConfig,
        apply_hermitian_symmetry: bool = True,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Sample a realization of the ice phase shifts as colored gaussian noise."""
        n_pixels = instrument_config.padded_n_pixels
        frequency_grid_in_angstroms = instrument_config.padded_frequency_grid_in_angstroms
        # Compute standard deviation, scaling up by the variance by the number
        # of pixels to make the realization independent pixel-independent in real-space.
        std = jnp.sqrt(n_pixels * self.variance_function(frequency_grid_in_angstroms))
        ice_integrated_potential_at_exit_plane = std * jr.normal(
            key,
            shape=frequency_grid_in_angstroms.shape[0:-1],
            dtype=complex,
        ).at[0, 0].set(0.0)
        ice_spectrum_at_exit_plane = convert_units_of_integrated_potential(
            ice_integrated_potential_at_exit_plane,
            instrument_config.wavelength_in_angstroms,
        )

        if apply_hermitian_symmetry:
            return ice_spectrum_at_exit_plane
        else:
            return fftn(
                irfftn(ice_spectrum_at_exit_plane, s=instrument_config.padded_shape)
            )
