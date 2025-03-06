"""
Abstraction of the ice in a cryo-EM image.
"""

from abc import abstractmethod
from typing import overload
from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
from equinox import field, Module
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..image import fftn, irfftn
from ..image.operators import FourierOperatorLike
from ..image.operators._fourier_operator import AbstractFourierOperator
from ..internal import error_if_negative
from ._instrument_config import InstrumentConfig
from ._scattering_theory import convert_units_of_integrated_potential


class AbstractIce(Module, strict=True):
    """Base class for an ice model."""

    @abstractmethod
    def sample_ice_spectrum(
        self,
        key: PRNGKeyArray,
        instrument_config: InstrumentConfig,
        get_rfft: bool = True,
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
        is_rfft: bool = True,
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
            key, instrument_config, get_rfft=is_rfft
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
        get_rfft: bool = True,
    ) -> (
        Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}",
        ]
        | Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}",
        ]
    ):
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

        if get_rfft:
            return ice_spectrum_at_exit_plane
        else:
            return fftn(
                irfftn(ice_spectrum_at_exit_plane, s=instrument_config.padded_shape)
            )


class Parkhurst2024_Gaussian(AbstractFourierOperator, strict=True):
    r"""
    This operator represents the sum of two gaussians.
    Specifically, this is

    .. math::
        P(k) = a_1 \exp(-(k-m_1)^2/(2 s_1^2)) + a_2 \exp(-(k-m_2)^2/(2 s_2^2)),

    Where default values given by Parkhurst et al. (2024) are:
    a_1 = 0.199
    s_1 = 0.731
    m_1 = 0
    a_2 = 0.801
    s_2 = 0.081
    m_2 = 1/2.88 (Å^(-1))
    """

    a1: Float[Array, ""] = field(default=0.199, converter=jnp.asarray)
    s1: Float[Array, ""] = field(default=0.731, converter=error_if_negative)
    m1: Float[Array, ""] = field(default=0, converter=error_if_negative)

    a2: Float[Array, ""] = field(default=0.801, converter=jnp.asarray)
    s2: Float[Array, ""] = field(default=0.081, converter=error_if_negative)
    m2: Float[Array, ""] = field(default=1 / 2.88, converter=error_if_negative)

    @overload
    def __call__(
        self, frequency_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]: ...

    @overload
    def __call__(
        self, frequency_grid: Float[Array, "z_dim y_dim x_dim 3"]
    ) -> Float[Array, "z_dim y_dim x_dim"]: ...

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        # SCALE a1, a2, s1, s2 based on pixel size in

        k_sqr = jnp.sum(frequency_grid**2, axis=-1)
        k = jnp.sqrt(k_sqr)
        scaling = self.a1 * jnp.exp(
            -0.5 * ((k - self.m1) / self.s1) ** 2
        ) + self.a2 * jnp.exp(-0.5 * ((k - self.m2) / self.s2) ** 2)
        return scaling


Parkhurst2024_Gaussian.__init__.__doc__ = """**Arguments:**
- `a1`: The amplitude of the first gaussian
- `s1`: The variance of the first gaussian
- `m1`: The center of the first gaussian
- `a2`: The amplitude of the second gaussian
- `s2`: The variance of the second gaussian
- `m2`: The center of the second gaussian
"""
