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


class UniformPhaseIce(AbstractIce, strict=True):
    r"""Ice modeled as uniform phase noise.

    **Attributes:**

    - `variance_function` :
        Power envelope -- ParkhurstGaussian
        Phase -- uniform from 0 to 2pi

        A function that computes the variance
        of the ice, modeled as colored gaussian noise.
        The dimensions of this function are the square
        of the dimensions of an integrated potential.
    """

    power_envelope_function: FourierOperatorLike

    def __init__(self, power_envelope_function: FourierOperatorLike):
        self.power_envelope_function = power_envelope_function

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
        # Compute variance, scaling up by the variance by the number
        # of pixels to make the realization independent pixel-independent in real-space.
        power_envelope = n_pixels * self.power_envelope_function(
            frequency_grid_in_angstroms
        )

        phase = (
            2
            * jnp.pi
            * jr.uniform(
                key,
                shape=frequency_grid_in_angstroms.shape[0:-1],
                # dtype=complex,
            )
            .at[0, 0]
            .set(0.0)
        )

        ice_integrated_potential_at_exit_plane = jnp.sqrt(power_envelope) * jnp.exp(
            1j * phase
        )

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


class Parkhurst2024_ExperimentalIce2(AbstractIce, strict=True):
    r"""Continuum model for ice from Parkhurst et al. (2024).

    **Attributes:**

    - 'N' :
        Number of water molecules per unit area in units of inverse length squared.
        Default value is 1.0 Å⁻²

    - `power_envelope_function` :
        A function that computes the variance
        of the ice, modeled as colored gaussian noise.
        The dimensions of this function are the square
        of the dimensions of an integrated potential.
        Defaults to Parkhurst2024_Gaussian.

    - 'mean_potential' :
        Mean potential of the ice in units of inverse length squared.
        Computed as U = 4*pi*N*(f_e^8(0) + 2*f_e^1(0)) where
        f_e^8 = 0.0974 + 0.2921 + 0.691  + 0.699  + 0.2039 = 1.9834
        f_e^1 = 0.0349 + 0.1201 + 0.197  + 0.0573 + 0.1195 = 0.5288
        are the sum of the Peng scattering factor a-values

    """

    N: Float[Array, ""] = field(default=1.0, converter=error_if_negative)
    mean_potential: Float[Array, ""] = field(init=False)
    power_envelope_function: FourierOperatorLike = field(init=False)

    def __post_init__(self):
        self.power_envelope_function = Parkhurst2024_Gaussian(N=self.N)
        self.mean_potential = 4 * jnp.pi * self.N * (1.9834 + 2 * 0.5288)

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

        # Compute variance, scaling up by the number of pixels to make
        # the realization independent pixel-independent in real-space.
        power_envelope = n_pixels * self.power_envelope_function(
            frequency_grid_in_angstroms
        )

        # Compute phase from a uniform random distribution of [0, 2pi]
        phase = (
            2
            * jnp.pi
            * jr.uniform(key, shape=frequency_grid_in_angstroms.shape[0:-1])
            .at[0, 0]
            .set(0.0)
        )

        # Multiply standard deviation and phase shifts together as a complex number
        # C = A*exp(i*phi)
        ice_integrated_potential_at_exit_plane = jnp.sqrt(power_envelope) * jnp.exp(
            1j * phase
        )

        # Add dc component (the expected/mean potential)
        # TODO: Should I be adding or setting the mean potential?
        # TODO: Should I be multiplying by n_pixels?
        # TODO: Do the units of the mean potential need to be converted?
        # I believe, so since they are in inverse length squared.
        ice_integrated_potential_at_exit_plane = (
            ice_integrated_potential_at_exit_plane.at[0, 0].set(
                n_pixels * self.mean_potential
            )
        )

        # Convert units of integrated potential to phase shifts
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
    This operator represents the sum of two Gaussians.

    .. math::
        P(k) = a_1 \exp(-(k-m_1)^2/(2 s_1^2)) + a_2 \exp(-(k-m_2)^2/(2 s_2^2)),

    where:
    - :math:`a_1` and :math:`a_2` are the amplitudes
    - :math:`s_1` and :math:`s_2` are the Gaussian widths in Å⁻¹,
    - :math:`m_1` and :math:`m_2` are the centers in Å⁻¹.

    Default values given by Parkhurst et al. (2024) are:
    a_1 = 0.199
    s_1 = 0.731
    m_1 = 0
    a_2 = 0.801
    s_2 = 0.081
    m_2 = 1/2.88 (Å^(-1))

    The number of water molecules per unit area, `N` (in Å⁻²) is used to
    scale `a_1` and `a_2`so that the total variance squared is 10195.82 N.
    a_1 = 0.199 * N
    a_2 = 0.801 * N
    """

    N: float = field(default=1.0, converter=error_if_negative)
    a1: float = field(default=0.199, converter=jnp.asarray)
    s1: float = field(default=0.731, converter=error_if_negative)
    m1: float = field(default=0.0, converter=error_if_negative)
    a2: float = field(default=0.801, converter=jnp.asarray)
    s2: float = field(default=0.081, converter=error_if_negative)
    m2: float = field(default=1 / 2.88, converter=error_if_negative)

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
    ) -> Float[Array, "..."]:
        """
        Evaluate the 2-Gaussian function at each pixel in `frequency_grid`, scaling
        amplitudes by `N`. If no arguments are passed, the default values from
        Parkhurst et al. (2024) are used.
        """

        # Compute the magnitude of the radial frequency vector
        # k = sqrt(kx^2 + ky^2 [+ kz^2]).
        k_sqr = jnp.sum(frequency_grid**2, axis=-1)
        k = jnp.sqrt(k_sqr)

        # Rescale the base amplitudes by N
        a1_scaled = self.a1 * self.N  # Amplitude of the first Gaussian
        a2_scaled = self.a2 * self.N  # Amplitude of the second Gaussian

        # Power spectrum formula
        scaling = a1_scaled * jnp.exp(
            -0.5 * ((k - self.m1) / self.s1) ** 2
        ) + a2_scaled * jnp.exp(-0.5 * ((k - self.m2) / self.s2) ** 2)

        return scaling


Parkhurst2024_Gaussian.__init__.__doc__ = """**Arguments:**
- `a1`: The amplitude of the first gaussian
- `s1`: The variance of the first gaussian
- `m1`: The center of the first gaussian
- `a2`: The amplitude of the second gaussian
- `s2`: The variance of the second gaussian
- `m2`: The center of the second gaussian
"""
