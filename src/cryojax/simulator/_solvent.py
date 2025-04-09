"""
Abstraction of the ice in a cryo-EM image.
"""

from abc import abstractmethod
from importlib.resources import files
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


class Parkhurst2024_ExperimentalIce(AbstractIce, strict=True):
    r"""Continuum model of ice.

    The ice is modelled as colored Gaussian noise in Fourier space
    with a variance profile matching the Parkhurst et al. (2024) model.
    The mean and variance of the real-space potential are scaled to match
    experimental values for a given voxel size and number of water
    molecules per unit area.

    **Attributes:**

    - `power_envelope_function` :
        A function that computes the variance
        of the ice, modeled as colored gaussian noise.
        The dimensions of this function are the square
        of the dimensions of an integrated potential.
        Defaults to Parkhurst2024_Gaussian.

    - 'image_mv' :
        Table of the real-space mean potential and variance
        as a function of voxel size.
        These represent the average value and spread of
        pixel intensity in the real-space image of the
        integrated potential.
        Defaults to 'image_mv__relaxed_small_box_tip3p.npy'

    - 'N_scalar' :
        This value is used to linearly scale the mean potential
        and variance of the ice.
        It is expressed as N/1468, where N is the number of
        water molecules per unit area, and 1468 is the number of
        water molecules in the relaxed_small_box_tip3p.pdb file,
        which is where the current image_mv values are precomputed
        from.
        The default value is 1.
    """

    image_mv: jnp.ndarray = field(init=False)
    power_envelope_function: FourierOperatorLike = field(init=False)
    N_scalar: float = field(default=1.0, converter=jnp.asarray)

    def __post_init__(self):
        self.image_mv = jnp.load(
            files("cryojax.simulator.data") / "image_mv__relaxed_small_box_tip3p.npy"
        )
        self.power_envelope_function = Parkhurst2024_Gaussian()

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

        # Compute variance, scaling up by the number of pixels to make
        # the realization pixel-independent in real-space.
        n_pixels = instrument_config.padded_n_pixels
        frequency_grid_in_angstroms = instrument_config.padded_frequency_grid_in_angstroms
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

        # Convert units of integrated potential to phase shifts
        ice_spectrum_at_exit_plane = convert_units_of_integrated_potential(
            ice_integrated_potential_at_exit_plane,
            instrument_config.wavelength_in_angstroms,
        )

        # Lookup precomputed real-space mean and variance for
        # nearest voxel size, scale by N_scalar
        pixel_size = instrument_config.pixel_size
        if (pixel_size < self.image_mv[0, 0]) or (pixel_size > self.image_mv[-1, 0]):
            # TODO: warn "Pixel size outside calibration range."
            pass
        _, m, v = self.image_mv[jnp.abs(self.image_mv[:, 0] - pixel_size).argmin()]
        image_mean_potential = m * self.N_scalar
        image_variance = v * self.N_scalar

        # Adjust image to match target mean and variance in real space
        image = irfftn(ice_spectrum_at_exit_plane, s=instrument_config.padded_shape)
        image = image + (image_mean_potential - jnp.mean(image))
        image = image * jnp.sqrt(image_variance / jnp.var(image))

        if get_rfft:
            return jnp.real(fftn(image))
        else:
            return fftn(image)


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
    """

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

        # Power spectrum formula
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
