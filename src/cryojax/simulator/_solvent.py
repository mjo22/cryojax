"""
Abstraction of the ice in a cryo-EM image.
"""

from abc import abstractmethod
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..constants import PARKHURST2024_POWER_CONSTANTS
from ..image import ifftn, irfftn, rescale_image
from ..image.operators import (
    AbstractFourierOperator,
    FourierGaussian,
    FourierGaussianWithRadialOffset,
    FourierOperatorLike,
)
from ..internal import error_if_negative
from ._instrument_config import InstrumentConfig


class SolventMixturePower(AbstractFourierOperator, strict=True):
    r"""A model for the power of the gaussian random field (GRF) for solvent in cryo-EM.
    This implementation takes inspiration from Parkhurst et al. (2024) and uses its
    implementation as default.

    Parkhurst et al. (2024) models the power as a sum of two gaussians, with
    one gaussian modeling an envelope for the low-resolution decay of the power and
    another gaussian modeling the peak from the solvent at high-resolutions. This
    expression is given by

    .. math::
        P(k) = a_1 \exp(-k^2/(2 s_1^2)) + a_2 \exp(-(k-m)^2/(2 s_2^2)),

    where index `1` is for the envelope and index `2` is for the high-resolution peak.

    More generally, this class models the power as a mixture of any two functions, called
    `envelope_function` and `peak_function`.
    """

    envelope_function: FourierOperatorLike
    peak_function: FourierOperatorLike

    def __init__(
        self,
        envelope_function: FourierOperatorLike | None = None,
        peak_function: FourierOperatorLike | None = None,
    ):
        if envelope_function is None:
            self.envelope_function = FourierGaussian(
                amplitude=PARKHURST2024_POWER_CONSTANTS.a_1,
                b_factor=2 / PARKHURST2024_POWER_CONSTANTS.s_1**2,
            )
        if peak_function is None:
            self.peak_function = FourierGaussianWithRadialOffset(
                amplitude=PARKHURST2024_POWER_CONSTANTS.a_2,
                b_factor=2 / PARKHURST2024_POWER_CONSTANTS.s_2**2,
                offset=PARKHURST2024_POWER_CONSTANTS.m,
            )

    @override
    def __call__(
        self, frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]:
        mixture_function = self.envelope_function * self.peak_function
        return mixture_function(frequency_grid_in_angstroms)


class AbstractSolvent(eqx.Module, strict=True):
    """Base class for a model of the solvent in cryo-EM."""

    thickness_in_angstroms: eqx.AbstractVar[Float[Array, ""]]

    @abstractmethod
    def sample_solvent_integrated_potential(
        self,
        rng_key: PRNGKeyArray,
        instrument_config: InstrumentConfig,
        outputs_rfft: bool = True,
        outputs_real_space: bool = False,
    ) -> (
        Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}",
        ]
        | Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}",
        ]
        | Float[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}",
        ]
    ):
        """Sample a stochastic realization of scattering potential due to the ice
        at the exit plane."""
        raise NotImplementedError

    def compute_integrated_potential_with_solvent(
        self,
        rng_key: PRNGKeyArray,
        fourier_integrated_potential_of_specimen: (
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
        input_is_rfft: bool = True,
        outputs_real_space: bool = False,
    ) -> (
        Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}",
        ]
        | Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}",
        ]
        | Float[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}",
        ]
    ):
        """Compute the combined spectrum of the ice and the specimen."""
        # Sample the realization of the phase due to the ice.
        # TODO: this function will also handle masking from a model for the
        # solvent shell.
        fourier_integrated_potential_of_solvent = (
            self.sample_solvent_integrated_potential(
                rng_key,
                instrument_config,
                outputs_rfft=input_is_rfft,
                outputs_real_space=False,
            )
        )
        fourier_integrated_potential = (
            fourier_integrated_potential_of_solvent
            + fourier_integrated_potential_of_specimen
        )

        if outputs_real_space:
            if input_is_rfft:
                return irfftn(
                    fourier_integrated_potential,
                    s=instrument_config.padded_shape,
                )
            else:
                return ifftn(
                    fourier_integrated_potential, s=instrument_config.padded_shape
                )
        else:
            return fourier_integrated_potential


class GRFSolvent(AbstractSolvent, strict=True):
    r"""Solvent modeled as a gaussian random field (GRF)."""

    thickness_in_angstroms: Float[Array, ""]
    molecules_per_angstrom_cubed: float
    total_potential_per_molecule: float
    power_spectrum_function: FourierOperatorLike
    samples_power: bool

    def __init__(
        self,
        thickness_in_angstroms: Float[Array, ""] | float,
        molecules_per_angstrom_cubed: float = 0.0325,
        total_potential_per_molecule: float = 38.214333,
        power_spectrum_function: FourierOperatorLike | None = None,
        samples_power: bool = False,
    ):
        """**Arguments:**

        - `thickness_in_angstroms`:
            The solvent thickness in angstroms.
        - `potential_per_molecule`:
            A dimensional factor that quantifies the total electrostatic
            potential per molecule. This has units of energy times volume,
            which in `cryojax` conventions is measured in Angstroms.
            By default, this is calibrated as the q = 0 component of
            an electron scattering factor for water, computed from
            `cryojax.constants`.
        - `molecules_per_angstrom_cubed`:
            A dimensional factor that quantifies the number of water molecules
            per unit volume. By default, this is calibrated from MD simulations
            of water.
        - `power_spectrum_function` :
            A function that computes the power spectrum of the solvent, up to the
            final mean and standard deviation of the gaussian random field.
        - `samples_power`:
            If `True`, the power spectrum of the sampled GRF is stochastic. If `False`,
            the power is deterministic (i.e. the GRF is generated by sampling
            uniform phases).
        """
        self.power_spectrum_function = power_spectrum_function or SolventMixturePower()
        self.samples_power = samples_power
        self.thickness_in_angstroms = error_if_negative(
            jnp.asarray(thickness_in_angstroms)
        )
        self.molecules_per_angstrom_cubed = molecules_per_angstrom_cubed
        self.total_potential_per_molecule = total_potential_per_molecule

    @override
    def sample_solvent_integrated_potential(
        self,
        rng_key: PRNGKeyArray,
        instrument_config: InstrumentConfig,
        outputs_rfft: bool = True,
        outputs_real_space: bool = False,
    ) -> (
        Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}",
        ]
        | Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}",
        ]
        | Float[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}",
        ]
    ):
        """Sample a realization of the ice integrated potential as a gaussian
        random field.
        """
        n_pixels = instrument_config.padded_n_pixels
        if outputs_real_space:
            frequency_grid_in_angstroms = (
                instrument_config.padded_frequency_grid_in_angstroms
            )
        else:
            if outputs_rfft:
                frequency_grid_in_angstroms = (
                    instrument_config.padded_frequency_grid_in_angstroms
                )
            else:
                frequency_grid_in_angstroms = (
                    instrument_config.padded_full_frequency_grid_in_angstroms
                )
        # Compute standard deviation, scaling up by the variance by the number
        # of pixels to make the realization independent pixel-independent in real-space.
        std = jnp.sqrt(
            n_pixels * self.power_spectrum_function(frequency_grid_in_angstroms)
        )
        solvent_grf = std * jr.normal(
            rng_key,
            shape=frequency_grid_in_angstroms.shape[0:-1],
            dtype=complex,
        ).at[0, 0].set(0.0)
        # Scale to desired mean and standard deviation
        molecules_per_angstrom_squared = (
            self.thickness_in_angstroms * self.molecules_per_angstrom_cubed
        )
        mean = self.total_potential_per_molecule * molecules_per_angstrom_squared
        std = 1.0  # TODO: set standard deviation
        fourier_integrated_potential_of_solvent = rescale_image(
            solvent_grf,
            mean=mean,
            std=std,
            input_is_real_space=False,
            input_is_rfft=outputs_rfft,
            shape_in_real_space=instrument_config.padded_shape,
        )
        if outputs_real_space:
            return irfftn(
                fourier_integrated_potential_of_solvent,
                s=instrument_config.padded_shape,
            )
        else:
            return fourier_integrated_potential_of_solvent
