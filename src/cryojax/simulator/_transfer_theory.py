"""
Models of instrument optics.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
from equinox import AbstractVar, field, Module
from jaxtyping import Array, Complex, Float

from .._errors import error_if_negative, error_if_not_fractional, error_if_not_positive
from ..constants import convert_keV_to_angstroms
from ..coordinates import cartesian_to_polar
from ..image.operators import (
    AbstractFourierOperator,
    Constant,
    FourierOperatorLike,
)
from ._config import ImageConfig


class AbstractTransferFunction(AbstractFourierOperator, strict=True):
    """An abstract base class for a transfer function."""

    @abstractmethod
    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        *,
        wavelength_in_angstroms: Optional[Float[Array, ""] | float] = None,
        defocus_offset: Float[Array, ""] | float = 0.0,
    ) -> Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"]:
        raise NotImplementedError


class ContrastTransferFunction(AbstractTransferFunction, strict=True):
    """Compute the Contrast Transfer Function (CTF) in for a weakly
    scattering specimen.
    """

    defocus_u_in_angstroms: Float[Array, ""] = field(
        default=10000.0, converter=error_if_not_positive
    )
    defocus_v_in_angstroms: Float[Array, ""] = field(
        default=10000.0, converter=error_if_not_positive
    )
    astigmatism_angle: Float[Array, ""] = field(default=0.0, converter=jnp.asarray)
    voltage_in_kilovolts: Float[Array, ""] | float = field(
        default=300.0, static=True
    )  # Mark `static=True` so that the voltage is not part of the model pytree
    # It is treated as part of the pytree upstream, in the Instrument!
    spherical_aberration_in_mm: Float[Array, ""] = field(
        default=2.7, converter=error_if_negative
    )
    amplitude_contrast_ratio: Float[Array, ""] = field(
        default=0.1, converter=error_if_not_fractional
    )
    phase_shift: Float[Array, ""] = field(default=0.0, converter=jnp.asarray)

    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        *,
        wavelength_in_angstroms: Optional[Float[Array, ""] | float] = None,
        defocus_offset: Float[Array, ""] | float = 0.0,
    ) -> Float[Array, "y_dim x_dim"]:
        # Convert degrees to radians
        phase_shift = jnp.deg2rad(self.phase_shift)
        astigmatism_angle = jnp.deg2rad(self.astigmatism_angle)
        # Convert spherical abberation coefficient to angstroms
        spherical_aberration_in_angstroms = self.spherical_aberration_in_mm * 1e7
        # Get the wavelength. It can either be passed from upstream or stored in the
        # CTF
        if wavelength_in_angstroms is None:
            wavelength_in_angstroms = convert_keV_to_angstroms(
                jnp.asarray(self.voltage_in_kilovolts)
            )
        else:
            wavelength_in_angstroms = jnp.asarray(wavelength_in_angstroms)
        # Compute phase shifts for CTF
        phase_shifts = _compute_phase_shifts(
            frequency_grid_in_angstroms,
            self.defocus_u_in_angstroms + jnp.asarray(defocus_offset),
            self.defocus_v_in_angstroms + jnp.asarray(defocus_offset),
            astigmatism_angle,
            wavelength_in_angstroms,
            spherical_aberration_in_angstroms,
            phase_shift,
            amplitude_contrast_ratio=self.amplitude_contrast_ratio,
        )
        # Compute the CTF
        return jnp.sin(phase_shifts).at[0, 0].set(0.0)


ContrastTransferFunction.__init__.__doc__ = """**Arguments:**

- `defocus_u_in_angstroms`: The major axis defocus in Angstroms.
- `defocus_v_in_angstroms`: The minor axis defocus in Angstroms.
- `astigmatism_angle`: The defocus angle.
- `voltage_in_kilovolts`: The accelerating voltage in kV.
- `spherical_aberration_in_mm`: The spherical aberration coefficient in mm.
- `amplitude_contrast_ratio`: The amplitude contrast ratio.
- `phase_shift`: The additional phase shift.
"""


class AbstractTransferTheory(Module, strict=True):
    """Base class for an optics model."""

    transfer_function: AbstractVar[AbstractTransferFunction]

    @abstractmethod
    def __call__(
        self,
        fourier_phase_or_wavefunction_in_exit_plane: (
            Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]
            | Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
        ),
        config: ImageConfig,
        wavelength_in_angstroms: Float[Array, ""] | float,
        defocus_offset: Float[Array, ""] | float = 0.0,
    ) -> (
        Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]
        | Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
    ):
        """Pass an image through the optics model."""
        raise NotImplementedError


class ContrastTransferTheory(AbstractTransferTheory, strict=True):
    """An optics model in the weak-phase approximation. Here, compute the image
    contrast by applying the CTF directly to the exit plane phase shifts.
    """

    transfer_function: ContrastTransferFunction
    envelope: FourierOperatorLike

    def __init__(
        self,
        transfer_function: ContrastTransferFunction,
        envelope: Optional[FourierOperatorLike] = None,
    ):
        self.transfer_function = transfer_function
        self.envelope = envelope or Constant(1.0)

    @override
    def __call__(
        self,
        fourier_phase_or_wavefunction_in_exit_plane: Complex[
            Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"
        ],
        config: ImageConfig,
        wavelength_in_angstroms: Float[Array, ""] | float,
        defocus_offset: Float[Array, ""] | float = 0.0,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Apply the CTF directly to the phase shifts in the exit plane."""
        fourier_phase_in_exit_plane = fourier_phase_or_wavefunction_in_exit_plane
        frequency_grid = config.wrapped_padded_frequency_grid_in_angstroms.get()
        # Compute the CTF
        ctf = self.envelope(frequency_grid) * self.transfer_function(
            frequency_grid,
            wavelength_in_angstroms=wavelength_in_angstroms,
            defocus_offset=defocus_offset,
        )
        # ... compute the contrast as the CTF multiplied by the exit plane
        # phase shifts
        fourier_contrast_in_detector_plane = ctf * fourier_phase_in_exit_plane

        return fourier_contrast_in_detector_plane


ContrastTransferTheory.__init__.__doc__ = """**Arguments:**

- `transfer_function`: The contrast transfer function model.
- `envelope`: The envelope function of the optics model.
"""


def _compute_phase_shifts(
    frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
    defocus_u_in_angstroms: Float[Array, ""],
    defocus_v_in_angstroms: Float[Array, ""],
    astigmatism_angle: Float[Array, ""],
    wavelength_in_angstroms: Float[Array, ""],
    spherical_aberration_in_angstroms: Float[Array, ""],
    phase_shift: Float[Array, ""],
    amplitude_contrast_ratio: Optional[Float[Array, ""]] = None,
) -> Float[Array, "y_dim x_dim"]:
    k_sqr, azimuth = cartesian_to_polar(frequency_grid_in_angstroms, square=True)
    defocus = 0.5 * (
        defocus_u_in_angstroms
        + defocus_v_in_angstroms
        + (defocus_u_in_angstroms - defocus_v_in_angstroms)
        * jnp.cos(2.0 * (azimuth - astigmatism_angle))
    )
    defocus_phase_shifts = -0.5 * defocus * wavelength_in_angstroms * k_sqr
    aberration_phase_shifts = (
        0.25
        * spherical_aberration_in_angstroms
        * (wavelength_in_angstroms**3)
        * (k_sqr**2)
    )
    phase_shifts = (2 * jnp.pi) * (
        defocus_phase_shifts + aberration_phase_shifts
    ) - phase_shift
    if amplitude_contrast_ratio is not None:
        amplitude_contrast_phase_shifts = jnp.arctan(
            amplitude_contrast_ratio / jnp.sqrt(1.0 - amplitude_contrast_ratio**2)
        )
        phase_shifts -= amplitude_contrast_phase_shifts

    return phase_shifts
