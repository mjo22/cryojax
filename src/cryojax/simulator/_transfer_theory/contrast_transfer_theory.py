from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
from equinox import field
from jaxtyping import Array, Complex, Float

from ..._errors import error_if_negative, error_if_not_fractional, error_if_not_positive
from ...constants import convert_keV_to_angstroms
from ...image.operators import (
    Constant,
    FourierOperatorLike,
)
from .._instrument_config import InstrumentConfig
from .base_transfer_theory import AbstractTransferFunction, AbstractTransferTheory
from .common_functions import compute_phase_shifts_with_amplitude_contrast_ratio


class AbstractContrastTransferFunction(AbstractTransferFunction, strict=True):
    """An abstract base class for a transfer function."""

    @abstractmethod
    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        *,
        wavelength_in_angstroms: Optional[Float[Array, ""] | float] = None,
        defocus_offset: Float[Array, ""] | float = 0.0,
    ) -> Float[Array, "y_dim x_dim"]:
        raise NotImplementedError


class ContrastTransferFunction(AbstractContrastTransferFunction, strict=True):
    """Compute an astigmatic Contrast Transfer Function (CTF) with a
    spherical aberration correction and amplitude contrast ratio.
    """

    defocus_in_angstroms: Float[Array, ""] = field(
        default=10000.0, converter=error_if_not_positive
    )
    astigmatism_in_angstroms: Float[Array, ""] = field(default=0.0, converter=jnp.asarray)
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
        defocus_axis_1_in_angstroms = self.defocus_in_angstroms + jnp.asarray(
            defocus_offset
        )
        defocus_axis_2_in_angstroms = (
            self.defocus_in_angstroms
            + self.astigmatism_in_angstroms
            + jnp.asarray(defocus_offset)
        )
        # Compute phase shifts for CTF
        phase_shifts = compute_phase_shifts_with_amplitude_contrast_ratio(
            frequency_grid_in_angstroms,
            defocus_axis_1_in_angstroms,
            defocus_axis_2_in_angstroms,
            astigmatism_angle,
            wavelength_in_angstroms,
            spherical_aberration_in_angstroms,
            phase_shift,
            self.amplitude_contrast_ratio,
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


class IdealContrastTransferFunction(AbstractContrastTransferFunction, strict=True):
    """Compute a perfect CTF, where frequency content is delivered equally
    over all frequencies.
    """

    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        *,
        wavelength_in_angstroms: Optional[Float[Array, ""] | float] = None,
        defocus_offset: Float[Array, ""] | float = 0.0,
    ) -> Float[Array, "y_dim x_dim"]:
        return jnp.ones(frequency_grid_in_angstroms.shape[0:2])


class ContrastTransferTheory(AbstractTransferTheory, strict=True):
    """An optics model in the weak-phase approximation. Here, compute the image
    contrast by applying the CTF directly to the exit plane phase shifts.
    """

    ctf: AbstractContrastTransferFunction
    envelope: FourierOperatorLike

    def __init__(
        self,
        ctf: AbstractContrastTransferFunction,
        envelope: Optional[FourierOperatorLike] = None,
    ):
        self.ctf = ctf
        self.envelope = envelope or Constant(1.0)

    @override
    def __call__(
        self,
        fourier_phase_at_exit_plane: Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}",
        ],
        instrument_config: InstrumentConfig,
        defocus_offset: Float[Array, ""] | float = 0.0,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Apply the CTF directly to the phase shifts in the exit plane."""
        frequency_grid = (
            instrument_config.wrapped_padded_frequency_grid_in_angstroms.get()
        )
        # Compute the CTF
        ctf_array = self.envelope(frequency_grid) * self.ctf(
            frequency_grid,
            wavelength_in_angstroms=instrument_config.wavelength_in_angstroms,
            defocus_offset=defocus_offset,
        )
        # ... compute the contrast as the CTF multiplied by the exit plane
        # phase shifts
        fourier_contrast_at_detector_plane = ctf_array * fourier_phase_at_exit_plane

        return fourier_contrast_at_detector_plane


ContrastTransferTheory.__init__.__doc__ = """**Arguments:**

- `ctf`: The contrast transfer function model.
- `envelope`: The envelope function of the optics model.
"""
