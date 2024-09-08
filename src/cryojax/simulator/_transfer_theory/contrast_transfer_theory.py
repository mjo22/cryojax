from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
from equinox import field
from jaxtyping import Array, Complex, Float

from ..._errors import error_if_negative, error_if_not_fractional
from ...constants import convert_keV_to_angstroms
from ...image.operators import (
    Constant,
    FourierOperatorLike,
)
from .._instrument_config import InstrumentConfig
from .base_transfer_theory import AbstractTransferFunction, AbstractTransferTheory
from .common_functions import compute_phase_shifts_with_amplitude_contrast_ratio


class ContrastTransferFunction(AbstractTransferFunction, strict=True):
    """Compute an astigmatic Contrast Transfer Function (CTF) with a
    spherical aberration correction and amplitude contrast ratio.

    !!! info
        `cryojax` uses a convention different from CTFFIND for
        astigmatism parameters. It returns defocus major and minor
        axes, called "defocus1" and "defocus2". In order to convert
        from CTFFIND to `cryojax`,

        ```python
        defocus1, defocus2 = ... # Read from CTFFIND
        ctf = ContrastTransferFunction(
            defocus_in_angstroms=(defocus1+defocus2)/2,
            astigmatism_in_angstroms=defocus1-defocus2,
            ...
        )
        ```
    """

    defocus_in_angstroms: Float[Array, ""]
    astigmatism_in_angstroms: Float[Array, ""]
    astigmatism_angle: Float[Array, ""]
    voltage_in_kilovolts: Float[Array, ""] | float = field(static=True)
    spherical_aberration_in_mm: Float[Array, ""]
    amplitude_contrast_ratio: Float[Array, ""]
    phase_shift: Float[Array, ""]

    def __init__(
        self,
        defocus_in_angstroms: float | Float[Array, ""] = 10000.0,
        astigmatism_in_angstroms: float | Float[Array, ""] = 0.0,
        astigmatism_angle: float | Float[Array, ""] = 0.0,
        voltage_in_kilovolts: float | Float[Array, ""] = 300.0,
        spherical_aberration_in_mm: float | Float[Array, ""] = 2.7,
        amplitude_contrast_ratio: float | Float[Array, ""] = 0.1,
        phase_shift: float | Float[Array, ""] = 0.0,
    ):
        """**Arguments:**

        - `defocus_in_angstroms`: The mean defocus in Angstroms.
        - `astigmatism_in_angstroms`: The amount of astigmatism in Angstroms.
        - `astigmatism_angle`: The defocus angle.
        - `voltage_in_kilovolts`:
            The accelerating voltage in kV. This field is treated as *static*, i.e.
            as part of the pytree. This is because the accelerating voltage is treated
            as a traced value in the `InstrumentConfig`, since many modeling components
            are interested in the accelerating voltage.
        - `spherical_aberration_in_mm`: The spherical aberration coefficient in mm.
        - `amplitude_contrast_ratio`: The amplitude contrast ratio.
        - `phase_shift`: The additional phase shift.
        """
        self.defocus_in_angstroms = error_if_negative(defocus_in_angstroms)
        self.astigmatism_in_angstroms = jnp.asarray(astigmatism_in_angstroms)
        self.astigmatism_angle = jnp.asarray(astigmatism_angle)
        self.voltage_in_kilovolts = voltage_in_kilovolts
        self.spherical_aberration_in_mm = error_if_negative(spherical_aberration_in_mm)
        self.amplitude_contrast_ratio = error_if_not_fractional(amplitude_contrast_ratio)
        self.phase_shift = jnp.asarray(phase_shift)

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
        defocus_axis_1_in_angstroms = (
            self.defocus_in_angstroms
            + jnp.asarray(defocus_offset)
            + self.astigmatism_in_angstroms / 2
        )
        defocus_axis_2_in_angstroms = (
            self.defocus_in_angstroms
            + jnp.asarray(defocus_offset)
            - self.astigmatism_in_angstroms / 2
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


class ContrastTransferTheory(AbstractTransferTheory, strict=True):
    """An optics model in the weak-phase approximation. Here, compute the image
    contrast by applying the CTF directly to the exit plane phase shifts.
    """

    ctf: ContrastTransferFunction
    envelope: FourierOperatorLike

    def __init__(
        self,
        ctf: ContrastTransferFunction,
        envelope: Optional[FourierOperatorLike] = None,
    ):
        """**Arguments:**

        - `ctf`: The contrast transfer function model.
        - `envelope`: The envelope function of the optics model.
        """

        self.ctf = ctf
        self.envelope = envelope or Constant(jnp.asarray(1.0))

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
        frequency_grid = instrument_config.padded_frequency_grid_in_angstroms
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
