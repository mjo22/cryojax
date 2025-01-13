from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...constants import convert_keV_to_angstroms
from ...internal import error_if_negative, error_if_not_fractional, error_if_not_positive
from .._instrument_config import InstrumentConfig
from .base_transfer_theory import AbstractTransferFunction, AbstractTransferTheory
from .common_functions import compute_phase_shifts


class WaveTransferFunction(AbstractTransferFunction, strict=True):
    """Compute an astigmatic lens transfer function
    (referred to as the "wave transfer function") with aspherical aberration correction.

    **References:**

    - For the definition of the wave transfer function, see Chapter 65, Page 1697
      from *Hawkes, Peter W., and Erwin Kasper. Principles of Electron
      Optics, Volume 3: Fundamental Wave Optics. Academic Press, 2022.*
    """

    defocus_in_angstroms: Float[Array, ""]
    astigmatism_in_angstroms: Float[Array, ""]
    astigmatism_angle: Float[Array, ""]
    spherical_aberration_in_mm: Float[Array, ""]
    amplitude_contrast_ratio: Float[Array, ""]
    phase_shift: Float[Array, ""]

    def __init__(
        self,
        defocus_in_angstroms: float | Float[Array, ""] = 10000.0,
        astigmatism_in_angstroms: float | Float[Array, ""] = 0.0,
        astigmatism_angle: float | Float[Array, ""] = 0.0,
        spherical_aberration_in_mm: float | Float[Array, ""] = 2.7,
        amplitude_contrast_ratio: float | Float[Array, ""] = 0.1,
        phase_shift: float | Float[Array, ""] = 0.0,
    ):
        """**Arguments:**

        - `defocus_u_in_angstroms`: The major axis defocus in Angstroms.
        - `defocus_v_in_angstroms`: The minor axis defocus in Angstroms.
        - `astigmatism_angle`: The defocus angle.
        - `spherical_aberration_in_mm`: The spherical aberration coefficient in mm.
        - `amplitude_contrast_ratio`: The amplitude contrast ratio.
        - `phase_shift`: The additional phase shift.
        """
        self.defocus_in_angstroms = error_if_not_positive(defocus_in_angstroms)
        self.astigmatism_in_angstroms = jnp.asarray(astigmatism_in_angstroms)
        self.astigmatism_angle = jnp.asarray(astigmatism_angle)
        self.spherical_aberration_in_mm = error_if_negative(spherical_aberration_in_mm)
        self.amplitude_contrast_ratio = error_if_not_fractional(amplitude_contrast_ratio)
        self.phase_shift = jnp.asarray(phase_shift)

    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        *,
        voltage_in_kilovolts: Float[Array, ""] | float = 300.0,
    ) -> Complex[Array, "y_dim x_dim"]:
        # Convert degrees to radians
        phase_shift = jnp.deg2rad(self.phase_shift)
        astigmatism_angle = jnp.deg2rad(self.astigmatism_angle)
        # Convert spherical abberation coefficient to angstroms
        spherical_aberration_in_angstroms = self.spherical_aberration_in_mm * 1e7
        # Get the wavelength
        wavelength_in_angstroms = convert_keV_to_angstroms(
            jnp.asarray(voltage_in_kilovolts)
        )
        defocus_axis_1_in_angstroms = (
            self.defocus_in_angstroms + self.astigmatism_in_angstroms / 2
        )
        defocus_axis_2_in_angstroms = (
            self.defocus_in_angstroms - self.astigmatism_in_angstroms / 2
        )
        # Compute phase shifts for CTF
        phase_shifts = compute_phase_shifts(
            frequency_grid_in_angstroms,
            defocus_axis_1_in_angstroms,
            defocus_axis_2_in_angstroms,
            astigmatism_angle,
            wavelength_in_angstroms,
            spherical_aberration_in_angstroms,
            jnp.asarray(0.0),
        )
        amplitude_contrast_phase_shifts = jnp.arctan(
            self.amplitude_contrast_ratio
            / jnp.sqrt(1.0 - self.amplitude_contrast_ratio**2)
        )
        # Compute the WTF, correcting for the amplitude contrast and additional phase
        # shift in the zero mode
        return jnp.exp(
            -1.0j
            * phase_shifts.at[0, 0].add(amplitude_contrast_phase_shifts + phase_shift)
        )


class WaveTransferTheory(AbstractTransferTheory, strict=True):
    """An optics model that propagates the exit wave to the detector plane."""

    wtf: WaveTransferFunction

    def __init__(
        self,
        wtf: WaveTransferFunction,
    ):
        """**Arguments:**

        - `wtf`: The wave transfer function model.
        """

        self.wtf = wtf

    @override
    def __call__(
        self,
        wavefunction_spectrum_at_exit_plane: Complex[
            Array,
            "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}",
        ],
        instrument_config: InstrumentConfig,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        """Apply the wave transfer function to the wavefunction in the exit plane."""
        frequency_grid = instrument_config.padded_full_frequency_grid_in_angstroms
        # Compute the wave transfer function
        wtf_array = self.wtf(
            frequency_grid,
            voltage_in_kilovolts=instrument_config.voltage_in_kilovolts,
        )
        # ... compute the contrast as the CTF multiplied by the exit plane
        # phase shifts
        wavefunction_spectrum_at_detector_plane = (
            wtf_array * wavefunction_spectrum_at_exit_plane
        )

        return wavefunction_spectrum_at_detector_plane
