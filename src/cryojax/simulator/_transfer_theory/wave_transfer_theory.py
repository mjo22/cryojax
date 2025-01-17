import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...internal import error_if_negative, error_if_not_fractional
from .._instrument_config import InstrumentConfig
from .base_transfer_theory import AbstractTransferFunction
from .common_functions import (
    compute_phase_shift_from_amplitude_contrast_ratio,
)


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
        self.defocus_in_angstroms = jnp.asarray(defocus_in_angstroms)
        self.astigmatism_in_angstroms = jnp.asarray(astigmatism_in_angstroms)
        self.astigmatism_angle = jnp.asarray(astigmatism_angle)
        self.spherical_aberration_in_mm = error_if_negative(spherical_aberration_in_mm)
        self.amplitude_contrast_ratio = error_if_not_fractional(amplitude_contrast_ratio)
        self.phase_shift = jnp.asarray(phase_shift)

    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        voltage_in_kilovolts: Float[Array, ""] | float,
    ) -> Complex[Array, "y_dim x_dim"]:
        # Compute aberration phase shifts
        aberration_phase_shifts = self.compute_aberration_phase_shifts(
            frequency_grid_in_angstroms, voltage_in_kilovolts=voltage_in_kilovolts
        )
        # Additional phase shifts only impact zero mode
        phase_shift = jnp.deg2rad(self.phase_shift)
        amplitude_contrast_phase_shift = (
            compute_phase_shift_from_amplitude_contrast_ratio(
                self.amplitude_contrast_ratio
            )
        )
        # Compute the WTF, correcting for the amplitude contrast and additional phase
        # shift in the zero mode
        return jnp.exp(
            -1.0j
            * aberration_phase_shifts.at[0, 0].add(
                phase_shift + amplitude_contrast_phase_shift
            )
        )


class WaveTransferTheory(eqx.Module, strict=True):
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

    def propagate_wavefunction_to_detector_plane(
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
