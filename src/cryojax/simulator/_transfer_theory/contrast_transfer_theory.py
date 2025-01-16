from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...constants import convert_keV_to_angstroms
from ...image.operators import FourierOperatorLike
from ...internal import error_if_negative, error_if_not_fractional
from .._instrument_config import InstrumentConfig
from .base_transfer_theory import AbstractTransferFunction
from .common_functions import (
    compute_phase_shift_from_amplitude_contrast_ratio,
    compute_phase_shifts_with_astigmatism_and_spherical_aberration,
)


class ContrastTransferFunction(AbstractTransferFunction, strict=True):
    """Compute an astigmatic Contrast Transfer Function (CTF) with a
    spherical aberration correction and amplitude contrast ratio.

    !!! info
        `cryojax` uses a convention different from CTFFIND for
        astigmatism parameters. CTFFIND returns defocus major and minor
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

        - `defocus_in_angstroms`: The mean defocus in Angstroms.
        - `astigmatism_in_angstroms`: The amount of astigmatism in Angstroms.
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

    @override
    def compute_phase_shifts_from_instrument(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        *,
        voltage_in_kilovolts: Float[Array, ""] | float = 300.0,
    ) -> Float[Array, "y_dim x_dim"]:
        """Compute the frequency-dependent phase shifts due to the instrument.

        This is often denoted as $\\chi(\\boldsymbol{q})$ for the in-plane
        spatial frequency $\\boldsymbol{q}$.

        **Arguments:**

        - `frequency_grid_in_angstroms`:
            The grid of frequencies in units of inverse angstroms. This can
            be computed with [`cryojax.coordinates.make_frequency_grid`](https://mjo22.github.io/cryojax/api/coordinates/making_coordinates/#cryojax.coordinates.make_frequency_grid)
        - `voltage_in_kilovolts`:
            The accelerating voltage of the microscope in kilovolts. This
            is converted to the wavelength of incident electrons using
            the function [`cryojax.constants.convert_keV_to_angstroms`](https://mjo22.github.io/cryojax/api/constants/units/#cryojax.constants.convert_keV_to_angstroms)
        """
        # Convert degrees to radians
        phase_shift = jnp.deg2rad(self.phase_shift)
        astigmatism_angle = jnp.deg2rad(self.astigmatism_angle)
        # Convert spherical abberation coefficient to angstroms
        spherical_aberration_in_angstroms = self.spherical_aberration_in_mm * 1e7
        # Get the wavelength
        wavelength_in_angstroms = convert_keV_to_angstroms(
            jnp.asarray(voltage_in_kilovolts)
        )
        # Compute phase shifts for CTF
        phase_shifts = compute_phase_shifts_with_astigmatism_and_spherical_aberration(
            frequency_grid_in_angstroms,
            self.defocus_in_angstroms,
            self.astigmatism_in_angstroms,
            astigmatism_angle,
            wavelength_in_angstroms,
            spherical_aberration_in_angstroms,
        )
        return phase_shifts - phase_shift

    @override
    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        *,
        voltage_in_kilovolts: Float[Array, ""] | float = 300.0,
    ) -> Float[Array, "y_dim x_dim"]:
        """Compute the CTF as a JAX array.

        **Arguments:**

        - `frequency_grid_in_angstroms`:
            The grid of frequencies in units of inverse angstroms. This can
            be computed with [`cryojax.coordinates.make_frequency_grid`](https://mjo22.github.io/cryojax/api/coordinates/making_coordinates/#cryojax.coordinates.make_frequency_grid)
        - `voltage_in_kilovolts`:
            The accelerating voltage of the microscope in kilovolts. This
            is converted to the wavelength of incident electrons using
            the function [`cryojax.constants.convert_keV_to_angstroms`](https://mjo22.github.io/cryojax/api/constants/units/#cryojax.constants.convert_keV_to_angstroms)
        """  # noqa: E501
        phase_shifts = self.compute_phase_shifts_from_instrument(
            frequency_grid_in_angstroms, voltage_in_kilovolts=voltage_in_kilovolts
        )
        phase_shifts -= compute_phase_shift_from_amplitude_contrast_ratio(
            self.amplitude_contrast_ratio
        )
        # Compute the CTF
        return jnp.sin(phase_shifts).at[0, 0].set(0.0)


class ContrastTransferTheory(eqx.Module, strict=True):
    """A transfer theory for the weak-phase approximation. This class
    propagates the fourier spectrum of the object from a plane directly below it to
    the plane of the detector. In other terms, it computes a noiseless cryo-EM
    image from a 2D projection.
    """

    ctf: ContrastTransferFunction
    envelope: Optional[FourierOperatorLike]

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
        self.envelope = envelope

    def propagate_object_to_detector_plane(
        self,
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
        *,
        is_projection_approximation: bool = True,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Apply the CTF directly to the phase shifts in the exit plane.

        **Arguments:**

        - `object_spectrum_at_exit_plane`:
            The fourier spectrum of the object in a plane directly below it.
        - `instrument_config`:
            The configuration of the resulting image.
        - `is_projection_approximation`:
            If `True`, the `object_spectrum_in_exit_plane` is a projection
            approximation and is therefore the fourier transform of a real-valued
            array. If `False`, `object_spectrum_in_exit_plane` is extracted from
            the ewald sphere and is therefore the fourier transform of a complex-valued
            array.
        """
        frequency_grid = instrument_config.padded_frequency_grid_in_angstroms
        if is_projection_approximation:
            # Compute the CTF
            ctf_array = self.ctf(
                frequency_grid,
                voltage_in_kilovolts=instrument_config.voltage_in_kilovolts,
            )
            # ... compute the contrast as the CTF multiplied by the exit plane
            # phase shifts
            contrast_spectrum_at_detector_plane = (
                ctf_array * object_spectrum_at_exit_plane
            )
        else:
            # chi = self.ctf.compute_phase_shifts_from_instrument(
            #     frequency_grid,
            #     voltage_in_kilovolts=instrument_config.voltage_in_kilovolts,
            # )
            # cos_chi, sin_chi = jnp.cos(chi), jnp.sin(chi)
            raise NotImplementedError
        if self.envelope is not None:
            contrast_spectrum_at_detector_plane *= self.envelope(frequency_grid)

        return contrast_spectrum_at_detector_plane
