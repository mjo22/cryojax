from abc import abstractmethod
from typing_extensions import override

import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array, Complex, Float

from ...constants import convert_keV_to_angstroms
from ...internal import error_if_negative, error_if_not_fractional
from .common_functions import (
    compute_phase_shift_from_amplitude_contrast_ratio,
    compute_phase_shifts_with_spherical_aberration,
)


class AbstractTransferFunction(Module, strict=True):
    """An abstract base class for a CTF in cryo-EM."""

    @abstractmethod
    def compute_aberration_phase_shifts(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        voltage_in_kilovolts: Float[Array, ""] | float,
    ) -> Float[Array, "y_dim x_dim"]:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        voltage_in_kilovolts: Float[Array, ""] | float,
        is_weak_phase_approximation: bool = True,
    ) -> Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"]:
        raise NotImplementedError


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

    def compute_aberration_phase_shifts(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        voltage_in_kilovolts: Float[Array, ""] | float,
    ) -> Float[Array, "y_dim x_dim"]:
        """Compute the frequency-dependent phase shifts due to wave aberration.

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
        astigmatism_angle = jnp.deg2rad(self.astigmatism_angle)
        # Convert spherical abberation coefficient to angstroms
        spherical_aberration_in_angstroms = self.spherical_aberration_in_mm * 1e7
        # Get the wavelength
        wavelength_in_angstroms = convert_keV_to_angstroms(
            jnp.asarray(voltage_in_kilovolts)
        )
        # Compute phase shifts for CTF
        phase_shifts = compute_phase_shifts_with_spherical_aberration(
            frequency_grid_in_angstroms,
            self.defocus_in_angstroms,
            self.astigmatism_in_angstroms,
            astigmatism_angle,
            wavelength_in_angstroms,
            spherical_aberration_in_angstroms,
        )
        return phase_shifts

    @override
    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        voltage_in_kilovolts: Float[Array, ""] | float,
        is_weak_phase_approximation: bool = True,
    ) -> Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"]:
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
        # Convert degrees to radians
        aberration_phase_shifts = self.compute_aberration_phase_shifts(
            frequency_grid_in_angstroms, voltage_in_kilovolts=voltage_in_kilovolts
        )
        # Additional phase shifts
        phase_shift = jnp.deg2rad(self.phase_shift)
        amplitude_contrast_phase_shift = (
            compute_phase_shift_from_amplitude_contrast_ratio(
                self.amplitude_contrast_ratio
            )
        )
        if is_weak_phase_approximation:
            # Compute the CTF
            return jnp.sin(
                aberration_phase_shifts - (phase_shift + amplitude_contrast_phase_shift)
            )
        else:
            # Compute the "complex CTF", correcting for the amplitude contrast
            # and additional phase shift in the zero mode
            return jnp.exp(
                -1.0j
                * aberration_phase_shifts.at[0, 0].add(
                    phase_shift + amplitude_contrast_phase_shift
                )
            )


class IdealTransferFunction(AbstractTransferFunction, strict=True):
    """A perfect transfer function, useful for imaging
    cryo-EM densities."""

    @override
    def compute_aberration_phase_shifts(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        voltage_in_kilovolts: Float[Array, ""] | float,
    ) -> Float[Array, "y_dim x_dim"]:
        shape = frequency_grid_in_angstroms.shape[:2]
        return jnp.zeros(shape)

    @override
    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        voltage_in_kilovolts: Float[Array, ""] | float,
        is_weak_phase_approximation: bool = True,
    ) -> Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"]:
        shape = frequency_grid_in_angstroms.shape[:2]
        if is_weak_phase_approximation:
            return jnp.ones(shape, dtype=float)
        else:
            return jnp.ones(shape, dtype=complex)
