from abc import abstractmethod

import jax.numpy as jnp
from equinox import AbstractVar, Module
from jaxtyping import Array, Complex, Float

from ...constants import convert_keV_to_angstroms
from .common_functions import compute_phase_shifts_with_spherical_aberration


class AbstractTransferFunction(Module, strict=True):
    """An abstract base class for a transfer function in cryo-EM."""

    defocus_in_angstroms: AbstractVar[Float[Array, ""]]
    astigmatism_in_angstroms: AbstractVar[Float[Array, ""]]
    astigmatism_angle: AbstractVar[Float[Array, ""]]
    spherical_aberration_in_mm: AbstractVar[Float[Array, ""]]
    amplitude_contrast_ratio: AbstractVar[Float[Array, ""]]
    phase_shift: AbstractVar[Float[Array, ""]]

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

    @abstractmethod
    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        voltage_in_kilovolts: Float[Array, ""] | float,
    ) -> Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"]:
        raise NotImplementedError
