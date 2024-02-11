"""
Routines to model image formation from 3D electron density
fields.
"""

from abc import abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
from equinox import Module, AbstractVar

from .._specimen import AbstractSpecimen
from .._potential import AbstractScatteringPotential, AbstractVoxels
from .._config import ImageConfig

from ...image import rfftn, irfftn
from ...typing import ComplexImage


class AbstractScatteringMethod(Module, strict=True):
    """A model of electron scattering onto the exit plane of the specimen."""

    config: AbstractVar[ImageConfig]

    @abstractmethod
    def __call__(self, specimen: AbstractSpecimen, **kwargs: Any) -> ComplexImage:
        """Compute an image at the exit plane, measured at the `ImageConfig.pixel_size`
        with phase shifts applied from `AbstractSpecimen.pose`.
        """
        raise NotImplementedError


class AbstractProjectionMethod(AbstractScatteringMethod, strict=True):
    """A model for computing projections from an electron density representation."""

    config: AbstractVar[ImageConfig]

    @abstractmethod
    def project_potential(self, potential: AbstractScatteringPotential) -> ComplexImage:
        """Compute the scattering potential in the exit plane.

        **Arguments:**

        `potential`: The scattering potential representation.
        """
        raise NotImplementedError

    def __call__(self, specimen: AbstractSpecimen, **kwargs: Any) -> ComplexImage:
        # Get potential in the lab frame
        potential = specimen.potential_in_lab_frame
        # Compute the fourier projection in the exit plane
        potential_at_exit_plane = self.project_potential(potential, **kwargs)
        # Rescale the pixel size if different from the voxel size
        if isinstance(potential, AbstractVoxels):
            rescale_fn = lambda fourier_potential: rfftn(
                self.config.rescale_to_pixel_size(
                    irfftn(fourier_potential, s=self.config.padded_shape),
                    potential.voxel_size,
                )
            )
            null_fn = lambda fourier_potential: fourier_potential
            potential_at_exit_plane = jax.lax.cond(
                jnp.isclose(potential.voxel_size, self.config.pixel_size),
                null_fn,
                rescale_fn,
                potential_at_exit_plane,
            )
        # Apply translation through phase shifts
        potential_at_exit_plane *= specimen.pose.compute_shifts(
            self.config.padded_frequency_grid_in_angstroms.get()
        )

        return potential_at_exit_plane
