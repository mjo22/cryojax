"""
Routines to model image formation from 3D electron density
fields.
"""

__all__ = ["AbstractScatteringMethod", "AbstractProjectionMethod"]

from abc import abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
from equinox import Module, AbstractVar

from .._specimen import AbstractSpecimen
from .._density import AbstractElectronDensity, AbstractVoxels
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
    def project_density(self, density: AbstractElectronDensity) -> ComplexImage:
        """Compute the scattered wave of the electron density in the exit plane.

        **Arguments:**

        `density`: The electron density representation.
        """
        raise NotImplementedError

    def __call__(self, specimen: AbstractSpecimen, **kwargs: Any) -> ComplexImage:
        # Get density in the lab frame
        density = specimen.density_in_lab_frame
        # Compute the fourier projection in the exit plane
        image_at_exit_plane = self.project_density(density, **kwargs)
        # Rescale the pixel size if different from the voxel size
        if isinstance(density, AbstractVoxels):
            rescale_fn = lambda fourier_image: rfftn(
                self.config.rescale_to_pixel_size(
                    irfftn(fourier_image, s=self.config.padded_shape),
                    density.voxel_size,
                )
            )
            null_fn = lambda fourier_image: fourier_image
            image_at_exit_plane = jax.lax.cond(
                jnp.isclose(density.voxel_size, self.config.pixel_size),
                null_fn,
                rescale_fn,
                image_at_exit_plane,
            )
        # Apply translation through phase shifts
        image_at_exit_plane *= specimen.pose.compute_shifts(
            self.config.padded_frequency_grid_in_angstroms.get()
        )

        return image_at_exit_plane
