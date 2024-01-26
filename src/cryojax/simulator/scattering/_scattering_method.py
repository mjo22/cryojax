"""
Routines to model image formation from 3D electron density
fields.
"""

from __future__ import annotations

__all__ = ["AbstractScatteringMethod", "AbstractProjectionMethod"]

from abc import abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
from equinox import Module

from ..specimen import AbstractSpecimen
from ..density import AbstractElectronDensity, AbstractVoxels
from ..manager import ImageManager

from ...image import rfftn, irfftn
from ...typing import ComplexImage


class AbstractScatteringMethod(Module):
    """
    A model of electron scattering onto the exit plane of the specimen.

    Attributes
    ----------
    manager:
        Handles image configuration and
        utility routines.
    """

    manager: ImageManager

    @abstractmethod
    def __call__(
        self, specimen: AbstractSpecimen, **kwargs: Any
    ) -> ComplexImage:
        """
        Compute an image at the exit plane, measured at the ScatteringMethod
        pixel size and post-processed with the ImageManager utilities.
        """
        raise NotImplementedError


class AbstractProjectionMethod(AbstractScatteringMethod):
    """
    A model for computing projections from an electron density representation.

    In subclasses, overwrite the ``ScatteringConfig.scatter``
    routine.

    Attributes
    ----------
    manager:
        Handles image configuration and
        utility routines.
    """

    manager: ImageManager

    @abstractmethod
    def project_density(
        self, density: AbstractElectronDensity
    ) -> ComplexImage:
        """
        Compute the scattered wave of the electron
        density in the exit plane.

        Arguments
        ---------
        density :
            The electron density representation.
        """
        raise NotImplementedError

    def __call__(
        self, specimen: AbstractSpecimen, **kwargs: Any
    ) -> ComplexImage:
        """
        Compute an image at the exit plane, measured at the ScatteringModel
        pixel size and post-processed with the ImageManager utilities.
        """
        # Get density in the lab frame
        density = specimen.density_in_lab_frame
        # Compute the fourier projection in the exit plane
        image_at_exit_plane = self.project_density(density, **kwargs)
        # Rescale the pixel size if different from the voxel size
        if isinstance(density, AbstractVoxels):
            rescale_fn = lambda fourier_image: rfftn(
                self.manager.rescale_to_pixel_size(
                    irfftn(fourier_image, s=self.manager.padded_shape),
                    density.voxel_size,
                )
            )
            null_fn = lambda fourier_image: fourier_image
            image_at_exit_plane = jax.lax.cond(
                jnp.isclose(density.voxel_size, self.manager.pixel_size),
                null_fn,
                rescale_fn,
                image_at_exit_plane,
            )
        # Apply translation through phase shifts
        image_at_exit_plane *= specimen.pose.shifts(
            self.manager.padded_frequency_grid_in_angstroms.get()
        )

        return image_at_exit_plane
