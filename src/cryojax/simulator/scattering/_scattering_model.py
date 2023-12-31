"""
Routines to model image formation from 3D electron density
fields.
"""

from __future__ import annotations

__all__ = ["ScatteringModel", "rescale_pixel_size"]

from abc import abstractmethod
from functools import partial, cached_property
from typing import Any

import jax
import jax.numpy as jnp

from ..density import ElectronDensity, Voxels
from ..manager import ImageManager

from ...utils import fftn, ifftn, scale
from ...core import field, Module
from ...typing import Real_, RealImage, ComplexImage


class ScatteringModel(Module):
    """
    A model of electron scattering onto the exit plane of the specimen.

    In subclasses, overwrite the ``ScatteringConfig.scatter``
    routine.

    Attributes
    ----------
    manager:
        Handles image configuration and
        utility routines.
    pixel_size :
        The pixel size of the image in Angstroms.
        For voxel-based ``ElectronDensity`` representations,
        if the pixel size is different than the voxel size,
        images will be interpolated in real space.
    method :
        The interpolation method used for measuring
        the image at the new ``pixel_size``. Passed to
        ``jax.image.scale_and_translate``.

    Methods
    -------
    scatter:
        The scattering model.
    """

    manager: ImageManager = field()
    pixel_size: Real_ = field()

    method: str = field(static=True, default="bicubic")

    @abstractmethod
    def scatter(self, density: ElectronDensity) -> ComplexImage:
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
        self, density: ElectronDensity, **kwargs: Any
    ) -> ComplexImage:
        """
        Compute an image at the exit plane, measured at the
        pixel size and post-processed with the ImageManager utilities.
        """
        image = self.scatter(density, **kwargs)
        image = ifftn(image).real
        # Resize the image to match the ImageManager config
        if self.manager.padded_shape != image.shape:
            image = self.manager.crop_or_pad_to_padded_shape(image)
        # Rescale the pixel size if different from the voxel size
        if isinstance(density, Voxels):
            current_pixel_size = density.voxel_size
            new_pixel_size = self.pixel_size
            rescale_fn = lambda image: rescale_pixel_size(
                image,
                current_pixel_size,
                new_pixel_size,
                method=self.method,
                antialias=False,
            )
            null_fn = lambda image: image
            image = jax.lax.cond(
                jnp.isclose(current_pixel_size, new_pixel_size),
                null_fn,
                rescale_fn,
                image,
            )
        # Give the image zero mean
        image = image - jnp.mean(image)
        return fftn(image)

    @cached_property
    def coordinate_grid_in_angstroms(self):
        return self.pixel_size * self.manager.coordinate_grid

    @cached_property
    def frequency_grid_in_angstroms(self):
        return self.manager.frequency_grid / self.pixel_size

    @cached_property
    def padded_coordinate_grid_in_angstroms(self):
        return self.pixel_size * self.manager.padded_coordinate_grid

    @cached_property
    def padded_frequency_grid_in_angstroms(self):
        return self.manager.padded_frequency_grid / self.pixel_size


@partial(jax.jit, static_argnames=["method", "antialias"])
def rescale_pixel_size(
    image: RealImage,
    current_pixel_size: Real_,
    new_pixel_size: Real_,
    **kwargs: Any,
) -> RealImage:
    """
    Measure an image at a given pixel size using interpolation.

    For more detail, see ``cryojax.utils.interpolation.scale``.

    Parameters
    ----------
    image :
        The image to be magnified.
    current_pixel_size :
        The pixel size of the input image.
    new_pixel_size :
        The new pixel size after interpolation.
    """
    scale_factor = current_pixel_size / new_pixel_size
    s = jnp.array([scale_factor, scale_factor])
    return scale(image, image.shape, s, **kwargs)
