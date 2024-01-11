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
from jax.image import scale_and_translate

from ..density import ElectronDensity, Voxels, VoxelGrid
from ..manager import ImageManager

from ...utils import rfftn, irfftn
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
        Compute an image at the exit plane, measured at the ScatteringModel
        pixel size and post-processed with the ImageManager utilities.
        """
        image = self.scatter(density, **kwargs)
        if isinstance(density, VoxelGrid):
            # Resize the image to match the ImageManager.padded_shape
            image = self.manager.crop_or_pad_to_padded_shape(
                irfftn(image, s=density.weights.shape[0:2])
            )
        else:
            # ... otherwise, assume the image is already at the padded_shape
            image = irfftn(image, s=self.manager.padded_shape)
        # Rescale the pixel size if different from the voxel size
        if isinstance(density, Voxels):
            current_pixel_size = density.voxel_size
            new_pixel_size = self.pixel_size
            rescale_fn = lambda image: rescale_pixel_size(
                image,
                current_pixel_size,
                new_pixel_size,
                method=self.method,
            )
            null_fn = lambda image: image
            image = jax.lax.cond(
                jnp.isclose(current_pixel_size, new_pixel_size),
                null_fn,
                rescale_fn,
                image,
            )
        # Transform back to fourier space and give the image zero mean
        image = rfftn(image).at[0, 0].set(0.0 + 0.0j)
        return image

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
    method: str = "bicubic",
    antialias: bool = False,
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
    method :
        Interpolation method. See ``jax.image.scale_and_translate``
        for documentation.
    kwargs :
        Keyword arguments passed to ``jax.image.scale_and_translate``.

    Returns
    -------
    rescaled_image :
        An image with pixels whose size are rescaled by
        ``current_pixel_size / new_pixel_size``.
    """
    # Compute scale factor for pixel size rescaling
    scale_factor = current_pixel_size / new_pixel_size
    # Scaling in both dimensions is the same
    scaling = jnp.asarray([scale_factor, scale_factor])
    # Compute the translation in the jax.image convention that leaves
    # cryojax images untranslated
    N1, N2 = image.shape
    translation = (1 - scaling) * jnp.array([N1 // 2, N2 // 2], dtype=float)
    # Rescale pixel sizes
    rescaled_image = scale_and_translate(
        image,
        image.shape,
        (0, 1),
        scaling,
        translation,
        method,
        antialias=antialias,
        **kwargs,
    )

    return rescaled_image
