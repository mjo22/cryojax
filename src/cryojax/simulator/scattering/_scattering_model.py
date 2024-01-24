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
from equinox import Module

from ..specimen import Specimen
from ..density import ElectronDensity, Voxels, FourierVoxelGrid
from ..manager import ImageManager

from ...image import rfftn, irfftn, CoordinateGrid, FrequencyGrid
from ...core import field
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
    rescale_method :
        The interpolation method used for measuring
        the image at the new ``pixel_size``. Passed to
        ``jax.image.scale_and_translate``. This options
        applies to voxel-based ``ElectronDensity`` representations.
    """

    manager: ImageManager = field()

    rescale_method: str = field(static=True, default="bicubic")

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

    def __call__(self, specimen: Specimen, **kwargs: Any) -> ComplexImage:
        """
        Compute an image at the exit plane, measured at the ScatteringModel
        pixel size and post-processed with the ImageManager utilities.
        """
        # Get density in the lab frame
        density = specimen.density_in_lab_frame
        # Compute the image in the exit plane
        image_at_exit_plane = self.scatter(density, **kwargs)
        if isinstance(density, FourierVoxelGrid):
            # Resize the image to match the ImageManager.padded_shape
            image_at_exit_plane = self.manager.crop_or_pad_to_padded_shape(
                irfftn(image_at_exit_plane, s=density.weights.shape[0:2])
            )
        else:
            # ... otherwise, assume the image is already at the padded_shape
            image_at_exit_plane = irfftn(
                image_at_exit_plane, s=self.manager.padded_shape
            )
        # Rescale the pixel size if different from the voxel size
        if isinstance(density, Voxels):
            current_pixel_size = density.voxel_size
            new_pixel_size = self.manager.pixel_size
            rescale_fn = lambda image: rescale_pixel_size(
                image,
                current_pixel_size,
                new_pixel_size,
                method=self.rescale_method,
            )
            null_fn = lambda image: image
            image_at_exit_plane = jax.lax.cond(
                jnp.isclose(current_pixel_size, new_pixel_size),
                null_fn,
                rescale_fn,
                image_at_exit_plane,
            )
        # Transform back to fourier space and give the image zero mean
        image_at_exit_plane = (
            rfftn(image_at_exit_plane).at[0, 0].set(0.0 + 0.0j)
        )
        # Apply translation through phase shifts
        image_at_exit_plane *= specimen.pose.shifts(
            self.manager.padded_frequency_grid_in_angstroms.get()
        )

        return image_at_exit_plane


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
