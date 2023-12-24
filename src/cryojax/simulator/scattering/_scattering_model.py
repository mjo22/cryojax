"""
Routines to model image formation from 3D electron density
fields.
"""

from __future__ import annotations

__all__ = ["ScatteringModel", "rescale_pixel_size"]

from abc import abstractmethod
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp

from ..density import ElectronDensity, Voxels
from ..pose import Pose
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
        Rasterization pixel size. This is in
        dimensions of length.
    interpolation_method :
        The interpolation method used for measuring
        the image at the new ``pixel_size``.

    Methods
    -------
    scatter:
        The scattering model.
    """

    manager: ImageManager = field()
    pixel_size: Real_ = field()

    interpolation_method: str = field(static=True, default="bicubic")

    @abstractmethod
    def scatter(
        self, density: ElectronDensity, pose: Optional[Pose] = None
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
        self, density: ElectronDensity, **kwargs: Any
    ) -> ComplexImage:
        """
        Compute an image at the exit plane, measured at the
        pixel size and post-processed with the ImageManager utilities.
        """
        image = self.scatter(density, **kwargs)
        image = ifftn(image).real
        if self.manager.padded_shape != image.shape:
            image = self.manager.crop_or_pad(image)
        if isinstance(density, Voxels):
            image = rescale_pixel_size(
                image,
                current_pixel_size=density.voxel_size,
                new_pixel_size=self.pixel_size,
                method=self.interpolation_method,
                antialias=False,
            )
        image = self.manager.normalize_to_cistem(fftn(image), is_real=False)
        return image


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
