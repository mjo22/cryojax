"""
Abstraction of electron detectors in a cryo-EM image.
"""

__all__ = ["Detector", "NullDetector", "GaussianDetector", "pixelize_image"]

import jax
import jax.numpy as jnp

from dataclasses import KW_ONLY
from abc import ABCMeta, abstractmethod
from typing import Optional, Any
from functools import partial

from .noise import GaussianNoise
from .kernel import Kernel, Constant
from ..utils import scale, irfftn
from ..core import field, Module
from ..types import Real_, RealImage, ImageCoords


class Detector(Module, metaclass=ABCMeta):
    """
    Base class for an electron detector.

    Attributes
    ----------
    pixel_size :
        The pixel size measured by the detector.
        This is in dimensions of physical length.
    method :
        The interpolation method used for measuring
        the image at the ``pixel_size``.
    """

    pixel_size: Optional[Real_] = field(default=None)
    method: str = field(static=True, default="bicubic")

    def pixelize(self, image: RealImage, resolution: Real_) -> RealImage:
        """
        Pixelize an image at a given resolution to
        the detector pixel size.
        """
        pixel_size = resolution if self.pixel_size is None else self.pixel_size
        pixelized = pixelize_image(
            image,
            resolution,
            pixel_size,
            method=self.method,
            antialias=False,
        )
        return pixelized

    @abstractmethod
    def sample(
        self,
        freqs: ImageCoords,
        image: Optional[RealImage] = None,
    ) -> RealImage:
        """Sample a realization from the detector noise model."""
        raise NotImplementedError


class NullDetector(Detector):
    """
    A 'null' detector.
    """

    def sample(
        self,
        freqs: ImageCoords,
        image: Optional[RealImage] = None,
    ) -> RealImage:
        return jnp.zeros(jnp.asarray(freqs).shape[0:-1])


class GaussianDetector(GaussianNoise, Detector):
    """
    A detector with a gaussian noise model. By default,
    this is a white noise model.

    Attributes
    ----------
    variance :
        A kernel that computes the variance
        of the detector noise. By default,
        ``Constant()``.
    """

    variance: Kernel = field(default_factory=Constant)

    def sample(
        self,
        freqs: ImageCoords,
        image: Optional[RealImage] = None,
    ) -> RealImage:
        return irfftn(super().sample(freqs))


@partial(jax.jit, static_argnames=["method", "antialias"])
def pixelize_image(
    image: RealImage, resolution: Real_, pixel_size: Real_, **kwargs: Any
) -> RealImage:
    """
    Measure an image at a given pixel size using interpolation.

    For more detail, see ``cryojax.utils.interpolation.scale``.

    Parameters
    ----------
    image :
        The image to be magnified.
    resolution :
        The resolution, in physical length, of
        the image.
    pixel_size :
        The pixel size of the detector.
    """
    scale_factor = resolution / pixel_size
    s = jnp.array([scale_factor, scale_factor])
    return scale(image, image.shape, s, **kwargs)
