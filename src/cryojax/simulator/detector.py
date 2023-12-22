"""
Abstraction of electron detectors in a cryo-EM image.
"""

__all__ = [
    "Detector",
    "NullDetector",
    "NoiselessDetector",
    "GaussianDetector",
    "rescale_pixel_size",
]

from abc import abstractmethod
from typing import Optional, Any
from typing_extensions import override
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from .noise import GaussianNoise
from .kernel import Kernel, Constant
from ..utils import scale, ifftn
from ..core import field, Module
from ..typing import Real_, RealImage, ImageCoords


class Detector(Module):
    """
    Base class for an electron detector.

    Attributes
    ----------
    pixel_size :
        The pixel size measured by the detector.
        This is in dimensions of physical length.
        If this is given, the pixel size of the
        image will be interpolated to this new pixel
        size.
    interpolation_method :
        The interpolation method used for measuring
        the image at the new ``pixel_size``.
    """

    pixel_size: Optional[Real_] = field(default=None)
    interpolation_method: str = field(static=True, default="bicubic")

    def measure_at_pixel_size(self, image: RealImage, current_pixel_size: Real_) -> RealImage:
        """
        Measure an image at a given pixel size to
        the detector pixel size.
        """
        if self.pixel_size is None:
            return image
        else:
            return rescale_pixel_size(
                image,
                current_pixel_size,
                new_pixel_size=self.pixel_size,
                method=self.interpolation_method,
                antialias=False,
            )

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        freqs: ImageCoords,
        image: Optional[RealImage] = None,
    ) -> RealImage:
        """Sample a realization from the detector noise model."""
        raise NotImplementedError


class NullDetector(Detector):
    """
    A 'null' detector.
    """

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        freqs: ImageCoords,
        image: Optional[RealImage] = None,
    ) -> RealImage:
        return jnp.zeros(jnp.asarray(freqs).shape[0:-1])


class NoiselessDetector(Detector):
    """
    A detector with no noise model.
    """

    @override
    def sample(
        self,
        key: PRNGKeyArray,
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

    @override
    def sample(
        self,
        key: PRNGKeyArray,
        freqs: ImageCoords,
        image: Optional[RealImage] = None,
    ) -> RealImage:
        return ifftn(super().sample(key, freqs)).real


@partial(jax.jit, static_argnames=["method", "antialias"])
def rescale_pixel_size(
    image: RealImage, current_pixel_size: Real_, new_pixel_size: Real_, **kwargs: Any
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
