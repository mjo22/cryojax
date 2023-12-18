"""
Abstraction of electron detectors in a cryo-EM image.
"""

__all__ = ["Detector", "NullDetector", "GaussianDetector", "pixelize_image"]

from abc import abstractmethod
from typing import Optional, Any
from typing_extensions import override
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from .noise import GaussianNoise
from .kernel import Kernel, Constant
from ..utils import scale, irfftn
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
    interpolation_method :
        The interpolation method used for measuring
        the image at the ``pixel_size``.
    """

    pixel_size: Optional[Real_] = field(default=None)
    interpolation_method: str = field(static=True, default="bicubic")

    def pixelize(self, image: RealImage, resolution: Real_) -> RealImage:
        """
        Pixelize an image at a given resolution to
        the detector pixel size.
        """
        if self.pixel_size is None:
            return image
        else:
            return pixelize_image(
                image,
                resolution,
                self.pixel_size,
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
        return irfftn(super().sample(key, freqs))


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
