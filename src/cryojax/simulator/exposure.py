"""
Routines to handle variations in image intensity
due to electron exposure.
"""

from __future__ import annotations

__all__ = ["Exposure", "NullExposure", "UniformExposure", "rescale_image"]

from abc import ABCMeta, abstractmethod
from functools import partial

import jax

from ..core import field, Module, Image, Real_


class Exposure(Module, metaclass=ABCMeta):
    """
    Controls parameters related to variation in
    the image intensity.

    For example, this might include
    the incoming electron dose and radiation damage.
    """

    @abstractmethod
    def scale(self, image: Image, real: bool = False) -> Image:
        """
        Return the scaled image.
        """
        raise NotImplementedError


class NullExposure(Exposure):
    """
    A `null` exposure model. Do not change the
    image when it is passsed through the pipeline.
    """

    def scale(self, image: Image, real: bool = False) -> Image:
        """Return the image unchanged"""
        return image


class UniformExposure(Exposure):
    """
    Scale the signal intensity uniformly.

    Attributes
    ----------
    N : Intensity scaling.
    mu : Intensity offset.
    """

    N: Real_ = field(default=1e5)
    mu: Real_ = field(default=0.0)

    def scale(self, image: Image, real: bool = False) -> Image:
        """
        Return the scaled image.
        """
        return rescale_image(image, self.N, self.mu, real=real)


@partial(jax.jit, static_argnames=["real"])
def rescale_image(
    image: Image, N: float, mu: float, *, real: bool = False
) -> Image:
    """
    Normalize so that the image is mean mu
    and standard deviation N in real space.

    Parameters
    ----------
    image :
        The image in either real or Fourier space.
        If in Fourier space, the zero frequency
        component should be in the center of the image.
    N : Intensity scale factor.
    mu : Intensity offset.
    real :
        If ``True``, the given ``image`` is in real
        space. If ``False``, it is in Fourier space.

    Returns
    -------
    rescaled_image :
        Image rescaled by an offset ``mu`` and scale factor ``N``.
    """
    N1, N2 = image.shape
    if real:
        rescaled_image = N * image + mu
    else:
        rescaled_image = N * image
        rescaled_image = rescaled_image.at[0, 0].set(
            rescaled_image[0, 0] + (mu * N1 * N2)
        )
    return rescaled_image
