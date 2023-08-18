"""
Routines to handle variations in image intensity
due to electron exposure.
"""

from __future__ import annotations

__all__ = ["Exposure", "NullExposure", "UniformExposure", "rescale_image"]

from abc import ABCMeta, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp

from ..core import dataclass, Array, Scalar, CryojaxObject


@dataclass
class Exposure(CryojaxObject, metaclass=ABCMeta):
    """
    An PyTree that controls parameters related to
    variation in the image intensity. For example,
    this includes the incoming electron dose and
    radiation damage.
    """

    @abstractmethod
    def rescale(self, image: Array, real: bool = True) -> Array:
        """
        Return the rescaled image.
        """
        raise NotImplementedError


@dataclass
class NullExposure(Exposure):
    """
    A `null` exposure model. Do not change the
    image when it is passsed through the pipeline.
    """

    def rescale(self, image: Array, real: bool = True) -> Array:
        """Return the image unchanged"""
        return image


@dataclass
class UniformExposure(Exposure):
    """
    Rescale the signal intensity uniformly.

    Attributes
    ----------
    N : `cryojax.core.Scalar`
        Intensity standard deviation
    mu : `cryojax.core.Scalar`
        Intensity offset
    """

    N: Scalar = 1.0
    mu: Scalar = 0.0

    def rescale(self, image: Array, real: bool = True) -> Array:
        """
        Return the scaled image.
        """
        return rescale_image(image, self.N, self.mu, real=real)


@partial(jax.jit, static_argnames=["real"])
def rescale_image(
    image: Array, N: float, mu: float, *, real: bool = True
) -> Array:
    """
    Normalize so that the image is mean mu
    and standard deviation N in real space.

    Parameters
    ----------
    image : `jax.Array`, shape `(N1, N2)`
        The image in either real or Fourier space.
        If in Fourier space, the zero frequency
        component should be in the center of the image.
    N : `float`
        Intensity scale factor.
    mu : `float`
        Intensity offset.
    real : `bool`
        If ``True``, the given ``image`` is in real
        space. If ``False``, it is in Fourier space.

    Returns
    -------
    rescaled_image : `jax.Array`, shape `(N1, N2)`
        Image rescaled to have mean ``mu`` and standard
        deviation ``N``.
    """
    N1, N2 = image.shape
    # First normalize image to zero mean and unit standard deviation
    if real:
        normalized_image = (image - image.mean()) / image.std()
        rescaled_image = N * normalized_image + mu
    else:
        normalized_image = image.at[N1 // 2, N2 // 2].set(0.0)
        normalized_image /= jnp.linalg.norm(normalized_image) / (N1 * N2)
        rescaled_image = (
            (normalized_image * N).at[N1 // 2, N2 // 2].set(mu * N1 * N2)
        )
    return rescaled_image
