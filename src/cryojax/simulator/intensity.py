"""
Routines to handle varying imaging conditions.
"""

from __future__ import annotations

__all__ = ["Intensity", "rescale_image"]

import jax.numpy as jnp

from ..core import dataclass, Array, Scalar, Serializable


@dataclass
class Intensity(Serializable):
    """
    An PyTree that controls image intensity rescaling.

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
        Return the normalized image.
        """
        return rescale_image(image, self.N, self.mu, real=real)


def rescale_image(
    image: Array, N: float, mu: float, real: bool = True
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
