"""
Image normalization routines.
"""

__all__ = ["rescale_image", "normalize_image"]

from typing import Any

import jax.numpy as jnp

from ..types import Image


def normalize_image(image: Image, **kwargs: Any) -> Image:
    """
    Normalize so that the image is mean 0
    and standard deviation 1 in real space.
    """
    return rescale_image(image, 1.0, 0.0, **kwargs)


def rescale_image(
    image: Image, N: float, mu: float, *, real: bool = True
) -> Image:
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
    image = jnp.asarray(image)
    N1, N2 = image.shape
    # First normalize image to zero mean and unit standard deviation
    if real:
        normalized_image = (image - image.mean()) / image.std()
        rescaled_image = N * normalized_image + mu
    else:
        normalized_image = image.at[0, 0].set(0.0)
        normalized_image /= jnp.linalg.norm(normalized_image) / (N1 * N2)
        rescaled_image = (normalized_image * N).at[0, 0].set(mu * N1 * N2)
    return rescaled_image
