"""
Image normalization routines.
"""

__all__ = ["rescale_image", "normalize_image"]

from typing import Any

import jax.numpy as jnp

from ..typing import Image


def normalize_image(image: Image, **kwargs: Any) -> Image:
    """
    Normalize so that the image is mean 0
    and standard deviation 1 in real space.
    """
    return rescale_image(image, 1.0, 0.0, **kwargs)


def rescale_image(
    image: Image, std: float, mean: float, *, is_real: bool = True
) -> Image:
    """
    Normalize so that the image is mean mu
    and standard deviation N in real space.

    Parameters
    ----------
    image : `Array`, shape `(N1, N2)`
        The image in either real or Fourier space.
        If in Fourier space, the zero frequency
        component should be in the center of the image.
    std : `float`
        Intensity standard deviation.
    mean : `float`
        Intensity offset.
    is_real : `bool`
        If ``True``, the given ``image`` is in real
        space. If ``False``, it is in Fourier space.

    Returns
    -------
    rescaled_image : `Array`, shape `(N1, N2)`
        Image rescaled to have mean ``mu`` and standard
        deviation ``N``.
    """
    image = jnp.asarray(image)
    N1, N2 = image.shape
    # First normalize image to zero mean and unit standard deviation
    if is_real:
        normalized_image = (image - image.mean()) / image.std()
        rescaled_image = std * normalized_image + mean
    else:
        normalized_image = image.at[0, 0].set(0.0)
        normalized_image /= jnp.linalg.norm(normalized_image) / (N1 * N2)
        rescaled_image = (normalized_image * std).at[0, 0].set(mean * N1 * N2)
    return rescaled_image
