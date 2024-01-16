"""
Image normalization routines.
"""

__all__ = ["rescale_image", "normalize_image"]

from typing import Any, Optional

import jax.numpy as jnp

from ..typing import Image


def normalize_image(
    image: Image,
    *,
    is_real: bool = True,
    half_space: bool = True,
    shape_in_real_space: Optional[tuple[int, int]] = None,
) -> Image:
    """
    Normalize so that the image is mean 0
    and standard deviation 1 in real space.
    """
    return rescale_image(
        image,
        1.0,
        0.0,
        is_real=is_real,
        half_space=half_space,
        shape_in_real_space=shape_in_real_space,
    )


def rescale_image(
    image: Image,
    std: float,
    mean: float,
    *,
    is_real: bool = True,
    half_space: bool = True,
    shape_in_real_space: Optional[tuple[int, int]] = None,
) -> Image:
    """
    Normalize so that the image is mean mu
    and standard deviation N in real space.

    Parameters
    ----------
    image : `Array`, shape `(N1, N2)`
        The image in either real or Fourier space.
        If in Fourier space, the zero frequency
        component should be in the corner of the image.
        It should also be on the half space.
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
        if half_space:
            N_modes = (
                N1 * (2 * N2 - 1)
                if shape_in_real_space is None
                else shape_in_real_space[0] * shape_in_real_space[1]
            )
        else:
            N_modes = N1 * N2
        normalized_image = image.at[0, 0].set(0.0)
        normalizing_factor = (
            jnp.sqrt(
                jnp.sum(jnp.abs(normalized_image[:, 0]) ** 2)
                + 2 * jnp.sum(jnp.abs(normalized_image[:, 1:]) ** 2)
            )
            if half_space
            else jnp.linalg.norm(normalize_image)
        )
        normalized_image /= normalizing_factor / N_modes
        rescaled_image = (normalized_image * std).at[0, 0].set(mean * N_modes)
    return rescaled_image
