"""
Image normalization routines.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact


def normalize_image(
    image: Inexact[Array, "y_dim x_dim"],
    *,
    is_real: bool = True,
    half_space: bool = True,
) -> Inexact[Array, "y_dim x_dim"]:
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
    )


def rescale_image(
    image: Inexact[Array, "y_dim x_dim"],
    std: float | Float[Array, ""],
    mean: float | Float[Array, ""],
    *,
    is_real: bool = True,
    half_space: bool = True,
) -> Inexact[Array, "y_dim x_dim"]:
    """Normalize so that the image is mean mu
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
    # First normalize image to zero mean and unit standard deviation
    if is_real:
        normalized_image = (image - image.mean()) / image.std()
        rescaled_image = std * normalized_image + mean
    else:
        N1, N2 = image.shape
        n_pixels, n_modes = N1 * (2 * N2 - 1) if half_space else N1 * N2, N1 * N2
        image_with_zero_mean = image.at[0, 0].set(0.0)
        image_std = jnp.linalg.norm(image_with_zero_mean) / jnp.sqrt(n_modes)
        normalized_image = image_with_zero_mean / image_std
        rescaled_image = (normalized_image * std).at[0, 0].set(mean * jnp.sqrt(n_pixels))
    return rescaled_image
