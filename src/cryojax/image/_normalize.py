"""
Image normalization routines.
"""

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact


def normalize_image(
    image: Inexact[Array, "y_dim x_dim"],
    *,
    is_real: bool = True,
    half_space: bool = True,
    shape_in_real_space: Optional[tuple[int, int]] = None,
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
        shape_in_real_space=shape_in_real_space,
    )


def rescale_image(
    image: Inexact[Array, "y_dim x_dim"],
    std: float | Float[Array, ""],
    mean: float | Float[Array, ""],
    *,
    is_real: bool = True,
    half_space: bool = True,
    shape_in_real_space: Optional[tuple[int, int]] = None,
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
        _, measured_std = compute_mean_and_std_from_fourier_image(
            image, half_space=half_space, shape_in_real_space=shape_in_real_space
        )
        normalized_image = image.at[0, 0].set(0.0) / measured_std
        rescaled_image = (normalized_image * std).at[0, 0].set(mean * N_modes)
    return rescaled_image


def compute_mean_and_std_from_fourier_image(
    fourier_image: Complex[Array, "y_dim x_dim"],
    *,
    half_space: bool = True,
    shape_in_real_space: Optional[tuple[int, int]] = None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute the mean and standard deviation in real space from
    an image in fourier space.
    """
    N1, N2 = fourier_image.shape
    if half_space:
        N_modes = (
            N1 * (2 * N2 - 1)
            if shape_in_real_space is None
            else shape_in_real_space[0] * shape_in_real_space[1]
        )
    else:
        N_modes = N1 * N2
    # The mean is just the zero mode divided by the number of modes
    mean = fourier_image[0, 0].real / N_modes
    # The standard deviation is square root norm squared of the zero mean image
    std = (
        jnp.sqrt(
            jnp.sum(jnp.abs(fourier_image[:, 0]) ** 2)
            - jnp.abs(fourier_image[0, 0]) ** 2
            + 2 * jnp.sum(jnp.abs(fourier_image[:, 1:]) ** 2)
        )
        if half_space
        else jnp.linalg.norm(fourier_image.at[0, 0].set(0.0))
    ) / N_modes

    return (mean, std)
