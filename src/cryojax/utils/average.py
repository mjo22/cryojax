"""
Routines to compute radial averages of images.
"""

__all__ = ["radial_average"]

from typing import Optional, Union, Any

import jax
import jax.numpy as jnp

from ..typing import RealVector, Vector, Image, RealImage


@jax.jit
def radial_average(
    image: Image,
    radial_grid: RealImage,
    bins: RealVector,
    interpolating_radial_grid: Optional[RealImage] = None,
    **kwargs: Any,
) -> Union[Image, Vector]:
    """
    Radially average vectors r with a given magnitude
    coordinate system |r|.

    Arguments
    ---------
    image :
        Two-dimensional image.
    radial_grid :
        Radial coordinate system of image.
    bins :
        Radial bins for averaging. These
        must be evenly spaced.
    interpolating_radial_grid :
        If ``None``, evalulate the spectrum as a 1D
        profile. Otherwise, evaluate the spectrum on this
        2D grid of frequencies using linear interpolation.

    Returns
    -------
    average :
        Radial average of image.
    """
    bins = jnp.asarray(bins)

    dr = bins[1] - bins[0]

    def average(carry, r_i):
        mask = jnp.logical_and(radial_grid >= r_i, radial_grid < r_i + dr)
        return carry, jnp.sum(jnp.where(mask, image, 0.0)) / jnp.sum(mask)

    _, profile = jax.lax.scan(average, None, bins)

    if interpolating_radial_grid is not None:
        return jnp.interp(
            interpolating_radial_grid.ravel(), bins, profile, **kwargs
        ).reshape(interpolating_radial_grid.shape)
    else:
        return profile
