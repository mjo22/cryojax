"""
Routines to compute radial averages of images.
"""

__all__ = ["radial_average"]

from typing import Optional, Union

import jax
import jax.numpy as jnp

from ..typing import RealVector, Vector, Image, ImageCoords


@jax.jit
def radial_average(
    image: Image,
    norm: Image,
    bins: RealVector,
    grid: Optional[ImageCoords] = None,
) -> Union[Image, Vector]:
    """
    Radially average vectors r with a given magnitude
    coordinate system |r|.

    Arguments
    ---------
    image :
        Two-dimensional image.
    norm :
        Radial coordinate system of image.
    bins :
        Radial bins for averaging. These
        must be evenly spaced.
    grid :
        If ``None``, evalulate the spectrum as a 1D
        profile. Otherwise, evaluate the spectrum on this
        2D grid of frequencies.

    Returns
    -------
    average :
        Radial average of image.
    """
    bins = jnp.asarray(bins)

    dr = bins[1] - bins[0]

    def average(carry, r_i):
        mask = jnp.logical_and(norm >= r_i, norm < r_i + dr)
        return carry, jnp.sum(jnp.where(mask, image, 0.0)) / jnp.sum(mask)

    _, profile = jax.lax.scan(average, None, bins)

    if grid is not None:
        idxs = jnp.digitize(grid, bins).ravel()
        return jax.vmap(lambda idx: profile[idx])(idxs).reshape(grid.shape)
    else:
        return profile
