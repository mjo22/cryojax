"""
Routines to compute radial averages of images.
"""

__all__ = ["radial_average"]

from typing import Optional

import jax
import jax.numpy as jnp

from ..core import Array, ArrayLike


@jax.jit
def radial_average(
    image: ArrayLike,
    norm: ArrayLike,
    bins: ArrayLike,
    grid: Optional[ArrayLike] = None,
) -> Array:
    """
    Radially average vectors r with a given magnitude
    coordinate system |r|.

    Arguments
    ---------
    image : `ArrayLike`, shape `(M1, M2)`
        Two-dimensional image.
    norm : `ArrayLike`, shape `(M1, M2)`
        Radial coordinate system of image.
    bins : `ArrayLike`, shape `(M,)`
        Radial bins for averaging. These
        must be evenly spaced.
    grid : `ArrayLike`, shape `(N1, N2)`, optional
        If ``None``, evalulate the spectrum as a 1D
        profile. Otherwise, evaluate the spectrum on this
        2D grid of frequencies.

    Returns
    -------
    average : `ArrayLike`, shape `(M,)` or `(N1, N2)`
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
