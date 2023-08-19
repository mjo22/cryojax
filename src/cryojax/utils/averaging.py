"""
Routines to compute radial averages of images.
"""

__all__ = ["radial_average"]

from typing import Optional

import jax
import jax.numpy as jnp

from ..core import Array


@jax.jit
def radial_average(
    image: Array,
    norm: Array,
    bins: Array,
    grid: Optional[Array] = None,
) -> Array:
    """
    Radially average vectors r with a given magnitude
    coordinate system |r|.

    Arguments
    ---------
    image : `jax.Array`, shape `(M1, M2)`
        Two-dimensional image.
    norm : `jax.Array`, shape `(M1, M2)`
        Radial coordinate system of image.
    bins : `jax.Array`, shape `(M,)`
        Radial bins for averaging. These
        must be evenly spaced.
    grid : `jax.Array`, shape `(N1, N2)`, optional
        If ``None``, evalulate the spectrum as a 1D
        profile. Otherwise, evaluate the spectrum on this
        2D grid of frequencies.

    Returns
    -------
    average : `jax.Array`, shape `(M,)` or `(N1, N2)`
        Radial average of image.
    """
    dr = bins[1] - bins[0]

    def average(carry, r_i):
        mask = jnp.logical_and(norm >= r_i, norm < r_i + dr)
        return carry, jnp.sum(jnp.where(mask, image, 0.0)) / jnp.sum(mask)

    _, profile = jax.lax.scan(average, None, bins)
    # profile = jax.vmap(average)(bins)

    if grid is not None:
        idxs = jnp.digitize(grid, bins).ravel()
        return jax.vmap(lambda idx: profile[idx])(idxs).reshape(grid.shape)
    else:
        return profile
