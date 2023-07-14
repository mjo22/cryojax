"""
Routines to compute radial averages of images.
"""

__all__ = ["radial_average"]

import jax
import jax.numpy as jnp

from ..core import Array


def radial_average(
    image: Array, norm: Array, bins: Array, grid: bool = False
) -> Array:
    """
    Radially average vectors r with a given magnitude
    coordinate system |r|.

    Arguments
    ---------
    image : `jax.Array`, shape `(M1, M2)`
        Two-dimensional image.
    norm : `jax.Array`, shape `(N1, N2)`
        Radial coordinate system of image.
    bins : `jax.Array`, shape `(M,)`
        Radial bins for averaging. These
        must be evenly spaced.
    grid : `bool`
        If ``True``, evalulate profile on the
        2D grid set by ``norm``.

    Returns
    -------
    average : `jax.Array`, shape `(M,)` or `(N1, N2)`
        Radial average of image.
    """
    dr = bins[1] - bins[0]

    def average(r_i):
        mask = jnp.logical_and(norm >= r_i, norm < r_i + dr)
        return jnp.sum(jnp.where(mask, image, 0.0)) / jnp.sum(mask)

    profile = jax.vmap(average)(bins)

    if grid:
        return jax.vmap(lambda idx: profile[idx])(jnp.digitize(norm, bins))
    else:
        return profile
