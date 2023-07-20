"""
Routines for data masking.
"""

__all__ = ["enforce_bounds"]

import jax.numpy as jnp

from ..core import Array


def enforce_bounds(density: Array, coords: Array, box_size: Array) -> Array:
    """
    Use a boolean mask to set density values out of
    bounds to zero. The mask is ``True`` for
    all points outside of the box_size, and False otherwise.
    """
    x, y = coords.T
    Lx, Ly = box_size[0], box_size[1]
    x_mask = jnp.logical_or(x < -Lx / 2, x >= Lx / 2)
    y_mask = jnp.logical_or(y < -Ly / 2, y >= Ly / 2)
    mask = jnp.logical_or(x_mask, y_mask)
    masked_density = jnp.where(mask, complex(0.0), density)

    return masked_density
