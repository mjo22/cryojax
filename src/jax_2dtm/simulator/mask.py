"""
Masks to apply to images in real space.
"""

from __future__ import annotations

__all__ = [
    "compute_circular_mask",
    "Mask",
    "CircularMask",
]

from abc import ABCMeta, abstractmethod
from dataclasses import InitVar
from typing import Any

import jax.numpy as jnp

from ..core import dataclass, field, Array, ArrayLike


@dataclass
class Mask(metaclass=ABCMeta):
    """
    Base class for computing and applying an image filter.

    Attributes
    ----------
    coords : `jax.Array`
        The image coordinates in real space.
    """

    mask: Array = field(pytree_node=False, init=False)

    coords: InitVar[ArrayLike]

    def __post_init__(self, *args):
        object.__setattr__(self, "mask", self.compute(*args))

    @abstractmethod
    def compute(self, *args: tuple[Any, ...]) -> Array:
        """Compute the mask."""
        raise NotImplementedError

    def __call__(self, image: Array) -> Array:
        """Apply the mask to an image."""
        return self.mask * image


@dataclass
class CircularMask(Mask):
    """
    Apply a circular mask to an image.

    See documentation for
    ``jax_2dtm.simulator.compute_circular_mask``
    for more information.

    Attributes
    ----------
    radius : `float`
    rolloff : `float`
    """

    radius: float = field(pytree_node=False, default=1.0)
    rolloff: float = field(pytree_node=False, default=0.05)

    def compute(self, coords: Array) -> Array:
        return compute_circular_mask(
            coords,
            self.radius,
            self.rolloff,
        )


def compute_circular_mask(
    coords: Array,
    cutoff: float = 1.0,
    rolloff: float = 0.05,
) -> Array:
    """
    Create an anti-aliasing filter.

    Parameters
    ----------
    coords : `jax.Array`, shape `(N1, N2, 2)`
        The image coordiantes, in units of pixels.
    cutoff : `float`, optional
        The cutoff radius as a fraction of half
        the smallest box dimension. By default 1.0.
    rolloff : `float`, optional
        The rolloff width as a fraction of the smallest box dimension.
        By default 0.05.

    Returns
    -------
    mask : `jax.Array`, shape `(N1, N2)`
        An array representing the circular mask.
    """

    r_max = min(coords.shape[0:2]) // 2
    r_cut = cutoff * r_max

    coords_norm = jnp.linalg.norm(coords, axis=-1)

    coords_cut = coords_norm > r_cut

    rolloff_width = rolloff * r_max
    mask = 0.5 * (
        1
        + jnp.cos(
            (coords_norm - r_cut - rolloff_width) / rolloff_width * jnp.pi
        )
    )

    mask = jnp.where(coords_cut, 0.0, mask)
    mask = jnp.where(coords_norm <= r_cut - rolloff_width, 1.0, mask)

    return mask
