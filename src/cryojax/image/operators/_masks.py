"""
Masks to apply to images in real space.
"""

from __future__ import annotations

__all__ = ["Mask", "MaskT", "CircularMask", "compute_circular_mask"]

from typing import Any, TypeVar

import jax.numpy as jnp

from ._operator import OperatorAsBuffer
from ...core import field
from ...typing import RealImage, ImageCoords


MaskT = TypeVar("MaskT", bound="Mask")
"""TypeVar for the Mask base class."""


class Mask(OperatorAsBuffer):
    """
    Base class for computing and applying an image mask.

    Attributes
    ----------
    mask :
        The mask. Note that this is automatically
        computed upon instantiation.
    """

    def __init__(self, mask: RealImage):
        """Compute the mask."""
        self.operator = mask

    @property
    def mask(self) -> RealImage:
        return self.operator


class CircularMask(Mask):
    """
    Apply a circular mask to an image.

    See documentation for
    ``cryojax.simulator.compute_circular_mask``
    for more information.

    Attributes
    ----------
    radius :
        The radius of the mask in Angstroms.
    rolloff :
        By default, ``0.05``.
    """

    radius: float = field(static=True)
    rolloff: float = field(static=True)

    def __init__(
        self,
        coordinate_grid_in_angstroms: ImageCoords,
        radius: float,
        rolloff: float = 0.05,
    ) -> None:
        self.radius = radius
        self.rolloff = rolloff
        self.operator = compute_circular_mask(
            coordinate_grid_in_angstroms, self.radius, self.rolloff
        )


def compute_circular_mask(
    coordinate_grid_in_angstroms: ImageCoords,
    radius: float,
    rolloff: float = 0.05,
) -> RealImage:
    """
    Create a circular mask.

    Parameters
    ----------
    coordinate_grid :
        The image coordinates.
    grid_spacing :
        The grid spacing of ``coordinate_grid``.
    cutoff :
        The cutoff radius as a fraction of half
        the smallest box dimension. By default, ``0.95``.
    rolloff :
        The rolloff width as a fraction of the smallest box dimension.
        By default, ``0.05``.

    Returns
    -------
    mask : `Array`, shape `shape`
        An array representing the circular mask.
    """

    coords_norm = jnp.linalg.norm(coordinate_grid_in_angstroms, axis=-1)
    r_cut = radius

    coords_cut = coords_norm > r_cut

    rolloff_width = rolloff * coords_norm.max()
    mask = 0.5 * (
        1
        + jnp.cos(
            (coords_norm - r_cut - rolloff_width) / rolloff_width * jnp.pi
        )
    )

    mask = jnp.where(coords_cut, 0.0, mask)
    mask = jnp.where(coords_norm <= r_cut - rolloff_width, 1.0, mask)

    return mask
