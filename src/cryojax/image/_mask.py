"""
Masks to apply to images in real space.
"""

from __future__ import annotations

__all__ = ["Mask", "MaskType", "CircularMask", "compute_circular_mask"]

from abc import abstractmethod
from typing import Any, TypeVar
from equinox import Module

import jax.numpy as jnp

from ..core import field
from ..typing import RealImage, ImageCoords


MaskType = TypeVar("MaskType", bound="Mask")
"""TypeVar for the Mask base class."""


class Mask(Module):
    """
    Base class for computing and applying an image mask.

    Attributes
    ----------
    mask :
        The mask. Note that this is automatically
        computed upon instantiation.
    """

    mask: RealImage

    @abstractmethod
    def __init__(self, **kwargs: Any):
        """Compute the mask. This must be overwritten in subclasses."""
        super().__init__(**kwargs)

    def __call__(self, image: RealImage) -> RealImage:
        """Apply the mask to an image."""
        return self.mask * image

    def __mul__(self: MaskType, other: MaskType) -> _ProductMask:
        return _ProductMask(mask1=self, mask2=other)

    def __rmul__(self: MaskType, other: MaskType) -> _ProductMask:
        return _ProductMask(mask1=other, mask2=self)


class _ProductMask(Mask):
    """A helper to represent the product of two filters."""

    mask1: Mask
    mask2: Mask

    def __init__(self, mask1: MaskType, mask2: MaskType):
        self.mask1 = mask1
        self.mask2 = mask2
        self.mask = mask1.mask * mask2.mask

    def __repr__(self):
        return f"{repr(self.mask1)} * {repr(self.mask2)}"


class CircularMask(Mask):
    """
    Apply a circular mask to an image.

    See documentation for
    ``cryojax.simulator.compute_circular_mask``
    for more information.

    Attributes
    ----------
    radius :
        By default, ``0.95``.
    rolloff :
        By default, ``0.05``.
    """

    radius: float = field(static=True)
    rolloff: float = field(static=True)

    def __init__(
        self,
        freqs: ImageCoords,
        radius: float = 0.95,
        rolloff: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.radius = radius
        self.rolloff = rolloff
        self.mask = compute_circular_mask(freqs, self.radius, self.rolloff)


def compute_circular_mask(
    coords: ImageCoords, cutoff: float = 0.95, rolloff: float = 0.05
) -> RealImage:
    """
    Create a circular mask.

    Parameters
    ----------
    shape :
        The image coordinates.
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
