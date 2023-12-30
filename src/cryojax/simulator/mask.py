"""
Masks to apply to images in real space.
"""

from __future__ import annotations

__all__ = [
    "compute_circular_mask",
    "Mask",
    "ProductMask",
    "CircularMask",
]

from abc import abstractmethod
from typing import Any
from typing_extensions import override

import jax.numpy as jnp

from .manager import ImageManager

from ..core import field, Buffer
from ..typing import RealImage, ImageCoords


class Mask(Buffer):
    """
    Base class for computing and applying an image mask.

    Attributes
    ----------
    mask :
        The mask. Note that this is automatically
        computed upon instantiation.
    """

    mask: RealImage = field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any):
        self.mask = self.evaluate(*args, **kwargs)

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> RealImage:
        """Compute the mask."""
        raise NotImplementedError

    def __call__(self, image: RealImage) -> RealImage:
        """Apply the mask to an image."""
        return self.mask * image

    def __mul__(self, other: Mask) -> Mask:
        return ProductMask(self, other)

    def __rmul__(self, other: Mask) -> Mask:
        return ProductMask(other, self)


class ProductMask(Mask):
    """A helper to represent the product of two masks."""

    mask1: Mask = field()
    mask2: Mask = field()

    def evaluate(self) -> RealImage:
        return self.mask1.mask * self.mask2.mask


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

    manager: ImageManager = field()

    radius: float = field(static=True, default=0.95)
    rolloff: float = field(static=True, default=0.05)

    @override
    def evaluate(self, **kwargs: Any) -> RealImage:
        return compute_circular_mask(
            self.manager.coordinate_grid, self.radius, self.rolloff, **kwargs
        )


def compute_circular_mask(
    coords: ImageCoords,
    cutoff: float = 0.95,
    rolloff: float = 0.05,
    **kwargs: Any,
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
    kwargs :
        Keyword arguments passed to ``cryojax.utils.make_coordinates``.

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
