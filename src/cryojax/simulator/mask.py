"""
Masks to apply to images in real space.
"""

from __future__ import annotations

__all__ = [
    "compute_circular_mask",
    "Mask",
    "CircularMask",
]

from abc import abstractmethod
from typing import Any

import jax.numpy as jnp

from ..utils import make_coordinates
from ..core import field, Buffer
from ..typing import RealImage


class Mask(Buffer):
    """
    Base class for computing and applying an image mask.

    Attributes
    ----------
    shape :
        The image shape.
    mask :
        The mask. Note that this is automatically
        computed upon instantiation.
    """

    shape: tuple[int, int] = field()
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

    radius: float = field(static=True, default=0.95)
    rolloff: float = field(static=True, default=0.05)

    def evaluate(self, **kwargs: Any) -> RealImage:
        return compute_circular_mask(
            self.shape, self.radius, self.rolloff, **kwargs
        )


def compute_circular_mask(
    shape: tuple[int, int],
    cutoff: float = 0.95,
    rolloff: float = 0.05,
    **kwargs: Any,
) -> RealImage:
    """
    Create a circular mask.

    Parameters
    ----------
    shape :
        The shape of the mask. This is used to compute the image
        coordinates.
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
    coords = make_coordinates(shape, **kwargs)

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
