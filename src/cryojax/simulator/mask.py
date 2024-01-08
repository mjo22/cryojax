"""
Masks to apply to images in real space.
"""

from __future__ import annotations

__all__ = [
    "compute_circular_mask",
    "Mask",
    "MaskType",
    "CircularMask",
]

from abc import abstractmethod
from dataclasses import InitVar
from typing import Any, TypeVar, overload
from typing_extensions import override

import jax.numpy as jnp

from ..core import field, BufferModule
from ..typing import RealImage, ImageCoords


MaskType = TypeVar("MaskType", bound="Mask")
"""TypeVar for the Mask base class."""


class Mask(BufferModule):
    """
    Base class for computing and applying an image mask.

    Attributes
    ----------
    mask :
        The mask. Note that this is automatically
        computed upon instantiation.
    """

    coordinate_grid: InitVar[ImageCoords | None] = None
    mask: RealImage = field(init=False)

    def __post_init__(
        self, coordinate_grid: ImageCoords | None, **kwargs: Any
    ):
        self.mask = self.evaluate(coordinate_grid, **kwargs)

    @overload
    @abstractmethod
    def evaluate(self, coords: None, **kwargs: Any) -> RealImage:
        ...

    @overload
    @abstractmethod
    def evaluate(self, coords: ImageCoords, **kwargs: Any) -> RealImage:
        ...

    @abstractmethod
    def evaluate(
        self, coords: ImageCoords | None = None, **kwargs: Any
    ) -> RealImage:
        """Compute the filter."""
        raise NotImplementedError

    def __call__(self, image: RealImage) -> RealImage:
        """Apply the mask to an image."""
        return self.mask * image

    def __mul__(self, other: Mask) -> Mask:
        return _ProductMask(mask1=self, mask2=other)

    def __rmul__(self, other: Mask) -> Mask:
        return _ProductMask(mask1=other, mask2=self)


class _ProductMask(Mask):
    """A helper to represent the product of two filters."""

    mask1: MaskType = field()  # type: ignore
    mask2: MaskType = field()  # type: ignore

    @override
    def evaluate(
        self, coords: ImageCoords | None = None, **kwargs: Any
    ) -> RealImage:
        return self.mask1.mask * self.mask2.mask

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

    radius: float = field(static=True, default=0.95)
    rolloff: float = field(static=True, default=0.05)

    @override
    def evaluate(self, coords: ImageCoords | None, **kwargs: Any) -> RealImage:
        if coords is None:
            raise ValueError(
                "The coordinate grid must be given as an argument to the Mask."
            )
        else:
            return compute_circular_mask(coords, self.radius, self.rolloff)


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
