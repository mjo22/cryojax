"""
Interpolation routines.
"""

__all__ = ["resize", "map_coordinates"]

from typing import Any

from jax._src.third_party.scipy.interpolate import RegularGridInterpolator
from jax.scipy.ndimage import map_coordinates as _map_coordinates
from jax.image import resize as _resize
from ..core import Array


def resize(image: Array, shape: tuple[int, int], method="lanczos5", **kwargs):
    """
    Resize an image with interpolation.

    Wraps ``jax.image.resize``.
    """
    return _resize(image, shape, method, **kwargs)


def interpn(points: Array, values: Array, xi: Array, **kwargs: Any):
    """
    Interpolate a set of points on a grid with a
    given coordinate system onto a new coordinate system.

    Wraps ``jax._src.third_party.scipy.interpolate.RegularGridInterpolator``.
    """
    interpolator = RegularGridInterpolator(points, values, **kwargs)

    return interpolator(xi)


def map_coordinates(input: Array, coordinates: Array, order=1, **kwargs: Any):
    """
    Interpolate a set of points on a grid with a
    given coordinate system onto a new coordinate system.

    Wraps ``jax.ndimage.map_coordinates``.
    """

    return _map_coordinates(input, coordinates, order, **kwargs)
