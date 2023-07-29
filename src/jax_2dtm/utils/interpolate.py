"""
Interpolation routines.
"""

__all__ = ["resize", "RegularGridInterpolator"]

from typing import Any

from jax._src.third_party.scipy.interpolate import RegularGridInterpolator
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
