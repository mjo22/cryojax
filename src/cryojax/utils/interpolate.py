"""
Interpolation routines.
"""

__all__ = ["resize", "scale", "scale_and_translate", "map_coordinates"]

from typing import Any

import jax.numpy as jnp
from jax._src.third_party.scipy.interpolate import RegularGridInterpolator
from jax.scipy.ndimage import map_coordinates as _map_coordinates
from jax.image import resize as _resize
from jax.image import scale_and_translate as _scale_and_translate
from ..core import Array


def resize(image: Array, shape: tuple[int, int], method="lanczos5", **kwargs):
    """
    Resize an image with interpolation.

    Wraps ``jax.image.resize``.
    """
    return _resize(image, shape, method, **kwargs)


def scale_and_translate(
    image: Array,
    shape: tuple[int, int],
    scale: Array,
    translation: Array,
    method="lanczos5",
    **kwargs
):
    """
    Resize, scale, and translate an image with interpolation.

    Wraps ``jax.image.scale_and_translate``.
    """
    spatial_dims = (0, 1)
    N1, N2 = image.shape
    translation += (1 - scale) * jnp.array([N2 // 2, N1 // 2], dtype=float)
    return _scale_and_translate(
        image, shape, spatial_dims, scale, translation, method, **kwargs
    )


def scale(
    image: Array,
    shape: tuple[int, int],
    scale: Array,
    method="lanczos5",
    **kwargs
):
    """
    Resize and scale an image with interpolation.

    Wraps ``jax.image.scale_and_translate``.
    """
    translation = jnp.array([0.0, 0.0])
    return scale_and_translate(
        image, shape, scale, translation, method=method, **kwargs
    )


def interpn(points: Array, values: Array, xi: Array, **kwargs: Any):
    """
    Interpolate a set of points on a grid with a
    given coordinate system onto a new coordinate system.

    Wraps ``jax._src.third_party.scipy.interpolate.RegularGridInterpolator``.
    """
    interpolator = RegularGridInterpolator(points, values, **kwargs)

    return interpolator(xi)


def map_coordinates(
    input: Array, coordinates: Array, order=1, mode="wrap", cval=0.0
):
    """
    Interpolate a set of points on a grid with a
    given coordinate system onto a new coordinate system.

    Wraps ``jax.ndimage.map_coordinates`` to match typical
    python coordinate indexing conventions and also
    ``cryojax`` conventions, which takes axes 0, 1, and 2
    to be axes x, y, and z.
    """
    N1, N2, N3 = input.shape
    box_shape = jnp.array([N3, N2, N1], dtype=float)[:, None, None, None]
    # x, y, z = jnp.transpose(coordinates, axes=[3, 1, 0, 2])
    # coordinates = jnp.stack([z, -y, x])
    x, y, z = jnp.transpose(coordinates, axes=[3, 0, 1, 2])
    coordinates = jnp.stack([z, x, y])
    # Flip negative valued frequencies to get the "array index coordinates".
    coordinates = jnp.where(
        coordinates < 0, box_shape + coordinates, coordinates
    )

    return _map_coordinates(input, coordinates, order, mode=mode, cval=cval)
