"""
Interpolation routines.
"""

__all__ = ["resize", "scale", "scale_and_translate", "map_coordinates"]

from typing import Union, Any
from jaxtyping import Array, Float

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates as _map_coordinates
from jax.image import resize as _resize
from jax.image import scale_and_translate as _scale_and_translate


def resize(
    image: Array,
    shape: tuple[int, int],
    method="lanczos5",
    align_corners: bool = True,
    **kwargs: Any
) -> Array:
    """
    Resize an image with interpolation.

    Wraps ``jax.image.resize``.
    """
    if align_corners:
        return _resize_with_aligned_corners(image, shape, method, **kwargs)
    else:
        return _resize(image, shape, method, **kwargs)


def scale_and_translate(
    image: Array,
    shape: tuple[int, int],
    scale: Float[Array, "2"],
    translation: Float[Array, "2"],
    method="lanczos5",
    **kwargs
) -> Array:
    """
    Resize, scale, and translate an image with interpolation.

    Wraps ``jax.image.scale_and_translate``.
    """
    image = jnp.asarray(image)
    spatial_dims = (0, 1)
    N1, N2 = image.shape
    translation += (1 - scale) * jnp.array([N1 // 2, N2 // 2], dtype=float)
    return _scale_and_translate(
        image, shape, spatial_dims, scale, translation, method, **kwargs
    )


def scale(
    image: Array,
    shape: tuple[int, int],
    scale: Float[Array, "2"],
    method="lanczos5",
    **kwargs
) -> Array:
    """
    Resize and scale an image with interpolation.

    Wraps ``jax.image.scale_and_translate``.
    """
    translation = jnp.array([0.0, 0.0])
    return scale_and_translate(
        image, shape, scale, translation, method=method, **kwargs
    )


def map_coordinates(
    input: Array, coordinates: Array, order=1, mode="wrap", cval=0.0
) -> Array:
    """
    Interpolate a set of points in fourier space on a grid
    with a given coordinate system onto a new coordinate system.
    """
    input, coordinates = jnp.asarray(input), jnp.asarray(coordinates)
    N1, N2, N3 = input.shape
    box_shape = jnp.array([N1, N2, N3], dtype=float)[:, None, None, None]
    coordinates = jnp.transpose(coordinates, axes=[3, 0, 1, 2])
    # Flip negative valued frequencies to get the logical coordinates.
    coordinates = jnp.where(
        coordinates < 0, box_shape + coordinates, coordinates
    )
    return _map_coordinates(input, coordinates, order, mode=mode, cval=cval)


def _resize_with_aligned_corners(
    image: Array,
    shape: tuple[int, ...],
    method: Union[str, jax.image.ResizeMethod],
    antialias: bool = False,
) -> Array:
    """
    Alternative to jax.image.resize(), which emulates
    align_corners=True in PyTorch's interpolation functions.

    Adapted from https://github.com/google/jax/issues/11206.
    ."""
    image = jnp.asarray(image)
    spatial_dims = tuple(
        i
        for i in range(len(shape))
        if not jax.core.symbolic_equal_dim(image.shape[i], shape[i])
    )
    scale = jnp.array(
        [(shape[i] - 1.0) / (image.shape[i] - 1.0) for i in spatial_dims]
    )
    translation = -(scale / 2.0 - 0.5)
    return _scale_and_translate(
        image,
        shape,
        method=method,
        scale=scale,
        spatial_dims=spatial_dims,
        translation=translation,
        antialias=antialias,
    )
