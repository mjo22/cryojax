"""
Routines for dealing with image edges.
"""

__all__ = ["bound", "crop", "pad"]

from typing import Union
from jaxtyping import Array, Float

import jax
import jax.numpy as jnp

from ..typing import Cloud, CloudCoords2D, Image, Volume


@jax.jit
def bound(
    density: Cloud, coords: CloudCoords2D, box_size: Float[Array, "2"]
) -> Cloud:
    """
    Use a boolean mask to set density values out of
    bounds to zero. The mask is ``True`` for
    all points outside of the box_size, and False otherwise.
    """
    coords, box_size = jnp.asarray(coords), jnp.asarray(box_size)
    x, y = coords.T
    Lx, Ly = box_size[0], box_size[1]
    x_mask = jnp.logical_or(x < -Lx / 2, x >= Lx / 2)
    y_mask = jnp.logical_or(y < -Ly / 2, y >= Ly / 2)
    mask = jnp.logical_or(x_mask, y_mask)
    masked_density = jnp.where(mask, complex(0.0), density)

    return masked_density


def crop(image: Image, shape: tuple[int, ...]) -> Image:
    """
    Crop an image to a new shape.
    """
    if image.ndim == 2:
        M1, M2 = image.shape
        xc, yc = M1 // 2, M2 // 2
        w, h = shape
        cropped = image[
            xc - w // 2 : xc + w // 2 + w % 2, yc - h // 2 : yc + h // 2 + h % 2
        ]
    elif image.ndim == 3:
        M1, M2, M3 = image.shape
        xc, yc, zc = M1 // 2, M2 // 2, M3 // 2
        w, h, d = shape
        cropped = image[
            xc - w // 2 : xc + w // 2 + w % 2,
            yc - h // 2 : yc + h // 2 + h % 2,
            zc - d // 2 : zc + d // 2 + d % 2,
        ]
    return cropped


def pad(
    image: Union[Image, Volume], shape: tuple[int, ...], **kwargs
) -> Union[Image, Volume]:
    """
    Pad an image or volume to a new shape.
    """
    if len(shape) == 2:
        x_pad = shape[0] - image.shape[0]
        y_pad = shape[1] - image.shape[1]
        padding = (
            (x_pad // 2, x_pad // 2 + x_pad % 2),
            (y_pad // 2, y_pad // 2 + y_pad % 2),
        )
    elif len(shape) == 3:
        x_pad = shape[0] - image.shape[0]
        y_pad = shape[1] - image.shape[1]
        z_pad = shape[2] - image.shape[2]
        padding = (
            (x_pad // 2, x_pad // 2 + x_pad % 2),
            (y_pad // 2, y_pad // 2 + y_pad % 2),
            (z_pad // 2, z_pad // 2 + z_pad % 2),
        )
    else:
        raise NotImplementedError(f"Cannot pad arrays with ndim={len(shape)}")
    return jnp.pad(image, padding, **kwargs)
