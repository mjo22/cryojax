"""
Routines for dealing with image edges.
"""

__all__ = ["crop", "pad", "resize_with_crop_or_pad"]

import jax.numpy as jnp
from jaxtyping import Shaped

from ..typing import Image, Volume


def crop(
    image: Shaped[Image, "..."] | Shaped[Volume, "..."],
    shape: tuple[int, int] | tuple[int, int, int],
) -> Shaped[Image, "..."] | Shaped[Volume, "..."]:
    """
    Crop an image to a new shape.

    The input image or volume may have leading axes. Only
    the last axes are padded.
    """
    if len(shape) == 2:
        M1, M2 = image.shape
        xc, yc = M1 // 2, M2 // 2
        w, h = shape
        cropped = image[
            ...,
            xc - w // 2 : xc + w // 2 + w % 2,
            yc - h // 2 : yc + h // 2 + h % 2,
        ]
    elif len(shape) == 3:
        M1, M2, M3 = image.shape
        xc, yc, zc = M1 // 2, M2 // 2, M3 // 2
        w, h, d = shape
        cropped = image[
            ...,
            xc - w // 2 : xc + w // 2 + w % 2,
            yc - h // 2 : yc + h // 2 + h % 2,
            zc - d // 2 : zc + d // 2 + d % 2,
        ]
    else:
        raise NotImplementedError(f"Cannot crop arrays with ndim={len(shape)}")
    return cropped


def pad(
    image: Shaped[Image, "..."] | Shaped[Volume, "..."],
    shape: tuple[int, int] | tuple[int, int, int],
    **kwargs,
) -> Shaped[Image, "..."] | Shaped[Volume, "..."]:
    """
    Pad an image or volume to a new shape.

    The input image or volume may have leading axes. Only
    the last axes are padded.
    """
    n_extra_dims = image.ndim - len(shape)
    extra_padding = tuple([(0, 0) for _ in range(n_extra_dims)])
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
    return jnp.pad(image, (*extra_padding, *padding), **kwargs)


def resize_with_crop_or_pad(
    image: Shaped[Image, "..."], shape: tuple[int, int], **kwargs
) -> Shaped[Image, "..."]:
    """
    Resize an image to a new shape using padding and cropping
    """
    N1, N2 = image.shape
    M1, M2 = shape
    if N1 >= M1 and N2 >= M2:
        image = crop(image, shape)
    elif N1 <= M1 and N2 <= M2:
        image = pad(image, shape, **kwargs)
    elif N1 <= M1 and N2 >= M2:
        image = crop(image, (N1, M2))
        image = pad(image, (M1, M2), **kwargs)
    else:
        image = crop(image, (M1, N2))
        image = pad(image, (M1, M2), **kwargs)

    return image
