"""
Routines for dealing with image edges.
"""

__all__ = ["crop", "pad", "crop_or_pad"]

import jax.numpy as jnp

from ..typing import Image, Volume


def crop(
    image: Image | Volume,
    shape: tuple[int, int] | tuple[int, int, int],
) -> Image | Volume:
    """
    Crop an image to a new shape.

    Will crop the last axes of the given Image or Volume.
    This is to support if they are really a stacks of images
    or volumes.
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
    image: Image | Volume,
    shape: tuple[int, int] | tuple[int, int, int],
    **kwargs,
) -> Image | Volume:
    """
    Pad an image or volume to a new shape.
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


def crop_or_pad(image: Image, shape: tuple[int, int], **kwargs) -> Image:
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
