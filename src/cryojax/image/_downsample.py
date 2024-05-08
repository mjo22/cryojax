"""Routines for downsampling arrays"""

from typing import overload

import jax.numpy as jnp
from jaxtyping import Array, Inexact

from ._edges import crop_to_shape


@overload
def downsample_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _ _"],
    downsampling_factor: float | int,
) -> Inexact[Array, "_ _ _"]: ...


@overload
def downsample_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _"],
    downsampling_factor: float | int,
) -> Inexact[Array, "_ _"]: ...


def downsample_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"],
    downsampling_factor: float | int,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    """Downsample an array using fourier cropping.

    **Arguments:**

    - `image_or_volume`: The image or volume array to downsample.
    - `downsample_factor`: A scale factor at which to downsample `image_or_volume`
                           by. Must be a value greater than `1`.

    **Returns:**

    The downsampled `image_or_volume`, with shape reduced by a factor of
    `downsample_factor`.
    """
    downsampling_factor = float(downsampling_factor)
    if downsampling_factor < 1.0:
        raise ValueError("`downsampling_factor` must be greater than 1.0")
    if image_or_volume.ndim == 2:
        image = image_or_volume
        new_shape = (
            int(image.shape[0] / downsampling_factor),
            int(image.shape[1] / downsampling_factor),
        )
        downsampled_array = _downsample_array_to_shape(image, new_shape)
    elif image_or_volume.ndim == 3:
        volume = image_or_volume
        new_shape = (
            int(volume.shape[0] / downsampling_factor),
            int(volume.shape[1] / downsampling_factor),
            int(volume.shape[2] / downsampling_factor),
        )
        downsampled_array = _downsample_array_to_shape(volume, new_shape)
    else:
        raise ValueError(
            "`downsample_with_fourier_cropping` can only crop images and volumes. "
            f"Got an array with number of dimensions {image_or_volume.ndim}."
        )

    return (
        downsampled_array.real
        if jnp.issubdtype(image_or_volume.dtype, jnp.floating)
        else downsampled_array
    )


def _downsample_array_to_shape(array, new_shape):
    fourier_array = jnp.fft.fftshift(jnp.fft.fftn(array))
    cropped_fourier_array = crop_to_shape(fourier_array, new_shape)
    downsampled_array = jnp.fft.ifftn(jnp.fft.ifftshift(cropped_fourier_array))
    return downsampled_array
