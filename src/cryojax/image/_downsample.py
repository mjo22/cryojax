"""Routines for downsampling arrays"""

from typing import overload

import jax.numpy as jnp
from jaxtyping import Array, Inexact

from ._edges import crop_to_shape
from ._fft import fftn, ifftn


@overload
def downsample_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _ _"],
    downsampling_factor: float | int,
    get_real: bool = True,
) -> Inexact[Array, "_ _ _"]: ...


@overload
def downsample_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _"],
    downsampling_factor: float | int,
    get_real: bool = True,
) -> Inexact[Array, "_ _"]: ...


def downsample_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"],
    downsampling_factor: float | int,
    get_real: bool = True,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    """Downsample an array using fourier cropping.

    **Arguments:**

    - `image_or_volume`: The image or volume array to downsample.
    - `downsample_factor`:
        A scale factor at which to downsample `image_or_volume`
        by. Must be a value greater than `1`.
    - `get_real`:
        If `False`, the `image_or_volume` is returned in fourier space.

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
        downsampled_array = downsample_to_shape_with_fourier_cropping(
            image, new_shape, get_real=get_real
        )
    elif image_or_volume.ndim == 3:
        volume = image_or_volume
        new_shape = (
            int(volume.shape[0] / downsampling_factor),
            int(volume.shape[1] / downsampling_factor),
            int(volume.shape[2] / downsampling_factor),
        )
        downsampled_array = downsample_to_shape_with_fourier_cropping(
            volume, new_shape, get_real=get_real
        )
    else:
        raise ValueError(
            "`downsample_with_fourier_cropping` can only crop images and volumes. "
            f"Got an array with number of dimensions {image_or_volume.ndim}."
        )

    if get_real:
        return (
            downsampled_array.real
            if jnp.issubdtype(image_or_volume.dtype, jnp.floating)
            else downsampled_array
        )
    else:
        return downsampled_array


@overload
def downsample_to_shape_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _"],
    downsampled_shape: tuple[int, int],
    get_real: bool = True,
) -> Inexact[Array, "_ _"]: ...


@overload
def downsample_to_shape_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _ _"],
    downsampled_shape: tuple[int, int, int],
    get_real: bool = True,
) -> Inexact[Array, "_ _ _"]: ...


def downsample_to_shape_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"],
    downsampled_shape: tuple[int, int] | tuple[int, int, int],
    get_real: bool = True,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    """Downsample an array to a specified shape using fourier cropping.

    **Arguments:**

    - `image_or_volume`: The image or volume array to downsample.
    - `downsampled_shape`:
        The new shape after fourier cropping.
    - `get_real`:
        If `False`, the `image_or_volume` is returned in fourier space.

    **Returns:**

    The downsampled `image_or_volume`, at the new real-space shape
    `downsampled_shape`. If `get_real = False`, return
    the downsampled array in fourier space assuming hermitian symmetry,
    with the zero frequency component in the corner.
    """
    fourier_array = jnp.fft.fftshift(fftn(image_or_volume))
    cropped_fourier_array = crop_to_shape(fourier_array, downsampled_shape)
    if get_real:
        return ifftn(jnp.fft.ifftshift(cropped_fourier_array))
    else:
        return jnp.fft.ifftshift(cropped_fourier_array)[
            ..., : downsampled_shape[-1] // 2 + 1
        ]
