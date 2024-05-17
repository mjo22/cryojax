"""Routines for downsampling arrays"""

import math
from typing import cast, overload

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
    - `downsample_factor`:
        A scale factor at which to downsample `image_or_volume`
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
        downsampled_array = downsample_to_shape_with_fourier_cropping(
            image, new_shape, is_real=True
        )
    elif image_or_volume.ndim == 3:
        volume = image_or_volume
        new_shape = (
            int(volume.shape[0] / downsampling_factor),
            int(volume.shape[1] / downsampling_factor),
            int(volume.shape[2] / downsampling_factor),
        )
        downsampled_array = downsample_to_shape_with_fourier_cropping(
            volume, new_shape, is_real=True
        )
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


@overload
def downsample_to_shape_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _"],
    downsampled_shape_in_real_space: tuple[int, int],
    is_real: bool = True,
    half_space: bool = True,
) -> Inexact[Array, "_ _"]: ...


@overload
def downsample_to_shape_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _ _"],
    downsampled_shape_in_real_space: tuple[int, int, int],
    is_real: bool = True,
    half_space: bool = True,
) -> Inexact[Array, "_ _ _"]: ...


def downsample_to_shape_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"],
    downsampled_shape_in_real_space: tuple[int, int] | tuple[int, int, int],
    is_real: bool = True,
    half_space: bool = True,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    """Downsample an array to a specified shape using fourier cropping.

    **Arguments:**

    - `image_or_volume`: The image or volume array to downsample.
    - `downsampled_shape_in_real_space`:
        The new shape after fourier cropping.
    - `is_real`:
        If `True`, the `image_or_volume` is given in real space.
        If `False`, the fourier convention is used with the zero
        frequency component in the corner.
    - `half_space`:
        If `True` and `is_real = False`, the `image_or_volume`
        is the result of an FFT using hermitian symmetry.

    **Returns:**

    The downsampled `image_or_volume`, at the new real-space shape
    `downsampled_shape_in_real_space`. The `image_or_volume` is
    returned in fourier-space if `is_real = False`.
    """
    n_pixels, new_n_pixels = (
        image_or_volume.size,
        math.prod(downsampled_shape_in_real_space),
    )
    average_preserving_scale_factor = new_n_pixels / n_pixels
    if is_real:
        cropped_fourier_array = average_preserving_scale_factor * crop_to_shape(
            jnp.fft.fftshift(jnp.fft.fftn(image_or_volume)),
            downsampled_shape_in_real_space,
        )
        downsampled_array = jnp.fft.ifftn(jnp.fft.ifftshift(cropped_fourier_array))
        return (
            downsampled_array.real
            if jnp.issubdtype(image_or_volume.dtype, jnp.floating)
            else downsampled_array
        )
    else:
        if half_space:
            axes = tuple(range(image_or_volume.ndim - 1))
            if image_or_volume.ndim == 2:
                downsampled_shape_in_real_space = cast(
                    tuple[int, int], downsampled_shape_in_real_space
                )
                downsampled_shape_in_fourier_space = (
                    downsampled_shape_in_real_space[0],
                    downsampled_shape_in_real_space[1] // 2 + 1,
                )
            else:
                downsampled_shape_in_real_space = cast(
                    tuple[int, int, int], downsampled_shape_in_real_space
                )
                downsampled_shape_in_fourier_space = (
                    downsampled_shape_in_real_space[0],
                    downsampled_shape_in_real_space[1],
                    downsampled_shape_in_real_space[2] // 2 + 1,
                )
            cropped_fourier_array = average_preserving_scale_factor * crop_to_shape(
                jnp.fft.fftshift(image_or_volume, axes=axes),
                downsampled_shape_in_fourier_space,
            )
            return jnp.fft.ifftshift(cropped_fourier_array, axes=axes)
        else:
            downsampled_shape_in_fourier_space = downsampled_shape_in_real_space
            cropped_fourier_array = average_preserving_scale_factor * crop_to_shape(
                jnp.fft.fftshift(image_or_volume), downsampled_shape_in_fourier_space
            )
            return jnp.fft.ifftshift(cropped_fourier_array)
