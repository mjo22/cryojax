"""
Filters to apply to images in Fourier space
"""

import math
from typing import Optional, overload

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact

from ...coordinates import make_frequency_grid
from .._average import interpolate_radial_average_on_grid
from .._edges import resize_with_crop_or_pad
from .._fft import irfftn, rfftn
from .._fourier_statistics import compute_binned_powerspectrum
from ._operator import AbstractImageMultiplier


class AbstractFilter(AbstractImageMultiplier, strict=True):
    """Base class for computing and applying an image filter."""

    @overload
    def __call__(
        self, image: Complex[Array, "y_dim x_dim"]
    ) -> Complex[Array, "y_dim x_dim"]: ...

    @overload
    def __call__(  # type: ignore
        self, image: Complex[Array, "z_dim y_dim x_dim"]
    ) -> Complex[Array, "z_dim y_dim x_dim"]: ...

    def __call__(
        self, image: Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"]
    ) -> Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"]:
        return image * jax.lax.stop_gradient(self.array)


FilterLike = AbstractFilter | AbstractImageMultiplier


class CustomFilter(AbstractFilter, strict=True):
    """Pass a custom filter as an array."""

    array: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]

    def __init__(
        self,
        filter: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"],
    ):
        self.array = filter


class LowpassFilter(AbstractFilter, strict=True):
    """Apply a low-pass filter to an image or volume, with
    a cosine soft-edge.
    """

    array: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]

    def __init__(
        self,
        frequency_grid_in_angstroms_or_pixels: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
        grid_spacing: float | Float[Array, ""] = 1.0,
        frequency_cutoff_fraction: float | Float[Array, ""] = 0.95,
        rolloff_width_fraction: float | Float[Array, ""] = 0.05,
    ):
        """**Arguments:**

        - `frequency_grid_in_angstroms_or_pixels`:
            The frequency grid of the image or volume.
        - `grid_spacing`:
            The pixel or voxel size of `frequency_grid_in_angstroms_or_pixels`.
        - `frequency_cutoff_fraction`:
            The cutoff frequency as a fraction of the Nyquist frequency.
            By default, `0.95`.
        - `rolloff_width_fraction`:
            The rolloff width as a fraction of the Nyquist frequency.
            By default, ``0.05``.
        """
        self.array = _compute_lowpass_filter(
            frequency_grid_in_angstroms_or_pixels,
            jnp.asarray(grid_spacing),
            jnp.asarray(frequency_cutoff_fraction),
            jnp.asarray(rolloff_width_fraction),
        )


class HighpassFilter(AbstractFilter, strict=True):
    """Apply a low-pass filter to an image or volume, with
    a cosine soft-edge.
    """

    array: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]

    def __init__(
        self,
        frequency_grid_in_angstroms_or_pixels: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
        grid_spacing: float | Float[Array, ""] = 1.0,
        frequency_cutoff_fraction: float | Float[Array, ""] = 0.95,
        rolloff_width_fraction: float | Float[Array, ""] = 0.05,
    ):
        """**Arguments:**

        - `frequency_grid_in_angstroms_or_pixels`:
            The frequency grid of the image or volume.
        - `grid_spacing`:
            The pixel or voxel size of `frequency_grid_in_angstroms_or_pixels`.
        - `frequency_cutoff_fraction`:
            The cutoff frequency as a fraction of the Nyquist frequency.
            By default, `0.95`.
        - `rolloff_width_fraction`:
            The rolloff width as a fraction of the Nyquist frequency.
            By default, ``0.05``.
        """
        self.array = 1.0 - _compute_lowpass_filter(
            frequency_grid_in_angstroms_or_pixels,
            jnp.asarray(grid_spacing),
            jnp.asarray(frequency_cutoff_fraction),
            jnp.asarray(rolloff_width_fraction),
        )


class WhiteningFilter(AbstractFilter, strict=True):
    """Compute a whitening filter from an image. This is taken
    to be the inverse square root of the 2D radially averaged
    power spectrum.

    This implementation follows the cisTEM whitening filter
    algorithm.
    """

    array: Inexact[Array, "y_dim x_dim"]

    def __init__(
        self,
        image_or_image_stack: (
            Float[Array, "image_y_dim image_x_dim"]
            | Float[Array, "n_images image_y_dim image_x_dim"]
        ),
        shape: Optional[tuple[int, int]] = None,
        *,
        interpolation_mode: str = "linear",
        outputs_squared: bool = False,
    ):
        """**Arguments:**

        - `image_or_image_stack`:
            The image (or stack of images) from which to compute the power spectrum.
        - `shape`:
            The shape of the resulting filter. This downsamples or
            upsamples the filter by cropping or padding in real space.
        - `interpolation_mode`:
            The method of interpolating the binned, radially averaged
            power spectrum onto a 2D grid. Either `nearest` or `linear`.
        - `outputs_squared`:
            If `False`, the whitening filter is the inverse square root of the image
            power. If `True`, the filter is the inverse of the image power.
        """
        image_stack = (
            jnp.expand_dims(image_or_image_stack, 0)
            if image_or_image_stack.ndim == 2
            else image_or_image_stack
        )
        if shape is not None:
            if shape[-2] > image_stack.shape[-2] or shape[-1] > image_stack.shape[-1]:
                raise ValueError(
                    "The requested shape at which to compute the "
                    "whitening filter is larger than the shape of "
                    "the image from which to compute the filter. "
                    f"The requested shape was {shape} and the image "
                    f"shape was {image_stack.shape[-2:]}."
                )
        self.array = _compute_whitening_filter(
            image_stack,
            shape,
            interpolation_mode=interpolation_mode,
            outputs_squared=outputs_squared,
        )


def _compute_lowpass_filter(
    frequency_grid: Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"],
    grid_spacing: Float[Array, ""],
    cutoff_fraction: Float[Array, ""],
    rolloff_width_fraction: Float[Array, ""],
) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
    k_max = 1.0 / (2.0 * grid_spacing)
    cutoff_radius = cutoff_fraction * k_max
    rolloff_width = rolloff_width_fraction * k_max

    radial_frequency_grid = jnp.linalg.norm(frequency_grid, axis=-1)

    def compute_filter_at_frequency(radial_frequency):
        return jnp.where(
            radial_frequency <= cutoff_radius,
            1.0,
            jnp.where(
                radial_frequency > cutoff_radius + rolloff_width,
                0.0,
                0.5
                * (
                    1
                    + jnp.cos(jnp.pi * (radial_frequency - cutoff_radius) / rolloff_width)
                ),
            ),
        )

    compute_filter = (
        jax.vmap(jax.vmap(compute_filter_at_frequency))
        if radial_frequency_grid.ndim == 2
        else jax.vmap(jax.vmap(jax.vmap(compute_filter_at_frequency)))
    )

    return compute_filter(radial_frequency_grid)


def _compute_whitening_filter(
    image_stack: Float[Array, "n_images y_dim x_dim"],
    shape: Optional[tuple[int, int]] = None,
    interpolation_mode: str = "linear",
    outputs_squared: bool = False,
) -> Float[Array, "{shape[0]} {shape[1]}"]:
    # Make coordinates
    frequency_grid = make_frequency_grid(image_stack.shape[1:])
    # Transform to fourier space
    n_pixels = math.prod(image_stack.shape[1:])
    fourier_image_stack = rfftn(image_stack, axes=(1, 2)) / jnp.sqrt(n_pixels)
    # Compute norms
    radial_frequency_grid = jnp.linalg.norm(frequency_grid, axis=-1)
    # Compute stack of power spectra
    compute_powerspectrum_stack = jax.vmap(
        lambda im, freq: compute_binned_powerspectrum(
            im, freq, maximum_frequency=math.sqrt(2) / 2
        ),
        in_axes=[0, None],
        out_axes=(0, None),
    )
    radially_averaged_powerspectrum_stack, frequency_bins = compute_powerspectrum_stack(
        fourier_image_stack, radial_frequency_grid
    )
    # Take the mean over the stack
    radially_averaged_powerspectrum = jnp.mean(
        radially_averaged_powerspectrum_stack, axis=0
    )
    # Put onto a grid
    radially_averaged_powerspectrum_on_grid = interpolate_radial_average_on_grid(
        radially_averaged_powerspectrum,
        frequency_bins,
        radial_frequency_grid,
        interpolation_mode=interpolation_mode,
    )
    # Resize to be the desired shape
    if shape is not None:
        new_shape = shape
        radially_averaged_powerspectrum_on_grid = rfftn(
            resize_with_crop_or_pad(
                irfftn(radially_averaged_powerspectrum_on_grid, s=image_stack.shape[1:]),
                shape,
                pad_mode="edge",
            )
        ).real
        # ... resizing and going back to fourier space can introduce negative values
        radially_averaged_powerspectrum_on_grid = jnp.where(
            radially_averaged_powerspectrum_on_grid < 0,
            0.0,
            radially_averaged_powerspectrum_on_grid,
        )
    else:
        new_shape = image_stack.shape[1], image_stack.shape[2]
    # Compute inverse square root (or inverse square)
    inverse_fn = jax.lax.reciprocal if outputs_squared else jax.lax.rsqrt
    whitening_filter = jnp.where(
        jnp.isclose(radially_averaged_powerspectrum_on_grid, 0.0),
        0.0,
        inverse_fn(radially_averaged_powerspectrum_on_grid),
    )
    # Set zero mode to 0, defining the filter to zero out these modes
    whitening_filter = whitening_filter.at[0, 0].set(0.0)
    # If the image size is even, there can be an issue with the nyquist corner
    if new_shape[0] % 2 == 0 and new_shape[1] % 2 == 0:
        whitening_filter = whitening_filter.at[new_shape[0] // 2, new_shape[1] // 2].set(
            0.0
        )

    return whitening_filter
