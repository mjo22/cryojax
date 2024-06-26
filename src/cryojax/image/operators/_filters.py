"""
Filters to apply to images in Fourier space
"""

import functools
import operator
from typing import Optional, overload

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact

from ...coordinates import make_frequency_grid
from .._edges import resize_with_crop_or_pad
from .._fft import irfftn, rfftn
from .._spectrum import powerspectrum
from ._operator import AbstractImageMultiplier


class AbstractFilter(AbstractImageMultiplier, strict=True):
    """Base class for computing and applying an image filter."""

    @overload
    def __call__(
        self, image: Complex[Array, "y_dim x_dim"]
    ) -> Complex[Array, "y_dim x_dim"]: ...

    @overload
    def __call__(
        self, image: Complex[Array, "z_dim y_dim x_dim"]
    ) -> Complex[Array, "z_dim y_dim x_dim"]: ...

    def __call__(
        self, image: Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"]
    ) -> Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"]:
        return image * jax.lax.stop_gradient(self.buffer)


class CustomFilter(AbstractFilter, strict=True):
    """Pass a custom filter as an array."""

    buffer: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]

    def __init__(
        self,
        filter: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"],
    ):
        self.buffer = filter


class InverseSincFilter(AbstractFilter, strict=True):
    """Apply sinc-correction to an image."""

    buffer: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]

    def __init__(
        self,
        frequency_grid_in_angstroms_or_pixels: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
        grid_spacing: float = 1.0,
    ):
        ndim = frequency_grid_in_angstroms_or_pixels.ndim - 1
        self.buffer = jax.lax.reciprocal(
            functools.reduce(
                operator.mul,
                [
                    jnp.sinc(frequency_grid_in_angstroms_or_pixels[..., i] * grid_spacing)
                    for i in range(ndim)
                ],
            )
        )


class LowpassFilter(AbstractFilter, strict=True):
    """Apply a low-pass filter to an image or volume, with
    a cosine soft-edge.
    """

    buffer: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]

    frequency_cutoff_fraction: Float[Array, ""]
    rolloff_fraction: Float[Array, ""]

    def __init__(
        self,
        frequency_grid_in_angstroms_or_pixels: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
        grid_spacing: float | Float[Array, ""] = 1.0,
        frequency_cutoff_fraction: float | Float[Array, ""] = 0.95,
        rolloff_fraction: float | Float[Array, ""] = 0.05,
    ):
        """**Arguments:**

        - `frequency_grid_in_angstroms_or_pixels`:
            The frequency grid of the image or volume.
        - `grid_spacing`:
            The pixel or voxel size of `frequency_grid_in_angstroms_or_pixels`.
        - `frequency_cutoff_fraction`:
            The cutoff frequency as a fraction of the Nyquist frequency.
            By default, `0.95`.
        - `rolloff_fraction`:
            The rolloff width as a fraction of the Nyquist frequency.
            By default, ``0.05``.
        """
        self.frequency_cutoff_fraction = jnp.asarray(frequency_cutoff_fraction)
        self.rolloff_fraction = jnp.asarray(rolloff_fraction)
        self.buffer = _compute_lowpass_filter(
            frequency_grid_in_angstroms_or_pixels,
            jnp.asarray(grid_spacing),
            self.frequency_cutoff_fraction,
            self.rolloff_fraction,
        )


class WhiteningFilter(AbstractFilter, strict=True):
    """Compute a whitening filter from an image. This is taken
    to be the inverse square root of the 2D radially averaged
    power spectrum.

    This implementation follows the cisTEM whitening filter
    algorithm.
    """

    buffer: Inexact[Array, "y_dim x_dim"]

    def __init__(
        self,
        image: Float[Array, "image_y_dim image_x_dim"],
        shape: Optional[tuple[int, int]] = None,
        interpolation_mode: str = "nearest",
    ):
        """**Arguments:**

        - `image`:
            The image from which to compute the power spectrum.
        - `shape`:
            The shape of the resulting filter. This downsamples or
            upsamples the filter by cropping or padding in real space.
        - `interpolation_mode`:
            The method of interpolating the binned, radially averaged
            power spectrum onto a 2D grid. Either `nearest` or `linear`.
        """
        if shape is not None:
            if shape[0] > image.shape[0] or shape[1] > image.shape[1]:
                raise ValueError(
                    "The requested shape at which to compute the "
                    "whitening filter is larger than the shape of "
                    "the image from which to compute the filter. "
                    f"The requested shape was {shape} and the image "
                    f"shape was {image.shape}."
                )
        self.buffer = _compute_whitening_filter(
            image, shape, interpolation_mode=interpolation_mode
        )


@overload
def _compute_lowpass_filter(
    frequency_grid: Float[Array, "y_dim x_dim 2"],
    grid_spacing: Float[Array, ""],
    cutoff_fraction: Float[Array, ""],
    rolloff_fraction: Float[Array, ""],
) -> Float[Array, "y_dim x_dim"]: ...


@overload
def _compute_lowpass_filter(
    frequency_grid: Float[Array, "z_dim y_dim x_dim 3"],
    grid_spacing: Float[Array, ""],
    cutoff_fraction: Float[Array, ""],
    rolloff_fraction: Float[Array, ""],
) -> Float[Array, "z_dim y_dim x_dim"]: ...


def _compute_lowpass_filter(
    frequency_grid: Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"],
    grid_spacing: Float[Array, ""],
    cutoff_fraction: Float[Array, ""],
    rolloff_fraction: Float[Array, ""],
) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
    k_max = 1.0 / (2.0 * grid_spacing)
    cutoff_radius = cutoff_fraction * k_max
    rolloff_width = rolloff_fraction * k_max

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
    image: Float[Array, "y_dim x_dim"],
    shape: Optional[tuple[int, int]] = None,
    interpolation_mode: str = "nearest",
) -> Float[Array, "{shape[0]} {shape[1]}"]:
    # Make coordinates
    image_frequency_grid_in_angstroms = make_frequency_grid(image.shape)
    # Transform to fourier space
    fourier_image = rfftn(image)
    # Compute norms
    radial_frequency_grid = jnp.linalg.norm(image_frequency_grid_in_angstroms, axis=-1)
    # Compute power spectrum
    _, gridded_spectrum, _ = powerspectrum(
        fourier_image,
        radial_frequency_grid,
        to_grid=True,
        interpolation_mode=interpolation_mode,
        k_max=jnp.sqrt(2) / 2.0,
    )  # type: ignore
    if shape is not None:
        gridded_spectrum = rfftn(
            resize_with_crop_or_pad(
                irfftn(gridded_spectrum, s=image.shape), shape, pad_mode="edge"
            )
        ).real
    # Compute inverse square root
    whitening_filter = jax.lax.rsqrt(gridded_spectrum)
    # Divide filter by maximum, excluding zero mode
    maximum = jnp.max(jnp.delete(whitening_filter, jnp.asarray((0, 0), dtype=int)))
    whitening_filter /= maximum
    # Set zero mode manually to 1 (this diverges from the cisTEM
    # algorithm).
    whitening_filter = whitening_filter.at[0, 0].set(1.0)

    return whitening_filter
