"""
Filters to apply to images in Fourier space
"""

import functools
import operator
from typing import Optional, overload

import jax
import jax.numpy as jnp
from equinox import field
from jaxtyping import Array, Complex, Float, Inexact

from ...coordinates import make_frequencies
from .._edges import resize_with_crop_or_pad
from .._fft import irfftn, rfftn
from .._spectrum import powerspectrum
from ._operator import AbstractImageMultiplier


class AbstractFilter(AbstractImageMultiplier, strict=True):
    """
    Base class for computing and applying an image filter.
    """

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
    """
    Pass a custom filter as an array.
    """

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
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
        grid_spacing: float = 1.0,
    ):
        ndim = frequency_grid.ndim - 1
        self.buffer = jax.lax.reciprocal(
            functools.reduce(
                operator.mul,
                [jnp.sinc(frequency_grid[..., i] * grid_spacing) for i in range(ndim)],
            )
        )


class LowpassFilter(AbstractFilter, strict=True):
    """
    Apply a low-pass filter to an image.

    Attributes
    ----------
    cutoff :
        By default, ``0.95``. This cuts off modes as
        a fraction of the Nyquist frequency. To keep
        modes up to Nyquist, ``cutoff = 1.0``
    rolloff :
        By default, ``0.05``.
    """

    buffer: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]

    cutoff: float = field(static=True)
    rolloff: float = field(static=True)

    def __init__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
        grid_spacing: float = 1.0,
        cutoff: float = 0.95,
        rolloff: float = 0.05,
    ):
        self.cutoff = cutoff
        self.rolloff = rolloff
        self.buffer = _compute_lowpass_filter(
            frequency_grid, grid_spacing, self.cutoff, self.rolloff
        )


class WhiteningFilter(AbstractFilter, strict=True):
    """
    Apply a whitening filter to an image.
    """

    buffer: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]

    def __init__(
        self,
        micrograph: Float[Array, "y_dim x_dim"],
        shape: Optional[tuple[int, int]] = None,
        interpolation_mode: str = "nearest",
    ):
        self.buffer = _compute_whitening_filter(
            micrograph, shape, interpolation_mode=interpolation_mode
        )


@overload
def _compute_lowpass_filter(
    frequency_grid: Float[Array, "y_dim x_dim 2"],
    grid_spacing: float,
    cutoff: float,
    rolloff: float,
) -> Float[Array, "y_dim x_dim"]: ...


@overload
def _compute_lowpass_filter(
    frequency_grid: Float[Array, "z_dim y_dim x_dim 3"],
    grid_spacing: float,
    cutoff: float,
    rolloff: float,
) -> Float[Array, "z_dim y_dim x_dim"]: ...


def _compute_lowpass_filter(
    frequency_grid: Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"],
    grid_spacing: float = 1.0,
    cutoff: float = 0.667,
    rolloff: float = 0.05,
) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
    """
    Create a low-pass filter.

    Parameters
    ----------
    frequency_grid :
        The frequency coordinate system at which to evaulate
        the filter.
    grid_spacing :
        The grid spacing of ``frequency_grid``.
    cutoff :
        The cutoff frequency as a fraction of the Nyquist frequency,
        By default, ``0.667``.
    rolloff :
        The rolloff width as a fraction of the Nyquist frequency.
        By default, ``0.05``.

    Returns
    -------
    mask : `Array`, shape `shape`
        An array representing the low pass filter.
    """

    k_max = 1.0 / (2.0 * grid_spacing)
    k_cut = cutoff * k_max

    freqs_norm = jnp.linalg.norm(frequency_grid, axis=-1)

    frequencies_cut = freqs_norm > k_cut

    rolloff_width = rolloff * k_max
    mask = 0.5 * (
        1 + jnp.cos((freqs_norm - k_cut - rolloff_width) / rolloff_width * jnp.pi)
    )

    mask = jnp.where(frequencies_cut, 0.0, mask)
    mask = jnp.where(freqs_norm <= k_cut - rolloff_width, 1.0, mask)

    return mask


def _compute_whitening_filter(
    micrograph: Float[Array, "y_dim x_dim"],
    shape: Optional[tuple[int, int]] = None,
    interpolation_mode="nearest",
) -> Float[Array, "y_dim x_dim"]:
    """
    Compute a whitening filter from a micrograph. This is taken
    to be the inverse square root of the 2D radially averaged
    power spectrum.

    This implementation follows the cisTEM whitening filter
    algorithm.

    Parameters
    ----------
    micrograph :
        The micrograph in real space.
    shape :
        The shape at which to compute the filter. This downsamples or
        upsamples the filter by cropping or padding in real space.

    Returns
    -------
    filter :
        The whitening filter.
    """
    # Make coordinates
    micrograph_frequency_grid_in_angstroms = make_frequencies(micrograph.shape)
    # Transform to fourier space
    fourier_micrograph = rfftn(micrograph)
    # Compute norms
    radial_frequency_grid = jnp.linalg.norm(
        micrograph_frequency_grid_in_angstroms, axis=-1
    )
    # Compute power spectrum
    _, spectrum, _ = powerspectrum(
        fourier_micrograph,
        radial_frequency_grid,
        to_grid=True,
        interpolation_mode=interpolation_mode,
        k_max=jnp.sqrt(2) / 2.0,
    )  # type: ignore
    if shape is not None:
        spectrum = rfftn(
            resize_with_crop_or_pad(
                irfftn(spectrum, s=micrograph.shape), shape, pad_mode="edge"
            )
        ).real
    # Compute inverse square root
    filter = jax.lax.rsqrt(spectrum)
    # Divide filter by maximum, excluding zero mode
    maximum = jnp.max(jnp.delete(filter, jnp.asarray((0, 0), dtype=int)))
    filter /= maximum
    # Set zero mode manually to 1 (this diverges from the cisTEM
    # algorithm).
    filter = filter.at[0, 0].set(1.0)

    return filter
