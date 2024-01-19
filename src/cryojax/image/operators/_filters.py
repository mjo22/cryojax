"""
Filters to apply to images in Fourier space
"""

from __future__ import annotations

__all__ = [
    "Filter",
    "FilterT",
    "LowpassFilter",
    "WhiteningFilter",
    "compute_lowpass_filter",
    "compute_whitening_filter",
]

from typing import TypeVar, Optional

import jax
import jax.numpy as jnp

from .._edges import crop_or_pad
from ._operator import ImageMultiplier
from .._spectrum import powerspectrum
from .._fft import rfftn, irfftn
from ..coordinates import make_frequencies
from ...core import field
from ...typing import Image, ComplexImage, ImageCoords, RealImage

FilterT = TypeVar("FilterT", bound="Filter")
"""TypeVar for the Filter base class."""


class Filter(ImageMultiplier):
    """
    Base class for computing and applying an image filter.

    Attributes
    ----------
    filter :
        The filter. Note that this is automatically
        computed upon instantiation.
    """

    def __init__(self, filter: Image):
        """Compute the filter."""
        self.buffer = filter

    def __call__(self, image: ComplexImage) -> ComplexImage:
        return image * jax.lax.stop_gradient(self.buffer)


class LowpassFilter(Filter):
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

    cutoff: float = field(static=True)
    rolloff: float = field(static=True)

    def __init__(
        self,
        frequency_grid: ImageCoords,
        grid_spacing: float = 1.0,
        cutoff: float = 0.95,
        rolloff: float = 0.05,
    ) -> None:
        self.cutoff = cutoff
        self.rolloff = rolloff
        self.buffer = compute_lowpass_filter(
            frequency_grid, grid_spacing, self.cutoff, self.rolloff
        )


class WhiteningFilter(Filter):
    """
    Apply a whitening filter to an image.
    """

    def __init__(
        self,
        micrograph: RealImage,
        shape: Optional[tuple[int, int]] = None,
    ):
        self.buffer = compute_whitening_filter(micrograph, shape)


def compute_lowpass_filter(
    frequency_grid: ImageCoords,
    grid_spacing: float = 1.0,
    cutoff: float = 0.667,
    rolloff: float = 0.05,
) -> RealImage:
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
        1
        + jnp.cos(
            (freqs_norm - k_cut - rolloff_width) / rolloff_width * jnp.pi
        )
    )

    mask = jnp.where(frequencies_cut, 0.0, mask)
    mask = jnp.where(freqs_norm <= k_cut - rolloff_width, 1.0, mask)

    return mask


def compute_whitening_filter(
    micrograph: RealImage,
    shape: Optional[tuple[int, int]] = None,
) -> RealImage:
    """
    Compute a whitening filter from a micrograph. This is taken
    to be the inverse square root of the 2D radially averaged
    power spectrum.

    This implementation follows the cisTEM whitening filter
    algorithm.

    Parameters
    ----------
    frequency_grid_in_angstroms :
        The frequency coordinate system at which to evaulate
        the filter.
    micrograph :
        The micrograph in real space.
    grid_spacing :
        The grid spacing of ``frequency_grid_in_angstroms``.

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
    spectrum, _ = powerspectrum(
        fourier_micrograph,
        radial_frequency_grid,
        to_grid=True,
        interpolation_mode="nearest",
        k_max=jnp.sqrt(2.0) / 2.0,
    )
    if shape is not None:
        spectrum = rfftn(
            crop_or_pad(
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
