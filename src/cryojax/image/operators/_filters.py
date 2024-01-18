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

from typing import Any, TypeVar
from typing_extensions import override

import jax
import jax.numpy as jnp

from ._operator import OperatorAsBuffer
from .._spectrum import powerspectrum
from .._fft import rfftn
from ..coordinates import make_frequencies
from ...core import field
from ...typing import Image, ImageCoords, RealImage

FilterT = TypeVar("FilterT", bound="Filter")
"""TypeVar for the Filter base class."""


class Filter(OperatorAsBuffer):
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
        self.operator = filter

    @property
    def filter(self) -> Image:
        return self.operator


class LowpassFilter(Filter):
    """
    Apply a low-pass filter to an image.

    See documentation for
    ``cryojax.simulator.compute_lowpass_filter``
    for more information.

    Attributes
    ----------
    cutoff :
        By default, ``0.95``, This cuts off
        modes above the Nyquist frequency.
    rolloff :
        By default, ``0.05``.
    """

    cutoff: float = field(static=True)
    rolloff: float = field(static=True)

    def __init__(
        self,
        freqs: ImageCoords,
        cutoff: float = 0.95,
        rolloff: float = 0.05,
    ) -> None:
        self.cutoff = cutoff
        self.rolloff = rolloff
        self.operator = compute_lowpass_filter(
            freqs, self.cutoff, self.rolloff
        )


class WhiteningFilter(Filter):
    """
    Apply a whitening filter to an image.
    """

    def __init__(
        self,
        frequency_grid: ImageCoords,
        micrograph: RealImage,
        *,
        grid_spacing: float = 1.0,
    ):
        self.operator = compute_whitening_filter(
            frequency_grid, micrograph, grid_spacing
        )


def compute_lowpass_filter(
    freqs: ImageCoords, cutoff: float = 0.667, rolloff: float = 0.05
) -> RealImage:
    """
    Create a low-pass filter.

    Parameters
    ----------
    freqs :
        The image coordinates.
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

    k_max = 1.0 / 2.0
    k_cut = cutoff * k_max

    freqs_norm = jnp.linalg.norm(freqs, axis=-1)

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
    freqs: ImageCoords,
    micrograph: RealImage,
    grid_spacing: float = 1.0,
) -> RealImage:
    """
    Compute a whitening filter from a micrograph. This is taken
    to be the inverse square root of the 2D radially averaged
    power spectrum.

    This implementation follows the cisTEM whitening filter
    algorithm.

    Parameters
    ----------
    freqs :
        The image coordinates.
    micrograph :
        The micrograph in real space.

    Returns
    -------
    filter :
        The whitening filter.
    """
    # Make coordinates
    micrograph_frequency_grid = make_frequencies(
        micrograph.shape, grid_spacing
    )
    # Transform to fourier space
    fourier_micrograph = rfftn(micrograph)
    # Compute norms
    radial_frequency_grid = jnp.linalg.norm(micrograph_frequency_grid, axis=-1)
    interpolating_radial_frequency_grid = jnp.linalg.norm(freqs, axis=-1)
    # Compute power spectrum
    spectrum, _ = powerspectrum(
        fourier_micrograph,
        radial_frequency_grid,
        k_max=jnp.sqrt(2.0) / 2.0,
        interpolating_radial_frequency_grid=interpolating_radial_frequency_grid,
    )
    # Compute inverse square root
    filter = jax.lax.rsqrt(spectrum)
    # Divide filter by maximum, excluding zero mode
    maximum = jnp.max(jnp.delete(filter, jnp.asarray((0, 0), dtype=int)))
    filter /= maximum
    # Set zero mode manually to 1 (this diverges from the cisTEM
    # algorithm).
    filter = filter.at[0, 0].set(1.0)

    return filter
