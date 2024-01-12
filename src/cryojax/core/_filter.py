"""
Filters to apply to images in Fourier space
"""

from __future__ import annotations

__all__ = [
    "Filter",
    "FilterType",
    "LowpassFilter",
    "WhiteningFilter",
]

from abc import abstractmethod
from typing import Any, TypeVar
from typing_extensions import override

import jax
import jax.numpy as jnp
from equinox import Module

from ..image import powerspectrum, rfftn, make_frequencies
from ._field import field
from ..typing import Image, ImageCoords, RealImage

FilterType = TypeVar("FilterType", bound="Filter")
"""TypeVar for the Filter base class."""


class Filter(Module):
    """
    Base class for computing and applying an image filter.

    Attributes
    ----------
    filter :
        The filter. Note that this is automatically
        computed upon instantiation.
    """

    filter: Image = field()

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Compute the filter."""
        super().__init__(**kwargs)

    def __call__(self, image: Image) -> Image:
        """Apply the filter to an image."""
        return self.filter * image

    def __mul__(self: FilterType, other: FilterType) -> _ProductFilter:
        return _ProductFilter(filter1=self, filter2=other)

    def __rmul__(self: FilterType, other: FilterType) -> _ProductFilter:
        return _ProductFilter(filter1=other, filter2=self)


class _ProductFilter(Filter):
    """A helper to represent the product of two filters."""

    filter1: FilterType  # type: ignore
    filter2: FilterType  # type: ignore

    @override
    def __init__(
        self, filter1: FilterType, filter2: FilterType, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter = filter1.filter * filter2.filter

    def __repr__(self):
        return f"{repr(self.filter1)} * {repr(self.filter2)}"


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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.rolloff = rolloff
        self.filter = _compute_lowpass_filter(freqs, self.cutoff, self.rolloff)


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
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.filter = _compute_whitening_filter(
            frequency_grid, micrograph, grid_spacing
        )


def _compute_lowpass_filter(
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


def _compute_whitening_filter(
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
