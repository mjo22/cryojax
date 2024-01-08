"""
Filters to apply to images in Fourier space
"""

from __future__ import annotations

__all__ = [
    "compute_lowpass_filter",
    "compute_whitening_filter",
    "Filter",
    "FilterType",
    "LowpassFilter",
    "WhiteningFilter",
]

from abc import abstractmethod
from typing import Any, TypeVar, overload
from typing_extensions import override

import jax
import jax.numpy as jnp

from .manager import ImageManager

from ..utils import powerspectrum, make_frequencies
from ..core import field, BufferModule
from ..typing import Image, ImageCoords, RealImage, ComplexImage

FilterType = TypeVar("FilterType", bound="Filter")
"""TypeVar for the Filter base class."""


class Filter(BufferModule):
    """
    Base class for computing and applying an image filter.

    Attributes
    ----------
    filter :
        The filter. Note that this is automatically
        computed upon instantiation.
    """

    frequency_grid: InitVar[ImageCoords | None] = None
    filter: Image = field(init=False)

    def __post_init__(self, frequency_grid: ImageCoords | None, **kwargs: Any):
        self.filter = self.evaluate(frequency_grid, **kwargs)

    @overload
    @abstractmethod
    def evaluate(self, freqs: None, **kwargs: Any) -> Image:
        ...

    @overload
    @abstractmethod
    def evaluate(self, freqs: ImageCoords, **kwargs: Any) -> Image:
        ...

    @abstractmethod
    def evaluate(self, freqs: ImageCoords | None = None, **kwargs: Any) -> Image:
        """Compute the filter."""
        raise NotImplementedError

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
    def evaluate(self, freqs: ImageCoords | None = None, **kwargs: Any) -> Image:
        return self.filter1.filter * self.filter2.filter

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

    cutoff: float = field(static=True, default=0.95)
    rolloff: float = field(static=True, default=0.05)

    @override
    def evaluate(self, freqs: ImageCoords | None, **kwargs: Any) -> RealImage:
        if freqs is None:
            raise ValueError(
                "The frequency grid must be given as an argument to the Filter."
            )
        else:
            return compute_lowpass_filter(
                freqs,
                self.cutoff,
                self.rolloff,
                **kwargs,
            )


class WhiteningFilter(Filter):
    """
    Apply an whitening filter to an image.

    See documentation for
    ``cryojax.simulator.compute_whitening_filter``
    for more information.
    """

    manager: ImageManager = field()

    micrograph: ComplexImage = field()

    @override
    def evaluate(self, freqs: ImageCoords | None, **kwargs: Any) -> RealImage:
        if freqs is None:
            raise ValueError(
                "The frequency grid must be given as an argument to the Filter."
            )
        else:
            return compute_whitening_filter(
                freqs, self.micrograph, **kwargs
            )


def compute_lowpass_filter(
    freqs: ImageCoords,
    cutoff: float = 0.667,
    rolloff: float = 0.05,
    **kwargs: Any,
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
    kwargs :
        Keyword arguments passed to ``cryojax.utils.make_coordinates``.

    Returns
    -------
    mask : `Array`, shape `shape`
        An array representing the anti-aliasing filter.
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
    freqs: ImageCoords, micrograph: ComplexImage, **kwargs: Any
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
        The micrograph in fourier space.

    Returns
    -------
    filter :
        The whitening filter.
    """
    micrograph = jnp.asarray(micrograph)
    # Make coordinates
    micrograph_frequency_grid = make_frequencies(micrograph.shape, *kwargs)
    # Compute norms
    radial_frequency_grid = jnp.linalg.norm(micrograph_frequency_grid, axis=-1)
    interpolating_radial_frequency_grid = jnp.linalg.norm(freqs, axis=-1)
    # Compute power spectrum
    spectrum, _ = powerspectrum(
        micrograph,
        radial_frequency_grid,
        k_max=jnp.sqrt(2.0) / 2.0,
        interpolating_radial_frequency_grid=interpolating_radial_frequency_grid,
    )
    # Compute inverse square root
    filter = jax.lax.rsqrt(spectrum)
    # Divide filter by maximum, excluding zero mode
    filter /= jnp.max(filter[1:, 1:])
    # Set zero mode manually to 1 (this diverges from the cisTEM
    # algorithm).
    filter = filter.at[0, 0].set(1.0)

    return filter
