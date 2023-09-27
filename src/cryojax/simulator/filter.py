"""
Filters to apply to images in Fourier space
"""

from __future__ import annotations

__all__ = [
    "compute_lowpass_filter",
    "compute_whitening_filter",
    "Filter",
    "LowpassFilter",
    "WhiteningFilter",
]

from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import jax.numpy as jnp

from ..utils import powerspectrum, make_frequencies
from ..core import field, Module, Image, RealImage, ComplexImage


class Filter(Module, metaclass=ABCMeta):
    """
    Base class for computing and applying an image filter.

    Attributes
    ----------
    shape :
        The image shape.
    filter :
        The filter. Note that this is automatically
        computed upon instantiation.
    """

    shape: tuple[int, int] = field(static=True)
    filter: Image = field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any):
        self.filter = self.compute(*args, **kwargs)

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> Image:
        """Compute the filter."""
        raise NotImplementedError

    def __call__(self, image: ComplexImage) -> ComplexImage:
        """Apply the filter to an image."""
        return self.filter * image


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

    def compute(self, **kwargs) -> RealImage:
        return compute_lowpass_filter(
            self.shape, self.cutoff, self.rolloff, **kwargs
        )


class WhiteningFilter(Filter):
    """
    Apply an whitening filter to an image.

    See documentation for
    ``cryojax.simulator.compute_whitening_filter``
    for more information.
    """

    micrograph: ComplexImage = field(static=True)

    def compute(self, **kwargs: Any) -> RealImage:
        return compute_whitening_filter(self.shape, self.micrograph, **kwargs)


def compute_lowpass_filter(
    shape: tuple[int, int],
    cutoff: float = 0.667,
    rolloff: float = 0.05,
    **kwargs: Any,
) -> RealImage:
    """
    Create a low-pass filter.

    Parameters
    ----------
    shape :
        The shape of the filter. This is used to compute the image
        coordinates.
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
    freqs = make_frequencies(shape, **kwargs)

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
    shape: tuple[int, int], micrograph: ComplexImage, **kwargs: Any
) -> RealImage:
    """
    Compute a whitening filter from a micrograph. This is taken
    to be the inverse square root of the 2D radially averaged
    power spectrum.

    Parameters
    ----------
    shape :
        The shape of the filter. This is used to compute the image
        coordinates.
    micrograph :
        The micrograph in fourier space.

    Returns
    -------
    spectrum :
        The power spectrum isotropically averaged onto a coordinate
        system whose shape is set by ``shape``.
    """
    micrograph = jnp.asarray(micrograph)
    # Make coordinates
    freqs = make_frequencies(shape, **kwargs)
    micrograph_freqs = make_frequencies(micrograph.shape, *kwargs)
    # Compute power spectrum
    micrograph /= jnp.sqrt(np.prod(micrograph.shape))
    spectrum, _ = powerspectrum(micrograph, micrograph_freqs, grid=freqs)

    return 1 / jnp.sqrt(spectrum)
