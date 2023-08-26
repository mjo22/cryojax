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
from dataclasses import InitVar
from typing import Any

import jax.numpy as jnp

from ..utils import powerspectrum
from ..core import dataclass, field, Array, ArrayLike


@dataclass
class Filter(metaclass=ABCMeta):
    """
    Base class for computing and applying an image filter.

    Attributes
    ----------
    freqs : `jax.Array`
        The fourier wavevectors in the imaging plane.
    """

    filter: Array = field(pytree_node=False, init=False)

    freqs: InitVar[ArrayLike]

    def __post_init__(self, *args: Any):
        object.__setattr__(self, "filter", self.compute(*args))

    @abstractmethod
    def compute(self, *args: tuple[Any, ...]) -> Array:
        """Compute the filter."""
        raise NotImplementedError

    def __call__(self, image: ArrayLike) -> Array:
        """Apply the filter to an image."""
        return self.filter * image


@dataclass
class LowpassFilter(Filter):
    """
    Apply a low-pass filter to an image.

    See documentation for
    ``cryojax.simulator.compute_lowpass_filter``
    for more information.

    Attributes
    ----------
    cutoff : `float`
        By default, this is set ``1.0``, which cuts off
        modes above the Nyquist frequency.
    rolloff : `float`
    """

    cutoff: float = field(pytree_node=False, default=1.0)
    rolloff: float = field(pytree_node=False, default=0.05)

    def compute(self, freqs: ArrayLike) -> Array:
        return compute_lowpass_filter(
            freqs,
            self.cutoff,
            self.rolloff,
        )


@dataclass
class WhiteningFilter(Filter):
    """
    Apply an whitening filter to an image.

    See documentation for
    ``cryojax.simulator.compute_whitening_filter``
    for more information.
    """

    micrograph_freqs: InitVar[ArrayLike]
    micrograph: InitVar[ArrayLike]

    def compute(
        self,
        freqs: ArrayLike,
        micrograph_freqs: ArrayLike,
        micrograph: ArrayLike,
    ) -> Array:
        return compute_whitening_filter(freqs, micrograph_freqs, micrograph)


def compute_lowpass_filter(
    freqs: ArrayLike,
    cutoff: float = 0.667,
    rolloff: float = 0.05,
) -> Array:
    """
    Create an anti-aliasing filter.

    Parameters
    ----------
    freqs : `ArrayLike`, shape `(N1, N2, 2)`
        The fourier wavevectors in the imaging plane.
    cutoff : `float`, optional
        The cutoff frequency as a fraction of the Nyquist frequency,
        by default 0.667.
    rolloff : `float`, optional
        The rolloff width as a fraction of the Nyquist frequency,
        by default 0.05.

    Returns
    -------
    mask : `Array`, shape `(N1, N2)`
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
    freqs: ArrayLike, micrograph_freqs: ArrayLike, micrograph: ArrayLike
) -> Array:
    """
    Compute a whitening filter from a micrograph. This is taken
    to be the inverse square root of the 2D radially averaged
    power spectrum.

    Parameters
    ----------
    micrograph : `ArrayLike`, shape `(M1, M2)`
        The micrograph in fourier space.
    micrograph_freqs : `ArrayLike`, shape `(M1, M2, 2)`
        The frequency range of the desired wavevectors.
        These should be in pixel units, not physical length.
    freqs : `ArrayLike`, shape `(N1, N2, 2)`
        The frequency range of the desired wavevectors.
        These should be in pixel units, not physical length.

    Returns
    -------
    spectrum : `jax.Array`, shape `(N1, N2)`
        The power spectrum isotropically averaged onto ``freqs``.
    """
    micrograph = jnp.asarray(micrograph)
    M1, M2 = micrograph.shape
    micrograph /= jnp.sqrt(M1 * M2)
    spectrum, _ = powerspectrum(micrograph, micrograph_freqs, grid=freqs)

    return 1 / jnp.sqrt(spectrum)