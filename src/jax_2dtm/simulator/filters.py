"""
Filters to apply to images in Fourier space
"""

from __future__ import annotations

__all__ = [
    "compute_anti_aliasing_filter",
    "compute_whitening_filter",
    "Filter",
    "AntiAliasingFilter",
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

    def __post_init__(self, *args):
        object.__setattr__(self, "filter", self.compute(*args))

    @abstractmethod
    def compute(self, *args: tuple[Any, ...]) -> Array:
        """Compute the filter."""
        raise NotImplementedError

    def __call__(self, image: Array) -> Array:
        """Apply the filter to an image."""
        return self.filter * image


@dataclass
class AntiAliasingFilter(Filter):
    """
    Apply an anti-aliasing filter to an image.

    See documentation for
    ``jax_2dtm.simulator.compute_anti_aliasing_filter``
    for more information.

    Attributes
    ----------
    cutoff : `float`
    rolloff : `float`
    """

    cutoff: float = field(pytree_node=False, default=0.667)
    rolloff: float = field(pytree_node=False, default=0.05)

    def compute(self, freqs: Array) -> Array:
        return compute_anti_aliasing_filter(
            freqs,
            self.cutoff,
            self.rolloff,
        )


@dataclass
class WhiteningFilter(Filter):
    """
    Apply an whitening filter to an image.

    See documentation for
    ``jax_2dtm.simulator.compute_whitening_filter``
    for more information.
    """

    micrograph_freqs: InitVar[Array]
    micrograph: InitVar[Array]

    def compute(
        self, freqs: Array, micrograph_freqs: Array, micrograph: Array
    ) -> Array:
        return compute_whitening_filter(freqs, micrograph_freqs, micrograph)


def compute_anti_aliasing_filter(
    freqs: Array,
    cutoff: float = 0.667,
    rolloff: float = 0.05,
) -> Array:
    """
    Create an anti-aliasing filter.

    Parameters
    ----------
    freqs : `jax.Array`
        The fourier wavevectors in the imaging plane.
    cutoff : `float`, optional
        The cutoff frequency as a fraction of the Nyquist frequency,
        by default 0.667.
    rolloff : `float`, optional
        The rolloff width as a fraction of the Nyquist frequency,
        by default 0.05.

    Returns
    -------
    mask : `jax.Array`
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
    freqs: Array, micrograph_freqs: Array, micrograph: Array
) -> Array:
    """
    Compute a whitening filter from a micrograph. This is taken
    to be the inverse square root of the 2D radially averaged
    power spectrum.

    Parameters
    ----------
    micrograph : `jax.Array`, shape `(M1, M2)`
        The micrograph in fourier space.
    micrograph_freqs : `jax.Array`, shape `(M1, M2, 2)`
        The frequency range of the desired wavevectors.
        These should be in pixel units, not physical length.
    freqs : `jax.Array`, shape `(N1, N2, 2)`
        The frequency range of the desired wavevectors.
        These should be in pixel units, not physical length.

    Returns
    -------
    spectrum : `jax.Array`, shape `(N1, N2)`
        The power spectrum isotropically averaged onto ``freqs``.
    """
    spectrum, _ = powerspectrum(micrograph, micrograph_freqs, grid=freqs)

    return 1 / jnp.sqrt(spectrum)
