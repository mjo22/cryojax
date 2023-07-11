"""
Filters to apply to images in Fourier space
"""

from __future__ import annotations

__all__ = [
    "compute_anti_aliasing_filter",
    "Filter",
    "AntiAliasingFilter",
    "WhiteningFilter",
]

from abc import ABCMeta, abstractmethod

import jax.numpy as jnp

from ..types import dataclass, field, Array
from .scattering import ImageConfig


@dataclass
class Filter(metaclass=ABCMeta):
    """
    Base class for computing and applying an image filter.

    Attributes
    ----------
    config : `jax_2dtm.simulator.ImageConfig`
        The image configuration.
    freqs : `jax.Array`
        The fourier wavevectors in the imaging plane.
    """

    config: ImageConfig = field(pytree_node=False)
    freqs: Array = field(pytree_node=False)
    filter: Array = field(pytree_node=False, init=False)

    def __post_init__(self):
        object.__setattr__(self, "filter", self.compute())

    @abstractmethod
    def compute(self) -> Array:
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

    cutoff: float = field(pytree_node=False, default=1.00)
    rolloff: float = field(pytree_node=False, default=0.05)

    def compute(self) -> Array:
        return compute_anti_aliasing_filter(
            self.freqs,
            self.config.pixel_size,
            self.cutoff,
            self.rolloff,
        )


@dataclass
class WhiteningFilter(Filter):
    """
    Apply an whitening filter to an image.

    See documentation for
    ``jax_2dtm.simulator.compute_anti_aliasing_filter``
    for more information.

    Attributes
    ----------

    """

    def compute(self) -> Array:
        raise NotImplementedError


def compute_anti_aliasing_filter(
    freqs: Array,
    pixel_size: float,
    cutoff: float = 0.667,
    rolloff: float = 0.05,
) -> Array:
    """
    Create an anti-aliasing filter.

    Parameters
    ----------
    freqs : `jax.Array`
        The fourier wavevectors in the imaging plane.
    pixel_size : `float`
        The pixel size of the image.
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

    k_max = 1 / (2 * pixel_size)
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
