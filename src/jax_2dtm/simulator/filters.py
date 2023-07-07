"""
Filters to apply to images in Fourier space
"""

from __future__ import annotations

__all__ = ["anti_aliasing_filter", "Filter", "AntiAliasingFilter"]

import jax.numpy as jnp
import dataclasses
from abc import ABCMeta, abstractmethod
from ..types import dataclass, field, Array
from .image import ImageConfig


@dataclasses.dataclass
class Filter(metaclass=ABCMeta):
    """
    Base class for computing and applying an image filter.

    Attributes
    ----------
    config : ImageConfig
        The image configuration.
    freqs : jax.Array
        The fourier wavevectors in the imaging plane.
    """

    config: ImageConfig
    freqs: Array

    def __post_init__(self):
        self.filter = self.compute_filter()

    @abstractmethod
    def compute_filter(self) -> Array:
        return NotImplementedError

    def __call__(self, scattering_image: Array) -> Array:
        return self.filter * scattering_image


@dataclasses.dataclass
class AntiAliasingFilter(Filter):
    """
    Apply an anti-aliasing filter to an image.

    Attributes
    ----------
    cutoff : float
        See documentation for ``jax_2dtm.simulator.anti_aliasing_filter``.
    rolloff : float
        See documentation for ``jax_2dtm.simulator.anti_aliasing_filter``.
    """

    cutoff: float = 1.00
    rolloff: float = 0.05

    def compute_filter(self) -> Array:
        return anti_aliasing_filter(
            self.freqs,
            self.config.pixel_size,
            self.cutoff,
            self.rolloff,
        )


def anti_aliasing_filter(
    freqs: Array,
    pixel_size: float,
    cutoff: float = 0.667,
    rolloff: float = 0.05,
) -> Array:
    """
    Create an anti-aliasing filter.

    Parameters
    ----------
    freqs : jax.Array
        The fourier wavevectors in the imaging plane.
    pixel_size : float
        The pixel size of the image.
    cutoff : float, optional
        The cutoff frequency as a fraction of the Nyquist frequency,
        by default 0.667.
    rolloff : float, optional
        The rolloff width as a fraction of the Nyquist frequency,
        by default 0.05.

    Returns
    -------
    mask : jax.Array
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
