"""
Filters to apply to images in Fourier space
"""

__all__ = ["anti_aliasing_filter"]


import jax.numpy as jnp
from typing import Optional
from ..types import Array
from .image import ImageConfig


def anti_aliasing_filter(
    config: ImageConfig,
    freqs: Array,
    cutoff: float = 0.667,
    rolloff: float = 0.05,
) -> Array:
    """
    Create an anti-aliasing filter.

    Parameters
    ----------
    image : ImageConfig
        The configuration of the image to create the filter for.
    freqs : jax.Array
    cutoff : float, optional
        The cutoff frequency as a fraction of the Nyquist frequency,
        by default 0.667.
    rolloff : float, optional
        The rolloff width as a fraction of the Nyquist frequency, by default 0.05

    Returns
    -------
    mask : jax.Array
        An array representing the anti-aliasing filter.
    """

    k_max = 1 / (2 * config.pixel_size)
    k_cut = cutoff * k_max

    freqs_norm = freqs.norm(dim=-1)

    frequencies_cut = freqs_norm > k_cut

    rolloff_width = rolloff * k_max
    mask = 0.5 * (
        1
        + jnp.cos(
            (freqs_norm - k_cut - rolloff_width) / rolloff_width * jnp.pi
        )
    )
    mask[frequencies_cut] = 0
    mask[freqs_norm <= k_cut - rolloff_width] = 1

    return mask
