"""
Helper routines to compute power spectra.
"""

__all__ = ["powerspectrum"]

from typing import Optional

import jax.numpy as jnp

from .averaging import radial_average
from ..core import Array


def powerspectrum(
    image: Array,
    freqs: Array,
    pixel_size: float = 1.0,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    grid: Optional[Array] = None,
):
    """
    Compute the power spectrum of an image averaged on a set
    of radial bins. This does not compute the zero mode of
    the spectrum.

    Arguments
    ---------
    image : `jax.Array`, shape `(N1, N2)`
        An image in Fourier space.
    freqs : `jax.Array`, shape `(N1, N2, 2)`
        The frequency range of the desired wavevectors.
    pixel_size : float
        The pixel size of the frequency grid.
    grid : `jax.Array`, shape `(M1, M2, 2)`, optional
        If ``None``, evalulate the spectrum as a 1D
        profile. Otherwise, evaluate the spectrum on this
        2D grid of frequencies.
    Returns
    -------
    spectrum : `jax.Array`, shape `(max(L1, L2) // 2,)` or `(L1, L2)`
        Power spectrum up to the Nyquist frequency. ``(L1, L2)`` can
        be ``(M1, M2)`` or ``(N1, N2)``.
    k_bins : `jax.Array`, shape `(max(L1, L2) // 2,)`
        Radial bins for averaging. The minimum wavenumber
        and wavenumber spacing is computed as ``1 / max(L1, L2)``.
        ``(L1, L2)`` can be ``(M1, M2)`` or ``(N1, N2)``.
    """
    power = (image * jnp.conjugate(image)).real
    k_norm = jnp.linalg.norm(freqs, axis=-1)
    k_max = 1.0 / (2.0 * pixel_size)
    if grid is None:
        q_norm = None
        k_min = 1.0 / (pixel_size * max(*k_norm.shape))
    else:
        q_norm = jnp.linalg.norm(grid, axis=-1)
        k_min = 1.0 / (pixel_size * max(*q_norm.shape))
    k_bins = jnp.arange(k_min, k_max, k_min)  # Left edges of bins
    spectrum = radial_average(power, k_norm, k_bins, grid=q_norm)

    return spectrum, k_bins
