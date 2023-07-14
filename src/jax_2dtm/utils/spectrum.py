"""
Helper routines to compute power spectra.
"""

__all__ = ["powerspectrum"]

import jax.numpy as jnp

from .averaging import radial_average
from ..core import Array


def powerspectrum(
    image: Array,
    freqs: Array,
    pixel_size: float = 1.0,
    grid: bool = True,
):
    """
    Compute the power spectrum of an image averaged on a set
    of radial bins. This does not compute the zero mode of
    the spectrum.

    Arguments
    ---------
    image : `jax.Array`, shape `(M1, M2)`
        An image in Fourier space.
    freqs : `jax.Array`, shape `(N1, N2, 2)`
        The frequency range of the desired wavevectors.
        We must have that ``M1 >= N1`` and ``M2 >= N2``.
    pixel_size : float
        The pixel size of the frequency grid.
    grid : `bool`
        If ``True``, evalulate spectrum on the
        2D grid set by ``freqs``.

    Returns
    -------
    spectrum : `jax.Array`, shape `(max(N1, N2) // 2,)` or `(N1, N2)`
        Power spectrum up to the Nyquist frequency.
    k_bins : `jax.Array`, shape `(max(N1, N2) // 2,)`
        Radial bins for averaging. The minimum wavenumber
        and wavenumber spacing is computed as ``1 / max(N1, N2)``.
    """
    k_norm = jnp.linalg.norm(freqs, axis=-1)
    k_min = 1.0 / (pixel_size * max(*k_norm.shape))
    k_max = 1.0 / (2.0 * pixel_size)
    k_bins = jnp.arange(k_min, k_max, k_min)  # Left edges of bins
    power = (image * jnp.conjugate(image)).real
    spectrum = radial_average(power, k_norm, k_bins, grid=grid)

    return spectrum, k_bins
