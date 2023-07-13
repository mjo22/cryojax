"""
Helper routines to compute power spectra.
"""

__all__ = ["powerspectrum"]

import jax.numpy as jnp

from .averaging import radial_average
from ..types import Array


def powerspectrum(
    image: Array,
    k_norm: Array,
    k_bins: Array,
    grid: bool = False,
):
    """
    Compute the power spectrum of an image averaged on a set
    of radial bins.

    Arguments
    ---------
    image : `jax.Array`, shape `(M1, M2)`
        An image in Fourier space.
    k_norm : `jax.Array`, shape `(N1, N2)`
        The radial coordinate system in the desired frequency range.
    k_bins : `jax.Array`, shape `(M,)`
        Radial bins for averaging. These
        must be evenly spaced.
    grid : `bool`
        If ``True``, evalulate spectrum on the
        2D grid set by ``k_norm``.

    Returns
    -------
    spectrum : `jax.Array`, shape `(M,)` or `(N1, N2)`
        Power spectrum
    """
    power = (image * jnp.conjugate(image)).real
    spectrum = radial_average(power, k_norm, k_bins, grid=grid)

    return spectrum
