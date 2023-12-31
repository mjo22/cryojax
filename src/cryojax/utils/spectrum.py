"""
Helper routines to compute power spectra.
"""

__all__ = ["powerspectrum"]

from typing import Optional, Union

import jax.numpy as jnp

from .average import radial_average
from ..typing import RealVector, RealImage, ComplexImage


def powerspectrum(
    fourier_image: ComplexImage,
    radial_frequency_grid: RealImage,
    interpolating_radial_frequency_grid: Optional[RealImage] = None,
) -> tuple[Union[RealImage, RealVector], RealVector]:
    """
    Compute the power spectrum of an image averaged on a set
    of radial bins. This does not compute the zero mode of
    the spectrum.

    Parameters
    ----------
    fourier_image :
        An image in Fourier space.
    radial_frequency_grid :
        The frequency range of the desired wavevectors.
    interpolating_radial_frequency_grid :
        If ``None``, evalulate the spectrum as a 1D
        profile. Otherwise, evaluate the spectrum on this
        2D grid of frequencies.
    Returns
    -------
    spectrum :
        Power spectrum up to the Nyquist frequency.
    bins :
        Radial wavenumber bins for averaging.
    """
    # Compute power
    power = (fourier_image * jnp.conjugate(fourier_image)).real
    k_min = 1 / max(*power.shape)
    k_max = 1.0 / 2.0
    k_step = k_min
    bins = jnp.arange(k_min, k_max, k_step)  # Left edges of bins
    spectrum = radial_average(
        power, radial_frequency_grid, bins, interpolating_radial_frequency_grid
    )

    return spectrum, bins
