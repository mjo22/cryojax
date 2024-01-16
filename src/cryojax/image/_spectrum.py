"""
Helper routines to compute power spectra.
"""

__all__ = ["powerspectrum"]

from typing import Optional, Union, overload

import jax.numpy as jnp

from ._average import radial_average
from ..typing import Real_, RealVector, RealImage, ComplexImage


@overload
def powerspectrum(
    fourier_image: ComplexImage,
    radial_frequency_grid: RealImage,
    pixel_size: Real_ | float,
    *,
    k_min: Optional[Real_ | float],
    k_max: Optional[Real_ | float],
    interpolating_radial_frequency_grid: None,
) -> tuple[RealVector, RealVector]:
    ...


@overload
def powerspectrum(
    fourier_image: ComplexImage,
    radial_frequency_grid: RealImage,
    pixel_size: Real_ | float,
    *,
    k_min: Optional[Real_ | float],
    k_max: Optional[Real_ | float],
    interpolating_radial_frequency_grid: RealImage,
) -> tuple[RealImage, RealVector]:
    ...


def powerspectrum(
    fourier_image: ComplexImage,
    radial_frequency_grid: RealImage,
    pixel_size: Real_ | float = 1.0,
    *,
    k_min: Optional[Real_ | float] = None,
    k_max: Optional[Real_ | float] = None,
    interpolating_radial_frequency_grid: Optional[RealImage] = None,
) -> tuple[RealVector | RealImage, RealVector]:
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
    pixel_size :
        The pixel size of the radial frequency grid.
    k_min :
        Minimum wavenumber bin. By default, ``0.0``.
    k_max :
        Maximum wavenumber bin. By default, ``1 / (2 * pixel_size)``.
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
    # Compute bins
    k_min = 0.0 if k_min is None else k_min
    k_max = 1.0 / (pixel_size * 2.0) if k_max is None else k_max
    k_step = 1.0 / (pixel_size * max(*power.shape))
    bins = jnp.arange(k_min, k_max, k_step)  # Left edges of bins
    # Compute radially averaged power spectrum as a 1D profile or
    # interpolated onto a 2D grid
    spectrum = radial_average(
        power, radial_frequency_grid, bins, interpolating_radial_frequency_grid
    )

    return spectrum, bins
