"""
Helper routines to compute power spectra.
"""

__all__ = ["powerspectrum"]

from typing import Optional, overload

import jax.numpy as jnp

from ._average import radial_average
from ..typing import (
    Real_,
    RealVector,
    RealImage,
    ComplexImage,
    RealVolume,
    ComplexVolume,
)


@overload
def powerspectrum(
    fourier_image: ComplexImage,
    radial_frequency_grid: RealImage,
    pixel_size: Real_ | float,
    *,
    k_min: Optional[Real_ | float],
    k_max: Optional[Real_ | float],
) -> tuple[RealVector, RealVector]:
    ...


@overload
def powerspectrum(
    fourier_image: ComplexVolume,
    radial_frequency_grid: RealVolume,
    pixel_size: Real_ | float,
    *,
    k_min: Optional[Real_ | float],
    k_max: Optional[Real_ | float],
) -> tuple[RealVector, RealVector]:
    ...


@overload
def powerspectrum(
    fourier_image: ComplexImage,
    radial_frequency_grid: RealImage,
    pixel_size: Real_ | float,
    *,
    to_grid: bool,
    interpolation_mode: str,
    k_min: Optional[Real_ | float],
    k_max: Optional[Real_ | float],
) -> tuple[RealVector, RealImage, RealVector]:
    ...


@overload
def powerspectrum(
    fourier_image: ComplexVolume,
    radial_frequency_grid: RealVolume,
    pixel_size: Real_ | float,
    *,
    to_grid: bool,
    interpolation_mode: str,
    k_min: Optional[Real_ | float],
    k_max: Optional[Real_ | float],
) -> tuple[RealVector, RealVolume, RealVector]:
    ...


def powerspectrum(
    fourier_image: ComplexImage | ComplexVolume,
    radial_frequency_grid: RealImage | RealVolume,
    pixel_size: Real_ | float = 1.0,
    *,
    to_grid: bool = False,
    interpolation_mode: str = "nearest",
    k_min: Optional[Real_ | float] = None,
    k_max: Optional[Real_ | float] = None,
) -> (
    tuple[RealVector, RealVector]
    | tuple[RealVector, RealImage | RealVolume, RealVector]
):
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
    to_grid :
        If ``False``, evalulate the spectrum as a 1D
        profile. Otherwise, evaluate the spectrum on the
        grid.
    interpolation_mode :
        If ``"linear"``, evaluate on the grid using linear
        interpolation. If ``False``,
    k_min :
        Minimum wavenumber bin. By default, ``0.0``.
    k_max :
        Maximum wavenumber bin. By default, ``1 / (2 * pixel_size)``.

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
    k_max = jnp.sqrt(2) / (pixel_size * 2.0) if k_max is None else k_max
    k_step = 1.0 / (pixel_size * max(*power.shape))
    bins = jnp.arange(k_min, k_max + k_step, k_step)  # Left edges of bins
    # Compute radially averaged power spectrum as a 1D profile or
    # interpolated onto a 2D grid
    if to_grid:
        spectrum_as_profile, spectrum_as_image = radial_average(
            power,
            radial_frequency_grid,
            bins,
            to_grid=True,
            interpolation_mode=interpolation_mode,
        )
        return spectrum_as_profile, spectrum_as_image, bins
    else:
        spectrum_as_profile = radial_average(
            power, radial_frequency_grid, bins
        )
        return spectrum_as_profile, bins
