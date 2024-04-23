"""
Helper routines to compute power spectra.
"""

from typing import Optional, overload

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ._average import radial_average


@overload
def powerspectrum(
    fourier_image: Complex[Array, "y_dim x_dim"],
    radial_frequency_grid: Float[Array, "y_dim x_dim"],
    pixel_size: Float[Array, ""] | float,
    *,
    k_min: Optional[Float[Array, ""] | float] = None,
    k_max: Optional[Float[Array, ""] | float] = None,
) -> tuple[Float[Array, " n_bins"], Float[Array, " n_bins"]]: ...


@overload
def powerspectrum(
    fourier_image: Complex[Array, "z_dim y_dim x_dim"],
    radial_frequency_grid: Float[Array, "z_dim y_dim x_dim"],
    pixel_size: Float[Array, ""] | float,
    *,
    k_min: Optional[Float[Array, ""] | float] = None,
    k_max: Optional[Float[Array, ""] | float] = None,
) -> tuple[Float[Array, " n_bins"], Float[Array, " n_bins"]]: ...


@overload
def powerspectrum(
    fourier_image: Complex[Array, "y_dim x_dim"],
    radial_frequency_grid: Float[Array, "y_dim x_dim"],
    pixel_size: Float[Array, ""] | float,
    *,
    to_grid: bool = False,
    interpolation_mode: str = "nearest",
    k_min: Optional[Float[Array, ""] | float] = None,
    k_max: Optional[Float[Array, ""] | float] = None,
) -> (
    tuple[Float[Array, " n_bins"], Float[Array, " n_bins"]]
    | tuple[Float[Array, " n_bins"], Float[Array, "y_dim x_dim"], Float[Array, " n_bins"]]
): ...


@overload
def powerspectrum(
    fourier_image: Complex[Array, "z_dim y_dim x_dim"],
    radial_frequency_grid: Float[Array, "z_dim y_dim x_dim"],
    pixel_size: Float[Array, ""] | float,
    *,
    to_grid: bool = False,
    interpolation_mode: str = "nearest",
    k_min: Optional[Float[Array, ""] | float] = None,
    k_max: Optional[Float[Array, ""] | float] = None,
) -> (
    tuple[Float[Array, " n_bins"], Float[Array, " n_bins"]]
    | tuple[
        Float[Array, " n_bins"],
        Float[Array, "z_dim y_dim x_dim"],
        Float[Array, " n_bins"],
    ]
): ...


def powerspectrum(
    fourier_image: Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"],
    radial_frequency_grid: (
        Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ),
    pixel_size: Float[Array, ""] | float = 1.0,
    *,
    to_grid: bool = False,
    interpolation_mode: str = "nearest",
    k_min: Optional[Float[Array, ""] | float] = None,
    k_max: Optional[Float[Array, ""] | float] = None,
) -> (
    tuple[Float[Array, " n_bins"], Float[Array, " n_bins"]]
    | tuple[
        Float[Array, " n_bins"],
        Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"],
        Float[Array, " n_bins"],
    ]
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
        spectrum_as_profile = radial_average(power, radial_frequency_grid, bins)
        return spectrum_as_profile, bins
