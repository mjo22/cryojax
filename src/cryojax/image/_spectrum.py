"""
Helper routines to compute power spectra.
"""

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ._average import compute_binned_radial_average


def compute_radially_averaged_powerspectrum(
    fourier_image: Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"],
    radial_frequency_grid: (
        Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ),
    pixel_size: Float[Array, ""] | float = 1.0,
    *,
    minimum_frequency: Optional[Float[Array, ""] | float] = None,
    maximum_frequency: Optional[Float[Array, ""] | float] = None,
) -> tuple[Float[Array, " n_bins"], Float[Array, " n_bins"]]:
    """Compute the power spectrum of an image averaged on a set
    of radial bins.

    **Arguments:**

    `fourier_image`:
        An image in Fourier space.
    `radial_frequency_grid`:
        The radial frequency coordinate system of `fourier_image`.
    `pixel_size`:
        The pixel size of `radial_frequency_grid`.
    `minimum_frequency`:
        Minimum frequency bin. By default, `0.0`.
    `minimum_frequency`:
        Maximum frequency bin. By default, `1 / (2 * pixel_size)`.

    **Returns:**

    A tuple of the radially averaged power spectrum and the frequency bins
    over which it is computed.
    """
    # Compute squared amplitudes
    squared_fourier_amplitudes = (fourier_image * jnp.conjugate(fourier_image)).real
    # Compute bins
    q_min = 0.0 if minimum_frequency is None else minimum_frequency
    q_max = (
        jnp.sqrt(2) / (pixel_size * 2.0)
        if maximum_frequency is None
        else maximum_frequency
    )
    q_step = 1.0 / (pixel_size * max(*squared_fourier_amplitudes.shape))
    frequency_bins = jnp.linspace(q_min, q_max, 1 + int((q_max - q_min) / q_step))
    # Compute radially averaged power spectrum as a 1D profile
    radially_averaged_powerspectrum = compute_binned_radial_average(
        squared_fourier_amplitudes, radial_frequency_grid, frequency_bins
    )

    return radially_averaged_powerspectrum, frequency_bins
