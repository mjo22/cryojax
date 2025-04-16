"""
Helper routines to compute power spectra.
"""

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Bool



import jax.numpy as jnp
import math
import equinox as eqx
from jax import Array
from jaxtyping import Float, Complex
from typing import Optional


import cryojax.coordinates as cc
from ._average import compute_binned_radial_average  # keep import inside jit-safe scope

@eqx.filter_jit
def compute_radially_averaged_powerspectrum(
    fourier_image: Complex[Array, "y x"] | Complex[Array, "z y x"],
    pixel_size: float = 1.0,
) -> tuple[Float[Array, "n_bins"], Float[Array, "n_bins"]]:

    radial_frequency_grid = cc.make_radial_frequency_grid(fourier_image.shape,grid_spacing=pixel_size)
    start = 0 
    stop = math.sqrt(2) / (pixel_size*2.0) 

    # hard code constant to avoid problems in linspace shape size.
    # jnp.sqrt(2) will get traced sadly.

    frequency_bins = jnp.linspace(start, stop, int(fourier_image.shape[0]*math.sqrt(2)/2)+1)
    squared_fourier_amplitudes = jnp.real(fourier_image * jnp.conjugate(fourier_image))

    spectrum = compute_binned_radial_average(squared_fourier_amplitudes, radial_frequency_grid, frequency_bins)
    #return_spec = jnp.where(spectrum, )

    return spectrum, frequency_bins

