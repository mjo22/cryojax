"""
Helper routines to compute power spectra.
"""

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Bool



import jax.numpy as jnp
import equinox as eqx
from jax import Array
from jaxtyping import Float, Complex
from typing import Optional


import cryojax.coordinates as cc
from ._average import compute_binned_radial_average  # keep import inside jit-safe scope

# -- This stays outside the JIT trace
#def _compute_frequency_bins(
#    shape: tuple[int, ...],
#    pixel_size: float,
#    minimum_frequency: Optional[float] = None,
#    maximum_frequency: Optional[float] = None,
#) -> Float[Array, "n_bins"]:
#    q_min = 0.0 if minimum_frequency is None else minimum_frequency
#    q_max = (jnp.sqrt(2) / (pixel_size * 2.0)) if maximum_frequency is None else maximum_frequency
#    q_step = 1.0 / (pixel_size * max(shape))
#    num_bins = 1 + jnp.array(shape[0]/2,int) 
#    print(num_bins)
#    return jnp.linspace(q_min, q_max, num_bins)


# -- JIT-compatible inner function
@eqx.filter_jit
def compute_radially_averaged_powerspectrum(
    fourier_image: Complex[Array, "y x"] | Complex[Array, "z y x"],
    pixel_size: float = 1.0,
    *,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
) -> tuple[Float[Array, "n_bins"], Float[Array, "n_bins"]]:

    import cryojax.coordinates  as cc

    radial_frequency_grid = cc.make_radial_frequency_grid(fourier_image.shape,grid_spacing=pixel_size)
    start = 0 
    stop = jnp.sqrt(2) / (pixel_size*2.0) 
    frequency_bins = jnp.linspace(start, stop, int(fourier_image.shape[0]/2)+1)
    squared_fourier_amplitudes = jnp.real(fourier_image * jnp.conjugate(fourier_image))

    spectrum = compute_binned_radial_average(squared_fourier_amplitudes, radial_frequency_grid, frequency_bins)

    return spectrum, frequency_bins

