"""
Helper routines to compute power spectra.
"""

from typing import Union, Optional

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float
import math
import equinox as eqx

from ._average import compute_binned_radial_average
import cryojax.coordinates as cc

@eqx.filter_jit
def compute_radial_fourier_correlation(
    image_1: Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"] | Float[Array, "y_dim x_dim z_dim"] | Complex[Array, "y_dim x_dim z_dim"],
    image_2: Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"] | Float[Array, "y_dim x_dim z_dim"] | Complex[Array, "y_dim x_dim z_dim"],
    pixel_size: Float[Array, ""] | float = 1.0,
    #default threshold is 0.5 for two 'known' volumes according to the half-bit criterion.
    # However, for half maps derived from ab initio refinemnts. The threshold is 0.143 by convention.
    # todo: add van heel criterion.
    do_fft: bool | bool = True,
    threshold: Float | float = 0.5,
) -> tuple[Float[Array, "n_bins"], Float]:

    """compute the fourier ring correlation or fourier shell correlation for two images or two voxel maps.

    **Arguments:**

    `image_1`:
        An image in real or fourier space.
    `image_2`:
        An image in real or fourier space.
    `pixel_size`:
        The pixel size of `the images`.
    `do_fft`:
        Choose whether to transform the volumes/images into fourier space.
    `threshold`:
        The threshold at which to draw the distinction between input maps.

    **Returns:**

    `frequency_bins`: The array of inverse frequencies for which we have calcualted the correlations
    `frequency_threshold`: The inverse frequnecy at which the correlation dropped below the specified threshold.
    `correlation_curve`: The value of the calculated radial fourier correlations. In 2D this is FRC and 3D this is FSC.
    """
    if do_fft == True:
        fourier_image_1 = jnp.fft.fftshift(jnp.fft.fftn(image_1))
        fourier_image_2 = jnp.fft.fftshift(jnp.fft.fftn(image_2))
    else:
        fourier_image_1 = image_1
        fourier_image_2 = image_2
    correlation_voxel_map = (fourier_image_1 * jnp.conjugate(fourier_image_2))
    normalisation_voxel_map = jnp.sqrt(jnp.abs(fourier_image_1)**2 * jnp.abs(fourier_image_2)**2)
    

    radial_frequency_grid = jnp.fft.ifftshift(cc.make_radial_frequency_grid(image_1.shape, grid_spacing=pixel_size, get_rfftfreqs=False))

    # hard code constant to avoid problems in linspace shape size.
    # jnp.sqrt(2) will get traced sadly.

    # Compute bins
    start = 0
    stop = math.sqrt(2) / (2.0 * pixel_size)

    frequency_bins = jnp.linspace(start, stop, int(math.sqrt(2) * image_1.shape[0]/2) + 1)

    correlation_voxel_map / normalisation_voxel_map

    # Compute radially averaged FSC as a 1D profile
    correlation_curve = jnp.real(compute_binned_radial_average(
        correlation_voxel_map/normalisation_voxel_map , radial_frequency_grid, frequency_bins
    ))

    ## Code block which finds where the FSC drops below the specified threshold. 
    where_below_threshold = jnp.where(correlation_curve < threshold, 0, 1) # 0s when below, 1s, when above
    ## Find minimum index where we flip from 0 to 1
    where_is_crossing = jnp.diff(where_below_threshold)
    # ... make an array that has a value of its index when we have a crossing, and a dummy value otherwise
    arr_size = where_is_crossing.size
    arr_indices = jnp.arange(arr_size, dtype=int)
    dummy_index = arr_size + 100
    indices_at_0_to_1_flips = jnp.where(where_is_crossing == -1, arr_indices, dummy_index)
    # ... get minimum of array
    threshold_crossing_index = jnp.amin(indices_at_0_to_1_flips)+1
    threshold_crossing_index = eqx.error_if(threshold_crossing_index, threshold_crossing_index == dummy_index, "Error in calculating thershold. Check arrays for nans and infs.")
    frequency_threshold = frequency_bins[threshold_crossing_index]

    return frequency_bins, frequency_threshold, correlation_curve
