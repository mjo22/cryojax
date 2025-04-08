"""
Helper routines to compute power spectra.
"""

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ._average import compute_binned_radial_average
import cryojax.coordinates as cc

def shell_correlation(
    shell1: Complex[Array, "y_dim x_dim"],
    shell2: Complex[Array, "y_dim x_dim"]
    ) -> Float:
    pass

def compute_radial_fourier_correlation(
    image_1: Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"],
    image_2: Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"],
    pixel_size: Float[Array, ""] | float = 1.0,
    threshold: Float[Array, ""] | float = 0.5, #default is 0.5 for two 'known' volumes, change for refinement to 0.143
                                               #todo: add van heel criterion.
    *,
    minimum_frequency: Optional[Float[Array, ""] | float] = None,
    maximum_frequency: Optional[Float[Array, ""] | float] = None,
) -> tuple[Float[Array, " n_bins"], Float]:
    """compute the fourier ring correlation or fourier shell correlation for two images or two maps.

    **Arguments:**

    `image_1`:
        An image in real or fourier space.
    `image_2`:
        An image in real or fourier space.
    `pixel_size`:
        The pixel size of `the images`.
    `minimum_frequency`:
        Minimum frequency bin. By default, `0.0`.
    `minimum_frequency`:
        Maximum frequency bin. By default, `1 / (2 * pixel_size)` nyquist frequency.

    **Returns:**

    A tuple where the first element is the coefficients of the FSC/FRC as a function of 
    frequency and the second element is the last value after which the corrleation
    drops below the threshold value. 
    """
    if image_1.shape != image_2.shape:
        raise ValueError('Calculating fourier correlations for two images or volumes is only supported when they have the same shape.')

    if type(image_1) == Complex:
        fourier_image_1 = image_1
    else:
        fourier_image_1 = jnp.fft.fftshift(jnp.fft.fftn(image_1))
    if type(image_2) == Complex:
        fourier_image_2 = image_2
    else:
        fourier_image_2 = jnp.fft.fftshift(jnp.fft.fftn(image_2))


    grid = cc.make_frequency_grid(jnp.array(image_1.shape), pixel_size,get_rfftfreqs=False)

    correlation_voxel_map = (fourier_image_1 * jnp.conjugate(fourier_image_2))
    normalisation_voxel_map = jnp.sqrt(jnp.abs(fourier_image_1)**2 * jnp.abs(fourier_image_2)**2)


    frequency_grid = cc.make_frequency_grid(jnp.array(image_1.shape), pixel_size,get_rfftfreqs=False)
    radial_frequency_grid = jnp.fft.ifftshift(jnp.linalg.norm(frequency_grid, axis=-1))



    # Compute bins
    q_min = 0.0 if minimum_frequency is None else minimum_frequency
    q_max = (
        # set maximum at nyquist
        1 / (pixel_size * 2.0)
        if maximum_frequency is None
        else maximum_frequency
    )
    
    q_step = 1.0 / (pixel_size * max(*fourier_image_1.shape))
    frequency_bins = jnp.linspace(q_min, q_max, 1 + int((q_max - q_min) / q_step))

     
    # Compute radially averaged FSC as a 1D profile
        
    correlation_curve = compute_binned_radial_average(
        correlation_voxel_map , radial_frequency_grid, frequency_bins
    )

    normalisation_curve = compute_binned_radial_average(
        normalisation_voxel_map , radial_frequency_grid, frequency_bins
    )



    FSC_curve = jnp.real(correlation_curve / normalisation_curve)
    #remove nans and infinities which could happen if there are 0s.
    FSC_curve = jnp.where(jnp.isnan(FSC_curve) | jnp.isinf(FSC_curve), 0.0, FSC_curve)

    # return the index imeditaely before the threshold was crossed
    # this appears to match the cryosparc conventions.
    threshold_crossing_index = jnp.argmax(FSC_curve <= threshold)
    frequency_threshold = frequency_bins[jnp.maximum(threshold_crossing_index - 1, 0)]       

    return FSC_curve, frequency_threshold, frequency_bins
