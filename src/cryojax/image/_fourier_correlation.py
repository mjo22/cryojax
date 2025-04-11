"""
Helper routines to compute power spectra.
"""

from typing import Union, Optional

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ._average import compute_binned_radial_average
import cryojax.coordinates as cc

def _handle_fourier_transform (
    image_1: Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"] | Float[Array, "y_dim x_dim z_dim"] | Complex[Array, "y_dim x_dim z_dim"],
    image_2: Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"] | Float[Array, "y_dim x_dim z_dim"] | Complex[Array, "y_dim x_dim z_dim"],
    ):
    
    if jnp.iscomplexobj(image_1):
        fourier_image_1 = image_1
    else:
        fourier_image_1 = jnp.fft.fftshift(jnp.fft.fftn(image_1))
    if jnp.iscomplexobj(image_2):
        fourier_image_2 = image_2
    else:
        fourier_image_2 = jnp.fft.fftshift(jnp.fft.fftn(image_2))
    return fourier_image_1, fourier_image_2

def compute_radial_fourier_correlation(
    # TODO make jit compatible.
    image_1: Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"] | Float[Array, "y_dim x_dim z_dim"] | Complex[Array, "y_dim x_dim z_dim"],
    image_2: Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"] | Float[Array, "y_dim x_dim z_dim"] | Complex[Array, "y_dim x_dim z_dim"],
    pixel_size: Float[Array, ""] | float = 1.0,
    #default threshold is 0.5 for two 'known' volumes according to the half-bit criterion.
    # However, for half maps derived from ab initio refinemnts. The threshold is 0.143 by convention.
    # todo: add van heel criterion.
    threshold: Float | float = 0.5, 
                                               
    minimum_frequency: Optional[Float[Array, ""] | float] = None,
    maximum_frequency: Optional[Float[Array, ""] | float] = None,
    real_space_mask: Optional[Float[Array, "y_dim x_dim z_dim"]] = None,
    fourier_space_mask: Optional[Float[Array, "y_dim x_dim z_dim"]] = None
   
) -> tuple[Float[Array, "n_bins"], Float]:

    """compute the fourier ring correlation or fourier shell correlation for two images or two voxel maps.

    **Arguments:**

    `image_1`:
        An image in real or fourier space.
    `image_2`:
        An image in real or fourier space.
    `pixel_size`:
        The pixel size of `the images`.
    `minimum_frequency`:
        Minimum frequency bin. By default, `0.0`.
    `maximum_frequency`:
        Maximum frequency bin. By default, `1 / (2 * pixel_size)` nyquist frequency.
    `mask`:
        mask for input volumes.

    **Returns:**

    A tuple where the first element is the coefficients of the FSC/FRC as a function of
    frequency and the second element is the last value after which the corrleation
    drops below the threshold value.
    """

    # check that maps have the same dimension. 
    if image_1.shape != image_2.shape:
        raise ValueError('Calculating fourier correlations for two images or volumes is only supported when they have the same shape.')

    if real_space_mask is not None:
        if jnp.any(real_space_mask > 1) or jnp.any(real_space_mask)  < 0:
            raise ValueError('mask values are outside valid range [0,1].')
        
    if fourier_space_mask is not None:
        if jnp.any(fourier_space_mask) < 0 or jnp.any(fourier_space_mask) > 1:
            raise ValueError('mask values are outside valid range [0,1].')

    # choose which mask to apply if any.
    if real_space_mask is None and fourier_space_mask is None:
        #no mask applied
        fourier_image_1, fourier_image_2 = _handle_fourier_transform(image_1, image_2)
    elif real_space_mask is not None and fourier_space_mask is None:
        # real space mask applied
        if real_space_mask.shape != image_1.shape:
            raise ValueError('mask and map must have same dimensions')
        image_1 = real_space_mask*image_1
        image_2 = real_space_mask*image_2
        fourier_image_1, fourier_image_2 = _handle_fourier_transform(image_1, image_2)
    elif real_space_mask is None and fourier_space_mask is not None:
        if fourier_space_mask.shape != image_1.shape:
            raise ValueError('mask and map must have same dimensions')
        fourier_image_1, fourier_image_2 = _handle_fourier_transform(image_1, image_2)

        # fourier mask applied
        fourier_image_1 = fourier_space_mask*fourier_image_1
        fourier_image_2 = fourier_space_mask*fourier_image_2
    else:
        raise ValueError('Specifying both a real space mask and a fourier mask is not supported.')

    correlation_voxel_map = (fourier_image_1 * jnp.conjugate(fourier_image_2))
    normalisation_voxel_map = jnp.sqrt(jnp.abs(fourier_image_1)**2 * jnp.abs(fourier_image_2)**2)

    frequency_grid = cc.make_frequency_grid(jnp.array(image_1.shape), pixel_size, get_rfftfreqs=False)
    radial_frequency_grid = jnp.fft.ifftshift(jnp.linalg.norm(frequency_grid, axis=-1))

    # Compute bins
    q_min = 0.0 if minimum_frequency is None else minimum_frequency
    q_max = (
        # set maximum at nyquist, ignore corners by default.
        1 / (pixel_size * 2.0)
        if maximum_frequency is None
        else maximum_frequency
    )
    
    q_step = 1.0 / (pixel_size * max(*fourier_image_1.shape))
    frequency_bins = jnp.linspace(q_min, q_max, 1 + int((q_max - q_min) / q_step))

     
    # Compute radially averaged FSC as a 1D profile
    FSC_curve = jnp.real(compute_binned_radial_average(
        correlation_voxel_map/normalisation_voxel_map , radial_frequency_grid, frequency_bins
    ))

    #remove nans and infs.
    FSC_curve = jnp.where(jnp.isnan(FSC_curve) | jnp.isinf(FSC_curve), 0.0, FSC_curve)

    # find threshold where radial correlation average drops below specified threshold.
    threshold_crossing_index = -1
    for i in jnp.arange(len(FSC_curve)):
        if FSC_curve[i] <= threshold:
            # max between index and 0 so that we handle the case where no good correlation exists, 
            # we will return the DC component in that case
            threshold_crossing_index = jnp.max(jnp.array((i, 0)))
            break

    # return the frequency where the threshold crosses over the threshold 
    frequency_threshold = frequency_bins[threshold_crossing_index]
    return FSC_curve, frequency_threshold, frequency_bins
