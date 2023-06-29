"""
Routines to compute FFTs.
"""

__all__ = ["nufft", "ifft"]

import jax.numpy as jnp
from functools import partial
from jax import jit
from jax_finufft import nufft1
from typing import Tuple
from jax_2dtm.types import Array


def nufft(
    density: Array, coords: Array, box: Array, shape: Tuple[int, int, int]
, **kwargs) -> Array:
    """
    Helper routine to compute a non-uniform FFT for a 3D
    point cloud. Mask out points that lie out of bounds.

    Arguments
    ---------

    Return
    ------

    """
    complex_density = density.astype(complex)
    x, y, z = (2 * jnp.pi * coords / box + jnp.pi).T
    x_mask = jnp.logical_or(x < 0, x >= 2*jnp.pi)
    y_mask = jnp.logical_or(y < 0, y >= 2*jnp.pi)
    mask = jnp.logical_not(jnp.logical_or(x_mask, y_mask))
    ft = nufft1(shape, complex_density[mask], x[mask], y[mask], z[mask], **kwargs)

    return ft


def ifft(ft: Array, **kwargs) -> Array:
    """
    Helper routine to compute the inverse fourier transform
    from the output of a type 1 non-uniform FFT. Assume that
    the imaginary part of the inverse transform is zero.

    Arguments
    ---------
    ft :
        Fourier transform array. Assumes that the zero
        frequency component is in the center.
    Returns
    -------
    ift :
        Inverse fourier transform.
    """
    ift = jnp.fft.ifftn(jnp.fft.ifftshift(ft), **kwargs)

    return ift.real
