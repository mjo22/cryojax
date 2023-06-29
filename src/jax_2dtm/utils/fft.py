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


@partial(jit, static_argnames=["shape"])
def nufft(density: Array, coords: Array, box: Array, shape: Tuple[int, int, int]) -> Array:
    """
    Helper routine to compute a non-uniform FFT for a 3D
    point cloud.

    Arguments
    ---------

    Return
    ------

    """
    complex_density = density.astype(complex)
    x, y, z = (2*jnp.pi * coords / box - jnp.pi).T
    ft = nufft1(shape, complex_density, x, y, z)

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
