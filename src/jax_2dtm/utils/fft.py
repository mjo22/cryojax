"""
Routines to compute FFTs.
"""

__all__ = ["nufft", "ifft"]

import jax
import tensorflow as tf
import jax.numpy as jnp
import tensorflow_nufft as tfft
from jax.experimental import jax2tf
from jax_finufft import nufft1
from functools import partial
from typing import Tuple
from jax_2dtm.types import Array


# @partial(jax.jit, static_argnums=(3, 4))
def nufft(
    density: Array,
    coords: Array,
    box: Array,
    shape: Tuple[int, int, int],
    eps: float = 1e-6,
) -> Array:
    """
    Helper routine to compute a non-uniform FFT for a 3D
    point cloud. Mask out points that lie out of bounds.

    Arguments
    ---------

    Return
    ------

    """
    complex_density = density.astype(complex)
    periodic_coords = 2 * jnp.pi * coords / box + jnp.pi  # .T
    x, y = periodic_coords[:, 0], periodic_coords[:, 1]
    masked_density = jax.vmap(_mask_density)(x, y, complex_density)
    # _nufft1 = jax2tf.call_tf(_tf_nufft1, output_shape_dtype=jax.ShapeDtypeStruct(shape, masked_density.dtype))
    # ft = _nufft1(masked_density, periodic_coords, shape, eps)
    x, y, z = periodic_coords.T
    ft = _nufft1(shape, masked_density, x, y, z, eps=eps)

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


# @partial(jax.jit, static_argnums=(0, 5))
def _nufft1(shape, density, x, y, z, eps=1e-6):
    """
    Jitted type-1 non-uniform FFT from jax_finufft. This
    does not currently support GPU usage.
    """
    return nufft1(shape, density, x, y, z, eps=eps)


def _tf_nufft1(source, points, shape, tol):
    """
    Wrapper for type-1 non-uniform FFT
    from tensorflow-nufft.
    """
    return tfft.nufft(
        source,
        points,
        grid_shape=shape,
        transform_type="type_1",
        tol=tol.numpy(),
    )


def _mask_density(x: Array, y: Array, density: Array) -> Array:
    """
    Use a boolean mask to set density values out of
    bounds to zero. The mask is ``True`` for
    all points outside the :math:`[0, 2\pi)` periodic
    domain, and False otherwise.
    """
    x_mask = jnp.logical_or(x < 0, x >= 2 * jnp.pi)
    y_mask = jnp.logical_or(y < 0, y >= 2 * jnp.pi)
    mask = jnp.logical_or(x_mask, y_mask)
    masked_density = jnp.where(mask, complex(0.0), density)

    return masked_density
