"""
Routines to compute FFTs.
"""

__all__ = ["nufft", "ifft", "fftfreqs"]

import jax

# import tensorflow as tf
import jax.numpy as jnp
import numpy as np

# import tensorflow_nufft as tfft
# from jax.experimental import jax2tf
from jax_finufft import nufft1
from functools import partial
from typing import Tuple

from ..types import Array, ArrayLike


# @partial(jax.jit, static_argnums=(0, 4))
def nufft(
    shape: Tuple[int, int, int],
    density: Array,
    coords: Array,
    box_size: Array,
    eps: float = 1e-6,
) -> Array:
    r"""
    Helper routine to compute a non-uniform FFT for a 3D
    point cloud. Mask out points that lie out of bounds.

    .. warning::
        If any values in ``coords`` lies out of bounds of
        :math:`$(-3\pi, 3\pi]$`, this method will crash.
        This means that ``density`` cannot be
        arbitrarily cropped and translated out of frame,
        rather only to a certain extent.

    Arguments
    ---------
    density : shape `(N,)`
        Density point cloud over which to compute
        the fourier transform.
    coords : shape `(N, 3)`
        Coordinate system for density cloud.
    box_size : shape `(3,)`
        3D cartesian box that ``coords`` lies in.
    shape :
        Desired output shape of the transform.
    eps :
        Precision of the non-uniform FFT. See
        `finufft <https://finufft.readthedocs.io/en/latest/>`_
        for more detail.
    Return
    ------
    ft : Array, shape ``shape``
        Fourier transform.
    """
    complex_density = density.astype(complex)
    periodic_coords = 2 * jnp.pi * coords / box_size
    x, y = periodic_coords[:, 0], periodic_coords[:, 1]
    masked_density = jax.vmap(_mask_density)(x, y, complex_density)
    # _nufft1 = jax2tf.call_tf(_tf_nufft1, output_shape_dtype=jax.ShapeDtypeStruct(shape, masked_density.dtype))
    # ft = _nufft1(masked_density, periodic_coords, shape, eps)
    x, y, z = periodic_coords.T
    ft = nufft1(shape, masked_density, x, y, z, eps=eps)

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
    ift = jnp.fft.fftshift(jnp.fft.ifftn(np.fft.ifftshift(ft), **kwargs))

    return ift.real


def fft(ift: Array, **kwargs) -> Array:
    """
    Helper routine to compute the fourier transform of an array
    to match the output of a type 1 non-uniform FFT.

    Arguments
    ---------
    ift :
        Array in real space. Assumes that the zero
        frequency component is in the center.
    Returns
    -------
    ft :
        Fourier transform of array.
    """
    ft = jnp.fft.fftshift(jnp.fft.fftn(np.fft.ifftshift(ift), **kwargs))

    return ft


def fftfreqs(
    shape: tuple[int, ...], pixel_size: float, real: bool = False
) -> ArrayLike:
    """
    Create a radial coordinate system on a grid.
    This can be used for real and fourier space
    calculations. If used for fourier space, the
    zero-frequency component is in the center.

    Arguments
    ---------
    shape :
        Shape of the voxel grid, with
        ``ndim = len(shape)``.
    pixel_size :
        Image pixel size.
    real :
        Choose whether to create coordinate system
        in real or fourier space.


    Returns
    -------
    rcoords : shape `(*shape, ndim)`
        2D or 3D cartesian coordinate system with
        zero in the center.
    """
    fftfreq = (
        lambda s: np.fft.fftfreq(s, 1 / pixel_size) * s
        if real
        else np.fft.fftfreq(s, pixel_size)
    )
    rcoords1D = [np.fft.fftshift(fftfreq(s)) for s in shape]

    rcoords = np.stack(np.meshgrid(*rcoords1D, indexing="ij"), axis=-1)

    return rcoords


# def _tf_nufft1(source, points, shape, tol):
#    """
#    Wrapper for type-1 non-uniform FFT
#    from tensorflow-nufft.
#    """
#    return tfft.nufft(
#        source,
#        points,
#        grid_shape=shape,
#        transform_type="type_1",
#        tol=tol.numpy(),
#    )


def _mask_density(x: Array, y: Array, density: Array) -> Array:
    """
    Use a boolean mask to set density values out of
    bounds to zero. The mask is ``True`` for
    all points outside the :math:`[0, 2\pi)` periodic
    domain, and False otherwise.
    """
    x_mask = jnp.logical_or(x < -jnp.pi, x >= jnp.pi)
    y_mask = jnp.logical_or(y < -jnp.pi, y >= jnp.pi)
    mask = jnp.logical_or(x_mask, y_mask)
    masked_density = jnp.where(mask, complex(0.0), density)

    return masked_density
