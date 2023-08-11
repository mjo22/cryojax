"""
Routines to compute FFTs.
"""

__all__ = [
    "ifft",
    "irfft",
    "fft",
    "fftfreqs",
    "fftfreqs1d",
    "cartesian_to_polar",
]

from typing import Any
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from ..core import Array, ArrayLike


def irfft(ft: Array, **kwargs: Any) -> Array:
    """
    Convenience wrapper for ``cryojax.utils.fft.ifft``
    for ``real = True``.
    """
    return ifft(ft, real=True, **kwargs)


def ifft(ft: Array, real: bool = False, **kwargs: Any) -> Array:
    """
    Helper routine to compute the inverse fourier transform
    from the output of a type 1 non-uniform FFT. Assume that
    the imaginary part of the inverse transform is zero.

    Arguments
    ---------
    ft : `Array`
        Fourier transform array. Assumes that the zero
        frequency component is in the center.
    real : `bool`
        If ``True``, then ``ft`` is the Fourier transform
        of a real-valued function.
    Returns
    -------
    ift : `Array`
        Inverse fourier transform.
    """
    ift = jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(ft), **kwargs))

    if real:
        return ift.real
    else:
        return ift


def fft(ift: Array, **kwargs: Any) -> Array:
    """
    Helper routine to compute the fourier transform of an array
    to match the output of a type 1 non-uniform FFT.

    Arguments
    ---------
    ift : `Array`
        Array in real space. Assumes that the zero
        frequency component is in the center.
    Returns
    -------
    ft : `Array`
        Fourier transform of array.
    """
    ft = jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(ift), **kwargs))

    return ft


def fftfreqs(
    shape: tuple[int, ...], pixel_size: float = 1.0, real: bool = False
) -> ArrayLike:
    """
    Create a radial coordinate system on a grid.
    This can be used for real and fourier space
    calculations. If used for fourier space, the
    zero-frequency component is in the center.

    Arguments
    ---------
    shape : `tuple[int, ...]`
        Shape of the voxel grid, with
        ``ndim = len(shape)``.
    pixel_size : `float`
        Image pixel size.
    real : `bool`
        Choose whether to create coordinate system
        in real or fourier space.


    Returns
    -------
    k_coords : shape `(*shape, ndim)`
        2D or 3D cartesian coordinate system with
        zero in the center.
    """
    ndim = len(shape)
    if isinstance(pixel_size, (np.floating, float)):
        pixel_size = ndim * [pixel_size]
    else:
        pixel_size = list(pixel_size)
    assert len(pixel_size) == ndim
    k_coords1D = []
    for idx in range(ndim):
        k_coords1D.append(fftfreqs1d(shape[idx], pixel_size[idx], real))

    k_coords = np.stack(np.meshgrid(*k_coords1D, indexing="ij"), axis=-1)

    return k_coords


def fftfreqs1d(s: int, pixel_size: float, real: bool = False) -> ArrayLike:
    """One-dimensional coordinates in real or fourier space"""
    fftfreq = (
        lambda s: np.fft.fftfreq(s, 1 / pixel_size) * s
        if real
        else np.fft.fftfreq(s, pixel_size)
    )
    return np.fft.fftshift(fftfreq(s))


def cartesian_to_polar(freqs: ArrayLike, square: bool = False) -> Array:
    """
    Convert from cartesian to polar coordinates.

    Arguments
    ---------
    freqs : `ArrayLike`, shape `(N1, N2, 2)`
        The cartesian coordinate system.
    square : `bool`, optional
        If ``True``, return the square of the
        radial coordinate :math:`|r|^2`. Otherwise,
        return :math:`|r|`.
    """
    theta = jnp.arctan2(freqs[..., 0], freqs[..., 1])
    k_sqr = jnp.sum(jnp.square(freqs), axis=-1)
    if square:
        return k_sqr, theta
    else:
        kr = jnp.sqrt(k_sqr)
        return kr, theta
