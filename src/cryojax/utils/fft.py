"""
Routines to compute FFTs.
"""

__all__ = ["ifft", "irfft", "fft"]

from typing import Any

import jax.numpy as jnp

from ..core import Array


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
    ift = jnp.fft.fftshift(jnp.fft.ifftn(ft, **kwargs))

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
    ft = jnp.fft.fftn(jnp.fft.ifftshift(ift), **kwargs)

    return ft
