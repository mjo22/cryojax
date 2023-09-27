"""
Routines to compute FFTs.
"""

__all__ = ["ifftn", "irfftn", "fftn"]

from typing import Any, Union

import jax.numpy as jnp

from ..core import Image, Volume


def irfftn(ft: Union[Image, Volume], **kwargs: Any) -> Union[Image, Volume]:
    """
    Convenience wrapper for ``cryojax.utils.fft.ifftn``
    for ``real = True``.
    """
    return ifftn(ft, real=True, **kwargs)


def ifftn(
    ft: Union[Image, Volume], real: bool = False, **kwargs: Any
) -> Union[Image, Volume]:
    """
    Helper routine to compute the inverse fourier transform
    from the output of a type 1 non-uniform FFT. Assume that
    the imaginary part of the inverse transform is zero.

    Arguments
    ---------
    ft :
        Fourier transform array. Assumes that the zero
        frequency component is in the center.
    real : `bool`
        If ``True``, then ``ft`` is the Fourier transform
        of a real-valued function.
    Returns
    -------
    ift :
        Inverse fourier transform.
    """
    ift = jnp.fft.fftshift(jnp.fft.ifftn(ft, **kwargs))

    if real:
        return ift.real
    else:
        return ift


def fftn(ift: Union[Image, Volume], **kwargs: Any) -> Union[Image, Volume]:
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
    ft = jnp.fft.fftn(jnp.fft.ifftshift(ift), **kwargs)

    return ft
