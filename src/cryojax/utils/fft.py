"""
Routines to compute FFTs.
"""

__all__ = ["ifftn", "irfftn", "fftn", "rfftn"]

from typing import Any, Union, Optional

import jax.numpy as jnp

from ..typing import RealImage, RealVolume, Image, Volume


def irfftn(
    ft: Union[Image, Volume], **kwargs: Any
) -> Union[RealImage, RealVolume]:
    """
    Convenience wrapper for ``cryojax.utils.fft.ifftn``
    for ``real = True``.
    """
    return ifftn(ft, real=True, **kwargs)


def ifftn(
    ft: Union[Image, Volume], real: bool = False, **kwargs: Any
) -> Union[Image, Volume]:
    """
    Helper routine to match the inverse fourier transform
    to the output of a type 1 non-uniform FFT with jax-finufft.

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
    Helper routine to match the fourier transform of an array
    with the output of a type 1 non-uniform FFT with jax-finufft.

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


def irfftn(
    ft: Union[Image, Volume],
    s: Optional[tuple[int, ...]] = None,
    **kwargs: Any
) -> Union[Image, Volume]:
    """
    Helper routine to compute the inverse fourier transform
    from real input.

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
    dim = len(ft.shape)
    ift = jnp.fft.fftshift(
        jnp.fft.irfftn(ft, s=s, **kwargs), axes=tuple(range(dim - 1))
    )

    return ift


def rfftn(ift: Union[Image, Volume], **kwargs: Any) -> Union[Image, Volume]:
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
    dim = len(ift.shape)
    ft = jnp.fft.rfftn(
        jnp.fft.ifftshift(ift, axes=tuple(range(dim - 1))), **kwargs
    )

    return ft
